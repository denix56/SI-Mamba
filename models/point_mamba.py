import os.path
from typing import Union, Optional
import math
import random
from functools import partial

from scipy.optimize import linear_sum_assignment

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import List, Callable, Optional, Tuple

from utils import misc
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *
from timm.models.layers import trunc_normal_
from timm.models.layers import DropPath

# from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from pytorch3d.loss import chamfer_distance
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.mamba2 import Mamba2

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

# from knn_cuda import KNN
from .block import Block
from .build import MODELS

from pytorch3d.ops import knn_points, sample_farthest_points

from .new_layers import StochasticNeuralSortPermuter


class Encoder(nn.Module):  ## Embedding module
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2, 1))  # BG 256 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)  # BG 512 n
        feature = self.second_conv(feature)  # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)


class Group(nn.Module):  # FPS + KNN
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        # self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        # center = misc.fps(xyz, self.num_group)  # B G 3
        center = sample_farthest_points(points = xyz, K = self.num_group)
        center = center[0]

        idx = knn_points(center.cuda(), xyz.cuda(), K=self.group_size, return_sorted=False)
        ######idx = knn_points(center.cpu(), xyz.cpu(), K=self.group_size, return_sorted=False)       
        idx = idx.idx
        idx = idx.long()
        #########_, idx = self.knn(xyz, center)  # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        # normalize
        neighborhood_org = neighborhood
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center, neighborhood_org


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
        module,
        n_layer,
        initializer_range=0.02,  # Now only used for embedding layer.
        rescale_prenorm_residual=True,
        n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/âˆšN where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def create_block(
        d_model,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        drop_path=0.,
        device=None,
        dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
        drop_path=drop_path,
    )
    block.layer_idx = layer_idx
    return block


class MixerModel(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_layer: int,
            ssm_cfg=None,
            norm_epsilon: float = 1e-5,
            rms_norm: bool = False,
            initializer_cfg=None,
            fused_add_norm=False,
            residual_in_fp32=False,
            drop_out_in_block: int = 0.,
            drop_path: int = 0.1,
            device=None,
            dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        # self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    drop_path=drop_path,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_out_in_block = nn.Dropout(drop_out_in_block) if drop_out_in_block > 0. else nn.Identity()

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, input_ids, pos, inference_params=None):
        hidden_states = input_ids  # + pos
        residual = None
        hidden_states = hidden_states + pos
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
            hidden_states = self.drop_out_in_block(hidden_states)
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        return hidden_states


class MixerModel2(MixerModel):
    def forward(self, input_ids, pos, eigen_emb, inference_params=None):
        input_ids = input_ids + eigen_emb
        return super().forward(input_ids, pos, inference_params=inference_params)


class MixerModel_add(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_layer: int,
            ssm_cfg=None,
            norm_epsilon: float = 1e-5,
            rms_norm: bool = False,
            initializer_cfg=None,
            fused_add_norm=False,
            residual_in_fp32=False,
            drop_out_in_block: int = 0.,
            drop_path: int = 0.1,
            device=None,
            dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        # self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    drop_path=drop_path,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_out_in_block = nn.Dropout(drop_out_in_block) if drop_out_in_block > 0. else nn.Identity()

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def cross_merg(self, ys, vec_indices):
        ys_reshaped = ys.reshape(ys.shape[0], int(vec_indices.shape[-1]*2), -1, ys.shape[-1]).permute(0, 1, 3, 2)

        B, K, D, N = ys_reshaped.shape

        argsorted_vec_indices = torch.argsort(vec_indices, 1).permute(0, 2, 1)
        argsorted_vec_indices = argsorted_vec_indices.unsqueeze(2).expand(-1, -1, D, -1)
        ys_partial = ys_reshaped[:, :int(K/2), :, :].reshape(B, int(K/2), D, -1)

        result = torch.gather(ys_partial, dim=-1, index=argsorted_vec_indices)

        result_flip = ys_reshaped[:, int(K/2):, :, :].reshape(B, int(K/2), D, -1).flip(-1)
        result_flip = torch.gather(result_flip, dim=-1, index=argsorted_vec_indices)

        ys = result + result_flip

        ys_ = 0
        for i in range(ys.shape[1]):
            ys_ += ys[:, i]

        return ys_.permute(0, 2, 1)

    def sort_points_by_fiedler(self, points, fiedler_vector):

        B, N, _ = points.shape
        _, sorted_indices = torch.sort(fiedler_vector, dim=1)

        expanded_indices = sorted_indices.unsqueeze(-1).expand(-1, -1, 384)

        sorted_points = torch.gather(points, 1, expanded_indices)

        return sorted_points

    def forward(self, input_ids, pos, top_k_eigenvectors, N_k_top_eigenvectors, reverse, inference_params=None):
        hidden_states = input_ids  # + pos
        residual = None
        hidden_states = hidden_states + pos
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
            hidden_states = self.drop_out_in_block(hidden_states)

            ############### cross merg and then traverse
            _, sorted_indices = torch.sort(top_k_eigenvectors, dim=1)

            hidden_states_org = self.cross_merg(hidden_states, sorted_indices)

            for i in range (N_k_top_eigenvectors):
                sorted_group_input_tokens = self.sort_points_by_fiedler(hidden_states_org, top_k_eigenvectors[:, :, i])

                if (i != 0):
                    sorted_group_input_tokens = torch.cat((sorted_group_input_tokens_t, sorted_group_input_tokens), 1)

                sorted_group_input_tokens_t = sorted_group_input_tokens


            if (reverse == True):
                sorted_group_input_tokens_t_reverse = sorted_group_input_tokens_t.flip(1)
                hidden_states = torch.cat((sorted_group_input_tokens_t, sorted_group_input_tokens_t_reverse), 1)

        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))

        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        return hidden_states

@MODELS.register_module()
class PointMamba(nn.Module):
    def __init__(self, config, **kwargs):
        super(PointMamba, self).__init__()
        self.config = config

        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.cls_dim = config.cls_dim

        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        self.use_cls_token = False if not hasattr(self.config, "use_cls_token") else self.config.use_cls_token
        self.drop_path = 0. if not hasattr(self.config, "drop_path") else self.config.drop_path
        self.rms_norm = False if not hasattr(self.config, "rms_norm") else self.config.rms_norm
        self.drop_out_in_block = 0. if not hasattr(self.config, "drop_out_in_block") else self.config.drop_out_in_block

        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
            self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))
            trunc_normal_(self.cls_token, std=.02)
            trunc_normal_(self.cls_pos, std=.02)

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        self.add_after_layer = config.add_after_layer

        if (self.add_after_layer):

            self.blocks = MixerModel_add(d_model=self.trans_dim,
                                         n_layer=self.depth,
                                         rms_norm=self.rms_norm,
                                         drop_out_in_block=self.drop_out_in_block,
                                         drop_path=self.drop_path)
        else:

            self.blocks = MixerModel(d_model=self.trans_dim,
                                     n_layer=self.depth,
                                     rms_norm=self.rms_norm,
                                     drop_out_in_block=self.drop_out_in_block,
                                     drop_path=self.drop_path)

        self.norm = nn.LayerNorm(self.trans_dim)

        self.HEAD_CHANEL = 1
        if self.use_cls_token:
            self.HEAD_CHANEL += 1

        self.cls_head_finetune = nn.Sequential(
            nn.Linear(self.trans_dim * self.HEAD_CHANEL, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, self.cls_dim)
        )

        self.build_loss_func()

        self.drop_out = nn.Dropout(config.drop_out) if "drop_out" in config else nn.Dropout(0)

        ########## Settings for our method
        self.method = config.method
        self.reverse = config.reverse
        self.reverse_2 = config.reverse_2
        self.reverse_3 = config.reverse_3
        self.k_top_eigenvectors = config.k_top_eigenvectors
        self.smallest = config.smallest
        self.knn_graph = config.knn_graph
        self.symmetric = config.symmetric
        self.self_loop = config.self_loop
        self.alpha = config.alpha
        self.binary = config.binary
        self.matrix = config.matrix

        self.eigen_embed = nn.Sequential(
            nn.Linear(2, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        self.logit_blocks = MixerModel2(d_model=self.trans_dim,
                                        n_layer=3,
                                        rms_norm=self.config.rms_norm)

        self.logit_norm = nn.LayerNorm(self.trans_dim)
        #
        head_dim = (self.trans_dim)
        assert self.trans_dim >= self.k_top_eigenvectors
        self.logit_head = nn.Sequential(nn.Linear(head_dim, self.trans_dim),
                                        nn.LayerNorm(self.trans_dim),
                                        nn.GELU(),
                                        nn.Linear(self.trans_dim, 1))
        self.logit_head2 = nn.Sequential(nn.Linear(head_dim, self.trans_dim),
                                         nn.LayerNorm(self.trans_dim),
                                         nn.GELU(),
                                         nn.Linear(self.trans_dim, 1))


        # self.logit_head_1 = nn.Linear(head_dim, self.trans_dim)
        # self.logit_head_norm = nn.LayerNorm(self.trans_dim)
        # self.logit_head_2 = nn.Linear(self.trans_dim, 1)
        #self.skip = nn.Parameter(torch.ones(self.k_top_eigenvectors))
        #self.skip2 = nn.Parameter(torch.ones(self.k_top_eigenvectors))
        #self.beta = nn.Parameter(torch.tensor(0.99))
        #self.beta2 = nn.Parameter(torch.tensor(0.99))

        self.permuter = StochasticNeuralSortPermuter()

        self.alpha = 0.99
        self.baseline = torch.tensor(torch.inf)

        scales = [0.02, 0.15, 0.5, 1.0]
        self.sgwt = GraphWaveletTransform(scales=scales, K=25, tight_frame=True, J=4)

        # for n, p in self.named_parameters():
        #     if 'logit_' in n:
        #         p.requires_grad = True
        #     else:
        #         p.requires_grad = False


    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss(reduction='none')

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path, map_location='cpu', weights_only=False)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder'):
                    base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log('missing_keys', logger='Mamba')
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger='Mamba'
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='Mamba')
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger='Mamba'
                )

            print_log(f'[Mamba] Successful Loading the ckpt from {bert_ckpt_path}', logger='Mamba')
        else:
            print_log('Training from scratch!!!', logger='Mamba')
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def create_graph_from_centers(self, points, k=5, alpha=1, symmetric = False, self_loop = False, binary = False):

        points = points.to('cuda')
        B, N, _ = points.shape

        # Compute pairwise Euclidean distances, shape (B, N, N), on GPU
        dist_matrix = torch.sqrt(torch.sum((points.unsqueeze(2) - points.unsqueeze(1)) ** 2, dim=-1))

        sigma = torch.mean(dist_matrix)

        # Find the k-nearest neighbors for each point, including itself
        distances, indices = torch.topk(-dist_matrix, k=k+1, largest=True, dim=-1)
        distances, indices = -distances[:, :, :], indices[:, :, :]

        if (self_loop):
            indices = indices[:, :, :]  # Remove self-loops
            distances = distances[:, :, :]  # Remove self-loops
        else:
            indices = indices[:, :, 1:]  # Remove self-loops
            distances = distances[:, :, 1:]  # Remove self-loops

        # Create a weighted adjacency matrix on GPU
        adjacency_matrix = torch.zeros(B, N, N, device='cuda')
        b_idx = torch.arange(B, device='cuda')[:, None, None]
        n_idx = torch.arange(N, device='cuda')[:, None]

        # Use gathered distances as weights
        if (self.alpha == 0):
            distances_new = torch.exp(-distances ** 2 / (2 * sigma ** 2))
        else:
            distances_new = torch.exp((-1) * alpha * (distances)**2)

        if (binary):
            adjacency_matrix[b_idx, n_idx, indices] = 1.
            if (symmetric):
                adjacency_matrix[b_idx, indices, n_idx] = 1.  # Ensure symmetry
        else:
            adjacency_matrix[b_idx, n_idx, indices] = distances_new
            if (symmetric):
                adjacency_matrix[b_idx, indices, n_idx] = distances_new  # Ensure symmetry

        return adjacency_matrix


    def create_graph_from_feature_space_gpu_weighted_adjacency(self, points, k=5, alpha=1, symmetric = False, self_loop = False, binary = False):
        """
        Create a graph from point cloud data in a feature space using k-nearest neighbors with Euclidean distances as weights, optimized for GPU.

        Parameters:
        - points: Tensor of shape (B, N, F) where B is the batch size,
                N is the number of points in the point cloud, and F is the feature dimensions.
                This tensor should already be on the GPU.
        - k: The number of nearest neighbors to consider for graph edges.

        Returns:
        - adjacency_matrix: Tensor of shape (B, N, N) representing the weighted adjacency matrix of the graph,
                            where weights are the Euclidean distances between points.
        """
        points = points.to('cuda')
        B, N, _ = points.shape

        # Compute pairwise Euclidean distances, shape (B, N, N), on GPU
        dist_matrix = torch.sqrt(torch.sum((points.unsqueeze(2) - points.unsqueeze(1)) ** 2, dim=-1))

        # Find the k-nearest neighbors for each point, including itself
        distances, indices = torch.topk(-dist_matrix, k=k+1, largest=True, dim=-1)
        #distances, indices = -distances[:, :, 1:], indices[:, :, 1:]  # Remove self-loops
        distances, indices = -distances[:, :, :], indices[:, :, :]

        if (self_loop):
            indices = indices[:, :, :]  # Remove self-loops
            distances = distances
        else:
            indices = indices[:, :, 1:]  # Remove self-loops
            distances = distances[..., 1:]

        # Create a weighted adjacency matrix on GPU
        adjacency_matrix = torch.zeros(B, N, N, device='cuda')
        b_idx = torch.arange(B, device='cuda')[:, None, None]
        n_idx = torch.arange(N, device='cuda')[:, None]

        # Use gathered distances as weights
        distances_new = torch.exp((-1) * alpha * (distances)**2)

        if (binary):
            adjacency_matrix[b_idx, n_idx, indices] = 1.
            if (symmetric):
                adjacency_matrix[b_idx, indices, n_idx] = 1.  # Ensure symmetry
        else:
            adjacency_matrix[b_idx, n_idx, indices] = distances_new
            if (symmetric):
                adjacency_matrix[b_idx, indices, n_idx] = distances_new  # Ensure symmetry

        # dist_matrix_new = torch.exp((-1) * alpha * (dist_matrix)**2)

        return adjacency_matrix

    def calc_top_k_eigenvalues_eigenvectors(self, adj_matrices, k, smallest):

        B, N, _ = adj_matrices.shape
        top_k_eigenvalues = torch.zeros((B, k)).cuda()
        top_k_eigenvectors = torch.zeros((B, N, k)).cuda()
        eigenvalues_l = torch.zeros((B, N)).cuda()
        eigenvectors_l = torch.zeros((B, N, N)).cuda()

        for i in range(B):
            # Extract the i-th adjacency matrix
            A = adj_matrices[i]

            # Ensure A is symmetric
            A = (A + A.t()) / 2

            # Compute the degree matrix D
            D = torch.diag(torch.sum(A, dim=1))

            # Compute D^-1
            D_inv = torch.diag(1.0 / (torch.diag(D) + 1e-6))

            # Perform Random-walk normalization: D^-1 * A
            I = torch.eye(N).cuda()
            normalized_A = I - torch.matmul(D_inv, A)

            eigenvalues, eigenvectors = torch.linalg.eigh(normalized_A)

            # Select the top k eigenvalues and corresponding eigenvectors
            if (smallest == False):
                # top_vals, top_indices = torch.topk(eigenvalues, k+1, largest=True, sorted=True)
                top_vals, top_indices = torch.topk(eigenvalues, k, largest=True, sorted=True)
                top_vecs = eigenvectors[:, top_indices]
            else:
                # top_vals, top_indices = torch.topk(eigenvalues, k+1, largest=False, sorted=True)
                top_vals, top_indices = torch.topk(eigenvalues, k, largest=False, sorted=True)
                top_vecs = eigenvectors[:, top_indices]

                # Store the results
            top_k_eigenvalues[i] = top_vals
            top_k_eigenvectors[i, :, :] = top_vecs

            eigenvalues_l[i] = eigenvalues
            eigenvectors_l[i, :, :] = eigenvectors

        return top_k_eigenvalues, top_k_eigenvectors, eigenvalues_l, eigenvectors_l


    def calc_top_k_eigenvalues_eigenvectors_symmetric(self, adj_matrices, k, smallest):

        B, N, _ = adj_matrices.shape
        top_k_eigenvalues = torch.zeros((B, k+1)).cuda()
        top_k_eigenvectors = torch.zeros((B, N, k+1)).cuda()
        eigenvalues_l = torch.zeros((B, N)).cuda()
        eigenvectors_l = torch.zeros((B, N, N)).cuda()

        for i in range(B):
            # Extract the i-th adjacency matrix
            A = adj_matrices[i]

            # Ensure A is symmetric
            A = (A + A.t()) / 2

            # Compute the degree matrix D
            D = torch.diag(torch.sum(A, dim=1))

            # Compute D^-1/2 (inverse square root of D)
            D_inv_sqrt = torch.diag(torch.pow(torch.diag(D), -0.5))

            # Symmetric normalization: D^-1/2 * A * D^-1/2
            normalized_A = torch.matmul(torch.matmul(D_inv_sqrt, A), D_inv_sqrt)

            # Identity matrix I
            I = torch.eye(A.size(0)).cuda()

            # Symmetric normalized Laplacian: L_sym = I - D^-1/2 * A * D^-1/2
            normalized_A = I - normalized_A

            #eigenvalues, eigenvectors = torch.linalg.eigh(normalized_A)
            eigenvalues, eigenvectors = torch.linalg.eigh(normalized_A)
            eigenvalues = eigenvalues.real
            eigenvectors = eigenvectors.real

            # Select the top k eigenvalues and corresponding eigenvectors
            if (smallest == False):
                top_vals, top_indices = torch.topk(eigenvalues, k+1, largest=True, sorted=True)
                top_vecs = eigenvectors[:, top_indices]
            else:
                top_vals, top_indices = torch.topk(eigenvalues, k+1, largest=False, sorted=True)
                top_vecs = eigenvectors[:, top_indices]

                # Store the results
            top_k_eigenvalues[i] = top_vals
            top_k_eigenvectors[i, :, :] = top_vecs

            eigenvalues_l[i] = eigenvalues
            eigenvectors_l[i, :, :] = eigenvectors

        return top_k_eigenvalues[:, 1:], top_k_eigenvectors[:, :, 1:], eigenvalues_l, eigenvectors_l


    def sort_points_by_fiedler(self, points, fiedler_vector):

        B, N, _ = points.shape
        _, sorted_indices = torch.sort(fiedler_vector, dim=1)

        expanded_indices = sorted_indices.unsqueeze(-1).expand(-1, -1, 384)

        sorted_points = torch.gather(points, 1, expanded_indices)

        return sorted_points


    def multilevel_travers(self, eigen_vectors, level):
        # Calculate mean for each batch element
        means = eigen_vectors.mean(dim=1, keepdim=True)                           ######### change for rebuttal
        # means, _= eigen_vectors.median(dim=1, keepdim=True)                           ######### change for rebuttal
        # Compute binaries for each batch element
        binaries = eigen_vectors >= means
        # Take the first 'level' rows and transpose
        binaries = binaries[:, :, :level]
        # Calculate integers
        num_bits = level
        powers_of_2 = 2 ** torch.arange(start=num_bits-1, end=-1, step=-1, device=eigen_vectors.device)
        integers = torch.sum(binaries * powers_of_2[None, None, :], dim=-1, keepdim=True)
        return integers.squeeze()

    def forward(self, pts, gt: torch.Tensor = None, tau: float = None, use_wavelets: bool = False, save_pts_dir: str = None, epoch: int = None):
        batch_size = pts.size(0)
        neighborhood, center, neighborhood_org = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)  # B G N
        pos = self.pos_embed(center)


        if (self.method == "MAMBA"):
            # reordering strategy
            center_x = center[:, :, 0].argsort(dim=-1)[:, :, None]
            center_y = center[:, :, 1].argsort(dim=-1)[:, :, None]
            center_z = center[:, :, 2].argsort(dim=-1)[:, :, None]
            group_input_tokens_x = group_input_tokens.gather(dim=1, index=torch.tile(center_x, (
                1, 1, group_input_tokens.shape[-1])))
            group_input_tokens_y = group_input_tokens.gather(dim=1, index=torch.tile(center_y, (
                1, 1, group_input_tokens.shape[-1])))
            group_input_tokens_z = group_input_tokens.gather(dim=1, index=torch.tile(center_z, (
                1, 1, group_input_tokens.shape[-1])))
            pos_x = pos.gather(dim=1, index=torch.tile(center_x, (1, 1, pos.shape[-1])))
            pos_y = pos.gather(dim=1, index=torch.tile(center_y, (1, 1, pos.shape[-1])))
            pos_z = pos.gather(dim=1, index=torch.tile(center_z, (1, 1, pos.shape[-1])))
            group_input_tokens = torch.cat([group_input_tokens_x, group_input_tokens_y, group_input_tokens_z],
                                           dim=1)
            pos = torch.cat([pos_x, pos_y, pos_z], dim=1)

        elif(self.method == "SAST"):

            # create graph
            # adjacency_matrix = self.create_graph_from_centers(center, self.knn_graph, self.alpha, self.symmetric, self.self_loop, self.binary)
            adjacency_matrix = self.create_graph_from_feature_space_gpu_weighted_adjacency(center, self.knn_graph, self.alpha, self.symmetric, self.self_loop, self.binary)

            orders = None
            if use_wavelets:
                laplacian = build_rw_laplacian(adjacency_matrix)
                coeffs = self.sgwt(center, laplacian)
                J = len(self.sgwt.scales) if self.sgwt.scales is not None else self.sgwt.J
                sig, orders = multi_scale_traversals(coeffs, F=3, J=coeffs.size(1), k=self.k_top_eigenvectors + 1, tau=tau)
                orders = torch.stack(orders, dim=1)

            # calculate eigenvectors and eigenvalues
            if (self.matrix == "laplacian"):
                top_k_eigenvalues, top_k_eigenvectors, eigenvalues, eigenvectors = self.calc_top_k_eigenvalues_eigenvectors(adjacency_matrix, k=self.k_top_eigenvectors, smallest = self.smallest)
            else:
                top_k_eigenvalues, top_k_eigenvectors, eigenvalues, eigenvectors = self.calc_top_k_eigenvalues_eigenvectors_symmetric(adjacency_matrix, k=self.k_top_eigenvectors, smallest = self.smallest)

            # ordering tokens
            for i in range (self.k_top_eigenvectors):
                sorted_group_input_tokens = self.sort_points_by_fiedler(group_input_tokens, top_k_eigenvectors[:, :, i])
                sorted_pos = self.sort_points_by_fiedler(pos, top_k_eigenvectors[:, :, i])

                if (i != 0):
                    sorted_group_input_tokens = torch.cat((sorted_group_input_tokens_t, sorted_group_input_tokens), 1)
                    sorted_pos = torch.cat((sorted_pos_t, sorted_pos), 1)

                sorted_group_input_tokens_t = sorted_group_input_tokens
                sorted_pos_t = sorted_pos

            policy = torch.zeros((batch_size,), device=center.device, dtype=center.dtype)
            # regularization = torch.tensor(0.0, device=center.device, dtype=center.dtype)
            if tau is not None:
                sorted_group_input_tokens_t_reverse = sorted_group_input_tokens_t.flip(1)
                sorted_group_input_tokens = torch.cat(
                    (sorted_group_input_tokens_t, sorted_group_input_tokens_t_reverse), 1)
                sorted_pos_t_reverse = sorted_pos_t.flip(1)
                sorted_pos = torch.cat((sorted_pos_t, sorted_pos_t_reverse), 1)
                sorted_vecs, _ = torch.sort(top_k_eigenvectors.transpose(1, 2), dim=2)
                eigen_emb = self.eigen_embed(torch.stack((-sorted_vecs,
                                                          top_k_eigenvalues.unsqueeze(-1).expand_as(sorted_vecs)),
                                                         dim=-1)).flatten(1, 2)
                eigen_emb = torch.cat((eigen_emb, eigen_emb.flip(1)), 1)

                logits_feats = self.logit_blocks(sorted_group_input_tokens.detach(), sorted_pos.detach(), eigen_emb)
                logits_feats = self.logit_norm(logits_feats)

                if self.reverse:
                    logits_feats1, logits_feats2 = torch.tensor_split(logits_feats, 2, dim=1)
                    logits_feats = logits_feats1 + logits_feats2.flip(1)

                logits_inner = self.logit_head(logits_feats).view(batch_size, -1, self.num_group)
                logits_outer = self.logit_head2(logits_feats.view(batch_size, -1, self.num_group, self.trans_dim).mean(dim=2)).squeeze(-1)

                seq_len = self.num_group
                P_inner = self.permuter(logits_inner.view(-1, seq_len), tau).view(batch_size, -1, seq_len, seq_len)
                P_outer = self.permuter(logits_outer, tau)
                perm_indices_outer = torch.argmax(P_outer, dim=2)
                perm_indices = torch.argmax(P_inner, dim=3) + perm_indices_outer.unsqueeze(-1) * seq_len
                perm_indices = perm_indices.view(batch_size, -1)

                #sorted_bool_masked_pos = torch.cat(sorted_bool_masked_pos_list, -1)
                #sorted_bool_masked_pos = sorted_bool_masked_pos.gather(dim=1, index=perm_indices)
                # sorted_bool_masked_pos = torch.bmm(P, sorted_bool_masked_pos.unsqueeze(-1).to(P.dtype)).squeeze(-1)
                # sorted_bool_masked_pos = sorted_bool_masked_pos > 0.5
                #sorted_bool_masked_pos_list = list(torch.split(sorted_bool_masked_pos, seq_len, dim=1))
                logits_inner = logits_inner.flatten(1, 2).gather(dim=1, index=perm_indices)
                logits_outer = logits_outer.gather(dim=1, index=perm_indices_outer)
                # logits_inner = logits_inner[~sorted_bool_masked_pos].reshape(batch_size, -1)
                #logits_outer = logits_inner.view(batch_size, -1, seq_len).mean(dim=2)
                #logits_outer = (self.beta2) * top_k_eigenvalues * self.skip2.unsqueeze(0) + (1-self.beta2) * logits_outer
                sorted_pos_t = sorted_pos_t.gather(dim=1, index=perm_indices[..., None].expand_as(sorted_pos_t))
                #sorted_pos_t = sorted_pos_t[~sorted_bool_masked_pos].reshape(batch_size, -1, C)
                #sorted_pos_mask_t = sorted_pos_full_t[sorted_bool_masked_pos].reshape(batch_size, -1, C)
                #sorted_neighborhood_t = sorted_neighborhood_t.gather(dim=1,
                #                                                     index=perm_indices[..., None, None].expand_as(
                #                                                         sorted_neighborhood_t))  # torch.einsum('bij,bjgd->bigd', P, sorted_neighborhood_t)
                #sorted_center_t = sorted_center_t.gather(dim=1,
                #                                         index=perm_indices[..., None].expand_as(sorted_center_t))

                policy = plackett_luce_dist(logits_inner.view(batch_size, -1, seq_len)).sum(dim=1) + plackett_luce_dist(
                    logits_outer)
                #regularization = self.calc_logit_dist_reg(sorted_center_t, logits_inner)
            else:
                ordered_vecs, _ = torch.sort(top_k_eigenvectors.transpose(1, 2), dim=-1)
                policy = plackett_luce_dist(-ordered_vecs).sum(-1) + plackett_luce_dist(top_k_eigenvalues)

            if orders is not None:
                orders = orders.unsqueeze(-1)

                sorted_group_input_tokens_t = group_input_tokens.unsqueeze(1).expand(-1, orders.size(1), -1, -1)
                sorted_group_input_tokens_t = sorted_group_input_tokens_t.gather(dim=2, index=orders.expand_as(
                    sorted_group_input_tokens_t))
                sorted_group_input_tokens_t = sorted_group_input_tokens_t.flatten(1, 2)

                sorted_pos_t = pos.unsqueeze(1).expand(-1, orders.size(1), -1, -1)
                sorted_pos_t = sorted_pos_t.gather(dim=2, index=orders.expand_as(sorted_pos_t))
                sorted_pos_t = sorted_pos_t.flatten(1, 2)

                if save_pts_dir is not None:
                    sorted_neighborhood = neighborhood.unsqueeze(1).expand(-1, orders.size(1), -1, -1, -1)
                    sorted_neighborhood = sorted_neighborhood.gather(dim=2, index=orders.unsqueeze(-1).expand_as(sorted_neighborhood))

                    sorted_center = center.unsqueeze(1).expand(-1, orders.size(1), -1, -1)
                    sorted_center = sorted_center.gather(dim=2, index=orders.expand_as(sorted_center))

                    np.savez_compressed(os.path.join(save_pts_dir, '{}.npz'.format(epoch)),
                                        center=sorted_center.cpu().numpy(),
                                        neighborhood=sorted_neighborhood.cpu().numpy(),
                                        orders=orders.cpu().numpy())


            if (self.reverse == True):
                sorted_group_input_tokens_t_reverse = sorted_group_input_tokens_t.flip(1)
                sorted_pos_t_reverse = sorted_pos_t.flip(1)
                sorted_group_input_tokens = torch.cat((sorted_group_input_tokens_t, sorted_group_input_tokens_t_reverse), 1)
                sorted_pos = torch.cat((sorted_pos_t, sorted_pos_t_reverse), 1)

                group_input_tokens = sorted_group_input_tokens.cuda()
                pos = sorted_pos.cuda()

            if (self.reverse_2 == True):

                sorted_group_input_tokens_t_reverse = sorted_group_input_tokens_t.flip(1)
                sorted_pos_t_reverse = sorted_pos_t.flip(1)

                B = sorted_group_input_tokens_t.shape[0]
                n_t = sorted_group_input_tokens_t.shape[1]
                n_c = sorted_group_input_tokens_t.shape[2]
                n_to = neighborhood.shape[1]

                sorted_group_input_tokens_t_reverse_2 = torch.zeros((B, n_t, n_c)).cuda()
                sorted_pos_t_reverse_2 = torch.zeros((B, n_t, n_c)).cuda()
                for i in range(self.k_top_eigenvectors):
                    if (i == 0):
                        sorted_group_input_tokens_t_reverse_2[:, (i*n_to):(i+1)*n_to, :] = sorted_group_input_tokens_t_reverse[:, (-1)*((i+1)*n_to):]
                        sorted_pos_t_reverse_2[:, (i*n_to):(i+1)*n_to, :] = sorted_pos_t_reverse[:, (-1)*((i+1)*n_to):]
                    else:
                        sorted_group_input_tokens_t_reverse_2[:, (i*n_to):(i+1)*n_to, :] = sorted_group_input_tokens_t_reverse[:, (-1)*((i+1)*n_to):(-1)*(i)*n_to]
                        sorted_pos_t_reverse_2[:, (i*n_to):(i+1)*n_to, :] = sorted_pos_t_reverse[:, (-1)*((i+1)*n_to):(-1)*(i)*n_to]

                sorted_group_input_tokens = torch.cat((sorted_group_input_tokens_t, sorted_group_input_tokens_t_reverse_2), 1)
                sorted_pos = torch.cat((sorted_pos_t, sorted_pos_t_reverse_2), 1)

                group_input_tokens = sorted_group_input_tokens.cuda()
                pos = sorted_pos.cuda()


            if (self.reverse_3 == True):

                sorted_group_input_tokens_t_reverse = sorted_group_input_tokens_t.flip(1)
                sorted_pos_t_reverse = sorted_pos_t.flip(1)

                B = sorted_group_input_tokens_t.shape[0]
                n_t = sorted_group_input_tokens_t.shape[1]
                n_c = sorted_group_input_tokens_t.shape[2]
                n_to = neighborhood.shape[1]

                sorted_group_input_tokens_t_reverse_2 = torch.zeros((B, n_t, n_c)).cuda()
                sorted_pos_t_reverse_2 = torch.zeros((B, n_t, n_c)).cuda()
                group_input_tokens = torch.zeros((B, int(n_t*2), n_c)).cuda()
                pos = torch.zeros((B, int(n_t*2), n_c)).cuda()

                for i in range(self.k_top_eigenvectors):
                    if (i == 0):
                        sorted_group_input_tokens_t_reverse_2[:, (i*n_to):(i+1)*n_to, :] = sorted_group_input_tokens_t_reverse[:, (-1)*((i+1)*n_to):]
                        sorted_pos_t_reverse_2[:, (i*n_to):(i+1)*n_to, :] = sorted_pos_t_reverse[:, (-1)*((i+1)*n_to):]
                    else:
                        sorted_group_input_tokens_t_reverse_2[:, (i*n_to):(i+1)*n_to, :] = sorted_group_input_tokens_t_reverse[:, (-1)*((i+1)*n_to):(-1)*(i)*n_to]
                        sorted_pos_t_reverse_2[:, (i*n_to):(i+1)*n_to, :] = sorted_pos_t_reverse[:, (-1)*((i+1)*n_to):(-1)*(i)*n_to]

                for i in range(self.k_top_eigenvectors):
                    if (i == 0):
                        group_input_tokens[:, (i*32):(i+1)*32, :] = sorted_group_input_tokens_t[:, (i*32):(i+1)*32, :]
                        group_input_tokens[:, (i+1)*32:(i+2)*32, :] = sorted_group_input_tokens_t_reverse_2[:, (i*32):(i+1)*32, :]
                        pos[:, (i*32):(i+1)*32, :] = sorted_pos_t[:, (i*32):(i+1)*32, :]
                        pos[:, (i+1)*32:(i+2)*32, :] = sorted_pos_t_reverse_2[:, (i*32):(i+1)*32, :]
                    else:
                        group_input_tokens[:, (i+1)*32:(i+2)*32, :] = sorted_group_input_tokens_t[:, (i*32):(i+1)*32, :]
                        group_input_tokens[:, (i+2)*32:(i+3)*32, :] = sorted_group_input_tokens_t_reverse_2[:, (i*32):(i+1)*32, :]

                        pos[:, (i+1)*32:(i+2)*32, :] = sorted_pos_t[:, (i*32):(i+1)*32, :]
                        pos[:, (i+2)*32:(i+3)*32, :] = sorted_pos_t_reverse_2[:, (i*32):(i+1)*32, :]

        elif (self.method == "HLT"):

            adjacency_matrix = self.create_graph_from_centers(center, self.knn_graph, self.alpha, self.symmetric, self.self_loop, self.binary)
            top_k_eigenvalues, top_k_eigenvectors, eigenvalues, eigenvectors = self.calc_top_k_eigenvalues_eigenvectors(adjacency_matrix, k = self.k_top_eigenvectors, smallest = self.smallest)

            integers = self.multilevel_travers(top_k_eigenvectors, self.k_top_eigenvectors)
            integers_before_random = integers

            random_value = torch.rand(integers.shape[0], integers.shape[1]).cuda()
            integers = integers + random_value

            integers_arg_sort = torch.argsort(integers, 1)

            integers_arg_sort_feature = integers_arg_sort.unsqueeze(-1).expand(-1, -1, 384)
            integers_arg_sort_center = integers_arg_sort.unsqueeze(-1).expand(-1, -1, 3)


            # Sort points based on the sorted indices
            sorted_group_input_tokens = torch.gather(group_input_tokens, 1, integers_arg_sort_feature)
            sorted_pos = torch.gather(pos, 1, integers_arg_sort_feature)
            sorted_center = torch.gather(center, 1, integers_arg_sort_center)

            number_of_groups = 2 ** (self.k_top_eigenvectors)
            number_of_devides = int(sorted_pos.shape[1] / (2 ** (self.k_top_eigenvectors)))


            sorted_group_input_tokens_zeros = torch.zeros(sorted_pos.shape[0], (sorted_pos.shape[1]*2), sorted_pos.shape[2]).cuda()
            sorted_pos_zeros = torch.zeros(sorted_pos.shape[0], (sorted_pos.shape[1]*2), sorted_pos.shape[2]).cuda()
            sorted_center_zeros = torch.zeros(sorted_pos.shape[0], (sorted_pos.shape[1]*2), 3).cuda()

            if (self.reverse == True):
                for i in range(number_of_devides):

                    if (i == 0):
                        sorted_group_input_tokens_zeros[:, (i*number_of_groups): (i+1)*number_of_groups] = sorted_group_input_tokens[:, (i*number_of_groups): (i+1)*number_of_groups]
                        sorted_pos_zeros[:, (i*number_of_groups): (i+1)*number_of_groups] = sorted_pos[:, (i*number_of_groups): (i+1)*number_of_groups]
                        sorted_center_zeros[:, (i*number_of_groups): (i+1)*number_of_groups] = sorted_center[:, (i*number_of_groups): (i+1)*number_of_groups]
                    else:
                        sorted_group_input_tokens_zeros[:, ((i+1)*number_of_groups): (i+2)*number_of_groups] = sorted_group_input_tokens[:, (i*number_of_groups): (i+1)*number_of_groups]
                        sorted_pos_zeros[:, ((i+1)*number_of_groups): (i+2)*number_of_groups] = sorted_pos[:, (i*number_of_groups): (i+1)*number_of_groups]
                        sorted_center_zeros[:, ((i+1)*number_of_groups): (i+2)*number_of_groups] = sorted_center[:, (i*number_of_groups): (i+1)*number_of_groups]

                    sorted_group_input_tokens_reverse = sorted_group_input_tokens[:, (i*number_of_groups): (i+1)*number_of_groups].flip(1)
                    sorted_pos_reverse = sorted_pos[:, (i*number_of_groups): (i+1)*number_of_groups].flip(1)
                    sorted_center_reverse = sorted_center[:, (i*number_of_groups): (i+1)*number_of_groups].flip(1)

                    if (i == 0):
                        sorted_group_input_tokens_zeros[:, ((i+1)*number_of_groups): (i+2)*number_of_groups] = sorted_group_input_tokens_reverse
                        sorted_pos_zeros[:, ((i+1)*number_of_groups): (i+2)*number_of_groups] = sorted_pos_reverse
                        sorted_center_zeros[:, ((i+1)*number_of_groups): (i+2)*number_of_groups] = sorted_center_reverse
                    else:
                        sorted_group_input_tokens_zeros[:, ((i+2)*number_of_groups): (i+3)*number_of_groups] = sorted_group_input_tokens_reverse
                        sorted_pos_zeros[:, ((i+2)*number_of_groups): (i+3)*number_of_groups] = sorted_pos_reverse
                        sorted_center_zeros[:, ((i+2)*number_of_groups): (i+3)*number_of_groups] = sorted_center_reverse


            group_input_tokens = sorted_group_input_tokens_zeros
            pos = sorted_pos_zeros
            sorted_center = sorted_center_zeros


        x = group_input_tokens
        # transformer
        x = self.drop_out(x)
        if (self.add_after_layer):
            x = self.blocks(x, pos, top_k_eigenvectors, self.k_top_eigenvectors, self.reverse)
        else:
            x = self.blocks(x, pos)
        x = self.norm(x)

        concat_f = x[:, :].mean(1)
        ret = self.cls_head_finetune(concat_f)

        if gt is not None:
            return ret, policy
        else:
            return ret

        # if gt is not None:
        #     loss, acc = self.get_loss_acc(ret, gt)
        #
        #     if tau is not None:
        #         reward = -loss.detach()
        #
        #         K_epochs = 4
        #         feat = x.detach()
        #         logits = self.logit_head(feat)
        #         logits_old = logits.detach()
        #
        #         value_baseline = self.value(feat)  # (B,)
        #         advantage = reward - value_baseline.detach()
        #
        #         for _ in range(K_epochs):
        #             logp_new =   # (B,)
        #             logp_old = pl_log_prob(logits_old, perm)  # (B,)
        #             ratio = torch.exp(logp_new - logp_old)  # r_t
        #
        #             # Surrogate objective
        #             surr1 = ratio * advantage
        #             surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
        #             policy_loss = -torch.min(surr1, surr2).mean()
        #
        #             # Value loss
        #             v_pred = self.value(feat)
        #             value_loss = 0.5 * (v_pred - reward).pow(2).mean()
        #
        #             # Entropy bonus (cheap MC estimate)
        #             entropy = pl_entropy(logits).mean()
        #
        #             # Total loss
        #             loss = policy_loss + self.val_coef * value_loss - self.ent_coef * entropy
        #
        #             # Optimisation step
        #             self.opt.zero_grad()
        #             loss.backward()
        #             torch.nn.utils.clip_grad_norm_(
        #                 list(self.policy.parameters()) + list(self.value.parameters()), 1.0
        #             )
        #             self.opt.step()
        #
        #
        #         if torch.isinf(self.baseline):
        #             self.baseline = reward
        #         else:
        #             self.baseline = self.alpha * self.baseline + (1 - self.alpha) * reward
        #             A = reward - self.baseline
        #             loss += -A * policy
        #     return ret, loss, acc
        # else:
        #     return ret


class MaskMamba(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.mask_ratio = config.transformer_config.mask_ratio
        self.trans_dim = config.transformer_config.trans_dim
        self.depth = config.transformer_config.depth
        self.num_heads = config.transformer_config.num_heads
        print_log(f'[args] {config.transformer_config}', logger='Mamba')
        # embedding
        self.encoder_dims = config.transformer_config.encoder_dims
        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        self.mask_type = config.transformer_config.mask_type
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )

        self.blocks = MixerModel(d_model=self.trans_dim,
                                 n_layer=self.depth,
                                 rms_norm=self.config.rms_norm)

        self.norm = nn.LayerNorm(self.trans_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _mask_center_block(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()
        # mask a continuous part
        mask_idx = []
        for points in center:
            # G 3
            points = points.unsqueeze(0)  # 1 G 3
            index = random.randint(0, points.size(1) - 1)
            distance_matrix = torch.norm(points[:, index].reshape(1, 1, 3) - points, p=2,
                                         dim=-1)  # 1 1 3 - 1 G 3 -> 1 G

            idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0]  # G
            ratio = self.mask_ratio
            mask_num = int(ratio * len(idx))
            mask = torch.zeros(len(idx))
            mask[idx[:mask_num]] = 1
            mask_idx.append(mask.bool())

        bool_masked_pos = torch.stack(mask_idx).to(center.device)  # B G

        return bool_masked_pos

    def _mask_center_rand(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        B, G, _ = center.shape
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()

        self.num_mask = int(self.mask_ratio * G)

        overall_mask = np.zeros([B, G])
        for i in range(B):
            mask = np.hstack([
                np.zeros(G - self.num_mask),
                np.ones(self.num_mask),
            ])
            np.random.shuffle(mask)
            overall_mask[i, :] = mask
        overall_mask = torch.from_numpy(overall_mask).to(torch.bool)

        return overall_mask.to(center.device)  # B G

    def forward(self, neighborhood, center, noaug=False):
        # generate mask
        if self.mask_type == 'rand':
            bool_masked_pos = self._mask_center_rand(center, noaug=noaug)  # B G
        else:
            bool_masked_pos = self._mask_center_block(center, noaug=noaug)

        group_input_tokens = self.encoder(neighborhood)  # B G C

        batch_size, seq_len, C = group_input_tokens.size()

        x_vis = group_input_tokens[~bool_masked_pos].reshape(batch_size, -1, C)
        # add pos embedding
        # mask pos center
        masked_center = center[~bool_masked_pos].reshape(batch_size, -1, 3)
        pos = self.pos_embed(masked_center)

        # transformer
        x_vis = self.blocks(x_vis, pos)
        x_vis = self.norm(x_vis)

        return x_vis, bool_masked_pos


def build_rw_laplacian(A: Tensor, eps: float = 1e-6) -> Tensor:
    """
    Dense random-walk Laplacian L_rw = I − D^{-1}A  (no sparse ops)
    """
    num_nodes = A.shape[-1]
    device = A.device
    dtype   = A.dtype

    A = 0.5 * (A + A.transpose(-1, -2))

    deg = A.sum(dim=-1, keepdim=True).clamp(min=eps)
    A_D = A / deg

    return torch.eye(num_nodes, device=device, dtype=dtype) - A_D

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Traversal ordering utilities (single + multi)
# -----------------------------------------------------------------------------

def _meyer_window(lam: Tensor, lam1: float = 0.5, lam2: float = 1.0) -> Tensor:
    """Smooth C¹ bump that rises from 1→0 on [lam1, lam2]."""
    out = torch.zeros_like(lam)
    mask1 = lam < lam1
    mask2 = (lam >= lam1) & (lam <= lam2)
    mask3 = lam > lam2
    out[mask1] = 1.0
    if mask2.any():
        t = (lam[mask2] - lam1) / (lam2 - lam1)  # ∈[0,1]
        out[mask2] = 0.5 * (1 + torch.cos(torch.pi * t))
    out[mask3] = 0.0
    return out


def make_tight_frame_kernels(J: int,
                             lam_max: float = 2.0,
                             lam1: float = 0.5,
                             lam2: float = 1.0) -> Tuple[List[Callable[[Tensor], Tensor]], Callable[[Tensor], Tensor]]:
    """Return list of wavelet kernels g_j and low‑pass h satisfying tight frame."""
    def h_kernel(lam: Tensor) -> Tensor:
        return _meyer_window(lam / lam_max, lam1, lam2)

    def g_base(lam: Tensor) -> Tensor:
        return torch.sqrt(torch.clamp(1 - h_kernel(lam) ** 2, min=0.0))

    g_kernels = []
    for j in range(J):
        scale = 2.0 ** j
        g_kernels.append(lambda l, s=scale: g_base(l * s))
    return g_kernels, h_kernel


################################################################################
class GraphWaveletTransform(torch.nn.Module):
    """Batched SGWT with optional Parseval tight frame.

    Parameters
    ----------
    scales : list[float]
        SGWT scales t_j (ignored if tight_frame=True where dyadic scales are used).
    K : int
        Chebyshev order.
    tight_frame : bool
        If True, ignore `scales` and construct Meyer tight‑frame kernels.
    """

    def __init__(self,
                 scales: List[float] | None = None,
                 K: int = 30,
                 J: Optional[int] = None,
                 tight_frame: bool = False):
        super().__init__()
        self.K = int(K)
        self.J = J
        self.tight = tight_frame
        if tight_frame:
            #assert scales is None or len(scales) == 0, "scales unused in tight_frame mode"
            self.scales = None  # will be implicit dyadic
        else:
            assert scales is not None and len(scales) > 0, "provide scales when not tight"
            self.register_buffer("scales", torch.tensor(scales, dtype=torch.float32))

    # -------------------------------------------------------------------------
    def forward(self, x: Tensor, laplacian: Tensor) -> Tensor:
        """Return (B,N, F×(J+1)) with scaling + J wavelet coeffs."""
        if laplacian.dim() != 3:
            raise ValueError("laplacian shape must be (B,N,N)")
        B, N, F = x.shape
        device, dtype = x.device, x.dtype
        I = torch.eye(N, device=device, dtype=dtype).expand(B, N, N)
        L_hat = laplacian - I  # eigen range in [‑1,1]

        # Chebyshev recursion ---------------------------------------------------
        T_prev = x                        # T0
        T_curr = torch.matmul(L_hat, x)   # T1
        polys: list[Tensor] = [T_prev, T_curr]
        for _ in range(2, self.K):
            T_next = 2 * torch.matmul(L_hat, T_curr) - T_prev
            polys.append(T_next)
            T_prev, T_curr = T_curr, T_next

        k_vec = torch.arange(self.K, device=device, dtype=dtype)
        cos_k = torch.cos(torch.pi * k_vec / self.K)  # (K,)

        if self.tight:
            # Build Meyer tight frame kernels ----------------------------------
            #J = 3  # dyadic levels 2^0, 2^1, 2^2  (change as needed)
            g_ker, h_ker = make_tight_frame_kernels(self.J)
            coeffs_all: list[Tensor] = []
            # scaling coeffs h(L)x
            w_h = h_ker(cos_k + 1.0)
            c_scaling = sum(w_h[k] * polys[k] for k in range(self.K))
            coeffs_all.append(c_scaling)  # first block
            # wavelet coeffs per dyadic scale
            for j in range(self.J):
                w = g_ker[j](cos_k + 1.0)  # broadcast over k
                c_j = sum(w[k] * polys[k] for k in range(self.K))
                coeffs_all.append(c_j)
        else:
            J = self.scales.numel()
            coeffs_all: list[Tensor] = []
            for t in self.scales:
                weights = (t * (cos_k + 1.0)) * torch.exp(-t * (cos_k + 1.0))  # default λe^{-λ}
                c = sum(weights[k] * polys[k] for k in range(self.K))
                coeffs_all.append(c)
        return torch.cat(coeffs_all, dim=2)  # (B,N,F*(J+1 or J))


def chebyshev_stack(x: Tensor, L_hat: Tensor, K: int) -> Tensor:
    """Return tensor (K,B,N,F) containing T_k(L_hat)x for k=0..K-1."""
    B, N, F = x.shape
    T_prev, T_curr = x, torch.matmul(L_hat, x)
    polys = [T_prev, T_curr]
    for _ in range(2, K):
        T_next = 2 * torch.matmul(L_hat, T_curr) - T_prev
        polys.append(T_next)
        T_prev, T_curr = T_curr, T_next
    return torch.stack(polys)  # (K,B,N,F)


################################################################################
# Analytic tight-frame spectral graph wavelet transform with Jackson damping
################################################################################

def meyer_lowpass(lam: Tensor, lam_max: float) -> Tensor:
    """Meyer-style low-pass bump on [0, lam_max]"""
    lam1, lam2 = 0.5 * lam_max, lam_max
    out = torch.zeros_like(lam)
    mask0 = lam <= lam1
    mask1 = (lam > lam1) & (lam < lam2)
    out[mask0] = 1.0
    if mask1.any():
        t = (lam[mask1] - lam1) / (lam2 - lam1)
        out[mask1] = 0.5 * (1 + torch.cos(torch.pi * t))
    return out

class ComplexMeyerSGWT(torch.nn.Module):
    """Analytic Meyer wavelets, optionally complex-valued, for sign-aware traversal.

    Parameters
    ----------
    J : int
        Number of dyadic wavelet scales.
    K : int
        Chebyshev polynomial order.
    lam_max : float
        Max Laplacian eigenvalue (default 2.0).
    use_complex : bool
        If True, return complex coefficients (g + i*h).
        If False, return only real g_j(L)x.
    """
    def __init__(self, J: int = 3, K: int = 30, lam_max: float = 2.0, use_complex: bool = False,
                 use_delta: bool = False, jackson: bool = False):
        super().__init__()
        self.J = J
        self.K = K
        self.lam_max = lam_max
        self.use_complex = use_complex
        self.use_delta = use_delta
        self.jackson = jackson

        if jackson:
            k = torch.arange(K, dtype=torch.float32)
            gamma = ((K - k + 1) * torch.cos(torch.pi * k / (K + 1)) +
                     torch.sin(torch.pi * k / (K + 1)) * (1 / torch.tan(torch.tensor(torch.pi) / (K + 1))))
            gamma = gamma / (K + 1)
            self.register_buffer('gamma', gamma)

    def forward(self, x: Tensor, L: Tensor) -> Tensor:
        B, N, F = x.shape
        device, dtype = x.device, x.dtype
        I = torch.eye(N, device=device, dtype=dtype).expand_as(L)
        Lhat = L - I

        T_prev, T_curr = x, torch.matmul(Lhat, x)
        polys = [T_prev, T_curr]
        for _ in range(2, self.K):
            T_next = 2 * torch.matmul(Lhat, T_curr) - T_prev
            polys.append(T_next)
            T_prev, T_curr = T_curr, T_next
        T_stack = torch.stack(polys)

        k_vec = torch.arange(self.K, device=device, dtype=dtype)
        cos_k = torch.cos(torch.pi * k_vec / self.K)
        lam_k = (cos_k + 1.0) * (self.lam_max / 2)

        channels = (1 if self.use_delta else 0) + self.J
        dtype_out = torch.cfloat if self.use_complex else dtype
        out = torch.zeros(B, N, F, channels, dtype=dtype_out, device=device)

        idx = 0
        if self.use_delta:
            # Batch-wise δ-band around λ1 from eigvalsh
            # L: (B,N,N)
            B = x.shape[0]
            # 1) compute the two smallest eigenvalues per graph
            eigvals = torch.linalg.eigvalsh(L)  # (B,N)
            lambda0 = eigvals[:, 0]  # (B,)
            lambda1 = eigvals[:, 1]  # (B,)
            # 2) determine δ-band half-width per batch
            eps0 = 0.05 * self.lam_max
            eps1 = (lambda1 - lambda0) * 0.5
            eps2 = self.lam_max / self.K
            delta_epsilon = torch.stack([
                eps0 * torch.ones_like(lambda1),
                eps1,
                eps2 * torch.ones_like(lambda1)
            ], dim=1).max(dim=1).values  # (B,)
            delta_lambda = lambda1  # (B,)
            # 3) build bump values g_delta[b,k]
            # lam_k: (K,)
            # lam_diff[b,k]
            lam_diff = lam_k[None, :] - delta_lambda[:, None]  # (B,K)
            mask = lam_diff.abs() <= delta_epsilon[:, None]  # (B,K)
            g_delta = torch.zeros_like(lam_diff)
            t = lam_diff[mask] / delta_epsilon[:, None].expand_as(mask)[mask]
            g_delta[mask] = torch.cos(0.5 * torch.pi * t)
            if self.jackson:
                # apply Jackson damping
                g_delta = g_delta * self.gamma[None]
            # 4) apply via Chebyshev stack T_stack: (K,B,N,F)
            # need g_delta.T: (K,B)
            g_bp = g_delta.transpose(0, 1).view(self.K, B, 1, 1)
            delta_coeff = (g_bp * T_stack).sum(dim=0)  # (B,N,F)
            out[..., idx] = delta_coeff
            idx += 1

        for j in range(self.J):
            lam1 = self.lam_max / (2**(j+1))
            lam2 = self.lam_max / (2**j)
            nu = (lam_k - lam1) / (lam2 - lam1)
            gk = torch.zeros_like(lam_k)
            hk = torch.zeros_like(lam_k)
            m0 = lam_k <= lam1
            m2 = lam_k >= lam2
            m1 = (~m0) & (~m2)
            hk[m0] = 1.0; gk[m0] = 0.0
            gk[m2] = 1.0; hk[m2] = 0.0
            if m1.any():
                t = nu[m1]
                gk[m1] = torch.sin(0.5 * torch.pi * t)
                hk[m1] = torch.cos(0.5 * torch.pi * t)
            if self.jackson:
                gk = gk * self.gamma[None]
                hk = hk * self.gamma[None]
            real_j = (gk.view(self.K,1,1,1) * T_stack).sum(dim=0)
            if self.use_complex:
                imag_j = (hk.view(self.K,1,1,1) * T_stack).sum(dim=0)
                out[..., idx] = real_j + 1j * imag_j
            else:
                out[..., idx] = real_j
            idx += 1
        return out


def sinkhorn_sort(X, epsilon=0.01, max_iter=100, stop_thresh=1e-3, use_hungarian: bool = False):
    """
    Perform Sinkhorn differentiable sorting on tensor X.

    Args:
        X: Input tensor (B, K, N)
        epsilon: Entropic regularization parameter
        max_iter: Number of Sinkhorn iterations

    Returns:
        P: Soft permutation matrix (B, K, N, N)
    """
    X = X.transpose(1, 2)
    B, K, N = X.shape

    # Expand tensors to compute pairwise cost
    X_expand = X.unsqueeze(-1)  # (B, K, N, 1)
    X_t_expand = X.unsqueeze(-2)  # (B, K, 1, N)

    # Compute pairwise squared distance as cost
    C = (X_expand - X_t_expand).pow(2)  # (B, K, N, N)

    # Create Gibbs kernel
    K_matrix = torch.exp(-C / epsilon)  # (B, K, N, N)

    # Initialize marginals (uniform)
    r = torch.ones((B, K, N), device=X.device) / N
    c = torch.ones((B, K, N), device=X.device) / N

    u = torch.ones_like(r)
    v = torch.ones_like(c)

    # Sinkhorn iterations
    for _ in range(max_iter):
        u_prev = u
        v_prev = v

        u = r / (K_matrix @ v.unsqueeze(-1)).squeeze(-1)  # Update u (B, K, N)
        v = c / (K_matrix.transpose(-2, -1) @ u.unsqueeze(-1)).squeeze(-1)  # Update v (B, K, N)

        diff = (u - u_prev).abs().sum(dim=-1).mean() + (v - v_prev).abs().sum(dim=-1).mean()
        if diff < stop_thresh:
            break

    # Compute doubly stochastic matrix (soft permutation)
    P_hat = torch.diag_embed(u) @ K_matrix @ torch.diag_embed(v)  # (B, K, N, N)

    if use_hungarian:
        P_ = P_hat.clone().detach().flatten(0, 1).float().cpu().numpy()
        P_hard = torch.zeros_like(P_hat).flatten(0, 1)
        for b in range(P_.shape[0]):
            row, col = linear_sum_assignment(-P_[b])
            P_hard[b, row, col] = 1
        P_hard = P_hard.view_as(P_hat)
    else:

        P_ = P_hat.clone().detach()
        for i in range(N - 1):
            P_[:, :, i + 1:].scatter_(-1, torch.argmax(P_[:, :, i], dim=-1, keepdim=True).unsqueeze(-1).expand(-1, -1, N - (i + 1), -1), 0)

        P_hard = torch.zeros_like(P_hat)
        P_hard.scatter_(-1, torch.argmax(P_, dim=-1, keepdim=True), 1)
    assert torch.all(P_hard.sum(-1) == 1)
    assert torch.all(P_hard.sum(-2) == 1)

    P = P_hard + (P_hat - P_hat.detach())

    return P, P_hat


def neural_sort(s: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """
    Computes the NeuralSort relaxation of the permutation matrix that sorts each row of s.

    Args:
        s: a float tensor of shape (batch_size, n)
           containing the scores to be sorted (higher scores come first).
        tau: positive float temperature parameter; as tau → 0, the output approaches
             a hard permutation matrix.

    Returns:
        P_hat: a float tensor of shape (batch_size, n, n), where each row of P_hat[b]
               is a unimodal, row-stochastic approximation to the permutation matrix
               that sorts s[b] in descending order.
    """
    s = s.transpose(1, 2)
    batch_size, J, n = s.size()
    # Step 1: lift to shape (batch_size, n, 1) for pairwise operations
    s_unsq = s.unsqueeze(-1)  # [B, n, 1]

    # Step 2: compute pairwise |s_i - s_j| matrix A_s
    A_s = torch.abs(s_unsq - s_unsq.transpose(-1, -2))  # [B, n, n]

    # Step 3: compute B = A_s @ (1_n 1_n^T) so that B[b,i,j] = sum_k |s[b,i] - s[b,k]|
    one = torch.ones((n, 1), device=s.device, dtype=s.dtype)  # [n, 1]
    B = torch.matmul(A_s, one.matmul(one.t()))  # [B, n, n]

    # Step 4: build the scaling vector c_j = n + 1 - 2*j for j = 1…n
    # and form C[b,i,j] = s[b,i] * c_j
    idx = torch.arange(1, n + 1, device=s.device, dtype=s.dtype)
    c = (n + 1 - 2 * idx).view(1, 1, 1, n)  # [1, 1, n]
    C = s_unsq * c  # [B, n, n]

    # Step 5: combine and apply softmax row-wise
    # P_max[b,i,j] = (n+1-2j)*s[b,i] - sum_k |s[b,i] - s[b,k]|
    P_max = C - B  # [B, n, n]
    P_hat = F.softmax(P_max / tau, dim=-1)  # row-stochastic [B, n, n]

    P_ = P_hat.clone().detach()
    for i in range(n - 1):
        P_[:, :, i + 1:].scatter_(-1, torch.argmax(P_[:, :, i], dim=-1, keepdim=True).unsqueeze(-1).expand(-1, -1, n - (i+1), -1), 0)

    P_hard = torch.zeros_like(P_hat)
    P_hard.scatter_(-1, torch.argmax(P_, dim=-1, keepdim=True), 1)
    assert torch.all(P_hard.sum(-1) == 1)
    assert torch.all(P_hard.sum(-2) == 1)

    P = P_hard + (P_hat - P_hat.detach())

    return P


################################################################################
# Traversal ordering utility
################################################################################

def traversal_order_from_coeffs_perm(
    coeffs: torch.Tensor,
    k: Optional[int] = None,
    strategy: str = "coarsest_k",
    use_diff_sort: bool = False,
    sort_tau: float = 1.0,
    eps: float = 1e-9,
    return_soft_perm: bool = False,
) -> Union[
    torch.Tensor,
    Tuple[torch.Tensor, List[torch.Tensor]],
    List[torch.Tensor]
]:
    """
    Generate traversal orders or soft permutations from wavelet coeffs.

    Parameters
    ----------
    coeffs : Tensor of shape (B, N, F, J)
        Real or complex wavelet/scattering coefficients.
    k : int, optional
        Number of traversals (<=J).  Defaults to J.
    strategy : {'coarsest_k','finest_k','top_energy'}
        How to pick scales.
    return_perm : bool
        If True (and use_sinkhorn=False), also returns hard permutation matrices.
    use_sinkhorn : bool
        If True, returns soft permutations P (and ignores return_perm/orders).
    sinkhorn_iters : int
        Number of Sinkhorn normalization steps.
    sinkhorn_tau : float
        Temperature for initial assignment softmax.
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    If use_sinkhorn:
        perms : list of length k, each a FloatTensor (B, N, N) of soft permutations.
    Else if return_perm:
        orders : LongTensor (B, k, N), perms : list of (B,N,N) hard one-hot perms.
    Else:
        orders : LongTensor of shape (B, k, N).
    """
    B, N, f, J = coeffs.shape
    if k is None:
        k = J

    # 1) compute per-point scores
    if coeffs.is_complex():
        score = torch.angle(coeffs).mean(dim=2)  # (B, N, J)
    else:
        score = coeffs.mean(dim=2)               # (B, N, J)

    # 2) pick scale indices to traverse
    if strategy == "coarsest_k":
        scale_ids = list(range(J-1, J-1-k, -1))
    elif strategy == "finest_k":
        scale_ids = list(range(0, k))
    elif strategy == "top_energy":
        energy = (score**2).sum(dim=1)           # (B, J)
        avg_e   = energy.mean(dim=0)            # (J,)
        scale_ids = torch.topk(avg_e, k).indices.tolist()
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")

    scale_ids = torch.as_tensor(scale_ids, device=score.device, dtype=torch.long).expand_as(score)
    score = score.gather(dim=2, index=scale_ids)

    if use_diff_sort:
        P, P_hat = sinkhorn_sort(score, epsilon=0.05, max_iter=40, use_hungarian=True)
        if return_soft_perm:
            return P, P_hat
    else:
        ord = torch.argsort(score.transpose(1, 2), dim=2, descending=False)
        P = F.one_hot(ord, num_classes=N).to(coeffs.dtype)  # (B, N, N)
    return P, None


    # orders = torch.stack(orders, dim=1)  # (B, k, N)
    # if return_perm:
    #     return orders, perms
    # return orders



# def multi_scale_traversals(coeffs: Tensor,
#                            F: int,
#                            J: int,
#                            k: int,
#                            strategy: str = "coarsest_k",
#                            aggregate: str = "signed_mean",
#                            tau: float = None) -> List[Tensor]:
#     """Return *k* batched permutations, one for each chosen scale.
#
#     coeffs : (B,N,F·J)
#     returns : list of length k; each item (B,N)
#     """
#     if k > J:
#         raise ValueError("k cannot exceed J")
#     B, N = coeffs.size(0), coeffs.size(1)
#
#     # choose scale indices per strategy
#     if strategy == "coarsest_k":
#         scale_ids = list(range(J - 1, J - 1 - k, -1))
#     elif strategy == "finest_k":
#         scale_ids = list(range(k))
#     elif strategy == "top_energy":
#         ener = coeffs.view(B, N, F, J).pow(2).sum(dim=(0, 1, 2))  # (J,)
#         scale_ids = torch.topk(ener, k).indices.tolist()
#     else:
#         raise ValueError("unknown strategy")
#
#     orders = [
#         traversal_order_from_coeffs_perm(coeffs, F, J,
#                                      aggregate=aggregate, ascending=False, tau=tau)
#         for sid in scale_ids
#     ]
#     return torch.stack(orders, dim=1)


class DiffusionWavelets:
    """
    Constructs an orthonormal diffusion wavelet basis on batched graphs.

    Attributes
    ----------
    W : List[Tensor]
        List of wavelet basis tensors for scales j=0..J-1, each of shape (B, N, r_j).
    VJ : Tensor
        Orthonormal scaling functions for coarsest scale V_J, shape (B, N, r_J).
    """
    def __init__(self, t: float, J: int, lam_max: float = 2.0):
        """
        Parameters
        ----------
        t : float
            Base diffusion time parameter (>0), used if not scaling-adaptive.
        J : int
            Number of wavelet scales.
        lam_max : float
            Maximum Laplacian eigenvalue (default 2.0 for normalized Laplacian).
        """
        self.t = t
        self.J = J
        self.lam_max = lam_max

    def fit(
        self,
        L: Tensor,
        rank_schedule: Optional[List[int]] = None,
        use_energy: bool = False,
        tol: float = 1e-3
    ) -> Tuple[List[Tensor], Tensor]:
        """
        Fit diffusion wavelets for graph Laplacian L, batched over B.

        Parameters
        ----------
        L : Tensor, shape (B, N, N)
            Symmetric normalized graph Laplacians.
        rank_schedule : list of int, optional
            If provided, specifies ranks r_j for each scale j=0..J.
        use_energy : bool
            If True, use energy-based compression at each scale.
        tol : float
            Energy threshold for SVD-based compression (only if use_energy=True).

        Returns
        -------
        W : list of Tensors
            Wavelet bases for scales j=0..J-1, each of shape (B, N, r_j).
        VJ : Tensor
            Scaling basis for coarsest scale, shape (B, N, r_J).
        """
        # L: (B,N,N)
        B, N, _ = L.shape
        device, dtype = L.device, L.dtype
        J = self.J

        # Initialize V0 as identity basis for each batch
        V_prev = torch.eye(N, device=device, dtype=dtype).unsqueeze(0).expand(B, N, N)
        V_bases = [V_prev]

        # Chebyshev-like coarse hop: power of T
        lam_max = 2.0
        jj = torch.arange(1, J+1, device=device, dtype=dtype)
        cutoff_j = lam_max / (2 ** jj)
        #t_j = math.log(2.0) / cutoff_j[:, None, None, None]
        t_j =  math.log(2.0) / lam_max
        T = torch.matrix_exp(-t_j * (2 ** (jj[:, None, None, None] - 1)) * L.unsqueeze(0).expand(J, -1, -1, -1))

        rank_schedule = torch.tensor([max(1, (N + (1 << j) - 1) // (1 << j)) for j in range(1, J + 1)])

        # eigvals = torch.linalg.eigvalsh(L)
        # rank_schedule = (eigvals[:, None, :].expand(-1, J, -1) <= cutoff_j[None, :, None].expand(B, -1, N)).sum(dim=2)
        # rank_schedule = torch.clamp(torch.max(rank_schedule, dim=0)[0], min=1)

        assert len(rank_schedule) == J, "rank_schedule must have length J"

        # Build V_j for j=1..J via batched operations
        for j in range(1, J+1):
            k = rank_schedule[j-1].item()
            M  = torch.matmul(T[j-1], V_prev)              # (B,N,r_prev)
            # Batched SVD
            # 1) form Gram matrix
            C = M.transpose(1, 2) @ M  # (B, r, r)

            # 2) eigh returns ascending
            evals, V = torch.linalg.eigh(C.float())  # evals[:, 0] is smallest

            # 3) get top-k via slicing
            V_k = V[:, :, -k:].flip(-1)  # (B, r, k)
            sigma_k = torch.sqrt(evals[:, -k:]).flip(-1) # (B, k)

            # 4) build U_k
            U_k = M @ V_k  # (B, N, k)
            Vj = U_k / sigma_k.view(B, 1, k)  # normalize columns

            #U, _, _ = torch.linalg.svd(M, full_matrices=False)
            #Vj = U[:, :, :k]                            # (B,N,k)
            V_bases.append(Vj)
            V_prev = Vj

        # Compute wavelet complements W_j = V_j \ V_{j+1}
        W = []

        for j in range(J):
            Vj   = V_bases[j]     # (B,N,r_j)
            Vjp1 = V_bases[j+1]   # (B,N,r_{j+1})
            # project Vj onto Vjp1: P = Vjp1 @ (Vjp1^T @ Vj)
            P = torch.einsum('bip,brp,brq->biq', Vjp1, Vjp1, Vj)  # (B,N,r_j)
            Wj = Vj - P
            # batched QR
            Qj, _ = torch.linalg.qr(Wj)  # (B,N,r_j)
            W.append(Qj.to(L.dtype))

        # Final scaling basis V_J
        VJ = V_bases[-1].to(L.dtype)  # (B,N,r_J)
        return W, VJ

# Example usage:
# L: (N,N) symmetric normalized Laplacian
# dw = DiffusionWavelets(t=1.0, J=3)
# W, VJ = dw.fit(L)
# W[j] gives an orthonormal basis for wavelets at scale j
# VJ gives the scaling functions at the coarsest level

################################################################################
# Diffusion Wavelet Spectral Transform (orthonormal)
################################################################################

class DiffusionWaveletSGWT(torch.nn.Module):
    """
    Computes an orthonormal diffusion-wavelet transform on batched graph signals.

    Parameters
    ----------
    t : float
        Diffusion time parameter (>0).
    J : int
        Number of wavelet scales.
    lam_max : float
        Maximum Laplacian eigenvalue (default 2.0).
    lam_cutoff : float, optional
        Reference eigenvalue to start spectral-gap scheduling. If None, uses lam_max.
    use_energy : bool
        If True, apply energy-based compression in diffusion wavelets.
    tol : float
        Energy threshold for SVD-based compression (only if use_energy=True).
    """
    def __init__(
        self,
        t: float = 1.0,
        J: int = 3,
        num_group: int = 1,
        in_features: int = 3,
        lam_max: float = 2.0,
        lam_cutoff: float = None,
        use_energy: bool = False,
        tol: float = 1e-3
    ):
        super().__init__()
        self.t = t
        self.J = J
        self.lam_max = lam_max
        self.lam_cutoff = lam_cutoff or lam_max
        self.use_energy = use_energy
        self.tol = tol
        self.dw = DiffusionWavelets(t, J, lam_max=self.lam_max)

        hidden_dim = 64

        self.pos_embed = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

       #  self.mixer = nn.Sequential(
       #     nn.Linear(in_features*(self.J + 1), self.J + 1),
       # )

        self.mixer = nn.Sequential(
            nn.Linear(hidden_dim * (self.J + 1), 2*hidden_dim),
            nn.LayerNorm(2*hidden_dim),
            nn.GELU(),
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim * (self.J + 1)),
        )

        def ortho(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        self.mixer.apply(ortho)

    def forward(self, x: Tensor, L: Tensor, tau: float = 0.5) -> Tensor:
        """
        Apply diffusion-wavelet transform to batched signals.

        x : Tensor, shape (B, N, F)
            Input features per node.
        L : Tensor, shape (B, N, N)
            Batch of symmetric normalized Laplacians.

        Returns
        -------
        coeffs : Tensor
            (B, N, F, J+1) tensor of coefficients: channel 0 is scaling,
            channels 1..J are wavelet bands.
        """
        B, N, F = x.shape
        device, dtype = x.device, x.dtype

        # Fit diffusion wavelets in batched mode
        W_list, VJ = self.dw.fit(
            L,
            rank_schedule=None,
            use_energy=self.use_energy,
            tol=self.tol
        )  # W_list: list of (B,N,r_j), VJ: (B,N,r_J)

        x = self.pos_embed(x)

        PJ = [torch.matmul(VJ, VJ.transpose(1, 2))] + [torch.matmul(Wj, Wj.transpose(1, 2)) for Wj in W_list]
        PJ = torch.stack(PJ, dim=1)
        coeffs = torch.matmul(PJ, x.unsqueeze(1)).permute(0, 2, 3, 1)

        # # Initialize output
        # coeffs = torch.zeros(B, N, F, self.J+1, device=device, dtype=dtype)
        #
        # # Scaling channel: projection onto VJ
        # # PJ: (B,N,N)
        # PJ = torch.matmul(VJ, VJ.transpose(1, 2))
        # # (B,N,F)
        # coeffs[..., 0] = torch.matmul(PJ, x)
        #
        # # Wavelet channels
        # for j, Wj in enumerate(W_list):
        #     Pj = torch.matmul(Wj, Wj.transpose(1, 2))  # (B,N,N)
        #     coeffs[..., j+1] = torch.matmul(Pj, x)

        #coeffs = self.norm(coeffs.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # coeffs = self.norm(coeffs)
        eps = torch.finfo(coeffs.dtype).eps

        rms = torch.sqrt((coeffs ** 2).mean(dim=(0, 1), keepdim=True) + eps)
        coeffs = coeffs / rms.clamp_min(1e-2)  # keeps node-to-node energy differences

        coeffs = coeffs + self.mixer(coeffs.flatten(2)).view_as(coeffs)
        coeffs = torch.sqrt((coeffs ** 2).sum(dim=2, keepdim=True)) / coeffs.shape[2]
        #return coeffs

        if self.training:
            g = -torch.log(-torch.log(torch.rand_like(coeffs) + eps) + eps)
            coeffs = coeffs + tau*g

        return coeffs


class GraphScattering(torch.nn.Module):
    def __init__(self, diffusion_sgwt, act: str = 'abs'):
        super().__init__()
        self.sgwt = diffusion_sgwt   # e.g. DiffusionWaveletSGWT
        if act == 'relu':
            self.nonlin = torch.relu
        elif act == 'gelu':
            self.nonlin = torch.nn.functional.gelu
        elif act == 'abs':
            self.nonlin = torch.abs

    def forward(self, x, L, level: int = 2):
        assert level in (0, 1, 2)
        # x: (B,N,F), L: (B,N,N)
        coeffs = self.sgwt(x, L)           # (B,N,F,C) with C=J+1
        # drop scaling channel:
        b1 = coeffs[...,1:]             # (B,N,F,J)
        B,N,F,J = b1.shape
        S0 = coeffs[...,0]                 # (B,N,F)

        if level >= 1:
            # first-order scattering:
            b1 = self.nonlin(b1)
        S1 = list(torch.unbind(b1, dim=-1))

        S2 = []
        if level >= 2:
            # second-order scattering:
            U1 = b1.permute(0, 3, 1, 2).flatten(0, 1)
            coeffs2 = self.sgwt(U1, L.repeat_interleave(J, dim=0))  # (B,N,F,J+1)
            b2 = self.nonlin(coeffs2.view(B, J, N, F, -1)[..., 1:])

            for j in range(J):
                for k in range(j + 1, J):
                    S2.append(b2[:, j, :, :, k])

        # you can go to higher orders similarly
        # return a list of all orders (or concatenate along new dim)
        return torch.stack([S0] + S1 + S2, dim=-1)


def plackett_luce_dist(logits):
    return torch.sum(logits - torch.logcumsumexp(logits.flip(-1), dim=-1).flip(-1), dim=-1)


class MaskMamba_2(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.mask_ratio = config.transformer_config.mask_ratio
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.trans_dim = config.transformer_config.trans_dim
        self.depth = config.transformer_config.depth
        self.num_heads = config.transformer_config.num_heads
        self.k_top_eigenvectors = config.transformer_config.k_top_eigenvectors
        print_log(f'[args] {config.transformer_config}', logger='Mamba')
        # embedding
        self.encoder_dims = config.transformer_config.encoder_dims
        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        self.mask_type = config.transformer_config.mask_type
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )

        self.blocks = MixerModel(d_model=self.trans_dim,
                                 n_layer=self.depth,
                                 rms_norm=self.config.rms_norm)

        # self.eigen_embed = nn.Sequential(
        #     nn.Linear(2, 128),
        #     nn.GELU(),
        #     nn.Linear(128, self.trans_dim)
        # )

        # self.logit_blocks = MixerModel2(d_model=self.trans_dim,
        #                                 n_layer=3,
        #                                 rms_norm=self.config.rms_norm)
        #
        # self.logit_norm = nn.LayerNorm(self.trans_dim)
        #
        # head_dim = (self.trans_dim)
        # assert self.trans_dim >= self.k_top_eigenvectors
        # self.logit_head = nn.Sequential(nn.Linear(head_dim, self.trans_dim),
        #                                 nn.LayerNorm(self.trans_dim),
        #                                 nn.GELU(),
        #                                 nn.Linear(self.trans_dim, 1))
        # self.logit_head2 = nn.Sequential(nn.Linear(head_dim, self.trans_dim),
        #                                  nn.LayerNorm(self.trans_dim),
        #                                  nn.GELU(),
        #                                  nn.Linear(self.trans_dim, 1))
        #
        # self.permuter = StochasticNeuralSortPermuter()

        self.norm = nn.LayerNorm(self.trans_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _mask_center_block(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()
        # mask a continuous part
        mask_idx = []
        for points in center:
            # G 3
            points = points.unsqueeze(0)  # 1 G 3
            index = random.randint(0, points.size(1) - 1)
            distance_matrix = torch.norm(points[:, index].reshape(1, 1, 3) - points, p=2,
                                         dim=-1)  # 1 1 3 - 1 G 3 -> 1 G

            idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0]  # G
            ratio = self.mask_ratio
            mask_num = int(ratio * len(idx))
            mask = torch.zeros(len(idx))
            mask[idx[:mask_num]] = 1
            mask_idx.append(mask.bool())

        bool_masked_pos = torch.stack(mask_idx).to(center.device)  # B G

        return bool_masked_pos

    def _mask_center_rand(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        B, G, _ = center.shape
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()

        self.num_mask = int(self.mask_ratio * G)

        overall_mask = np.zeros([B, G])
        for i in range(B):
            mask = np.hstack([
                np.zeros(G - self.num_mask),
                np.ones(self.num_mask),
            ])
            np.random.shuffle(mask)
            overall_mask[i, :] = mask
        overall_mask = torch.from_numpy(overall_mask).to(torch.bool)

        return overall_mask.to(center.device)  # B G


    def sort_points_by_fiedler(self, points, bool_masked_pos, fiedler_vector):

        B, N, C = points.shape
        _, sorted_indices = torch.sort(fiedler_vector, dim=1)

        expanded_indices_mask = sorted_indices
        expanded_indices = sorted_indices.unsqueeze(-1).expand(-1, -1, C)

        sorted_points = torch.gather(points, 1, expanded_indices)
        sorted_bool_masked_pos = torch.gather(bool_masked_pos, 1, expanded_indices_mask)

        return sorted_points, sorted_bool_masked_pos

    def sort_points_by_fiedler_for_neighberhood(self, points, fiedler_vector):

        B, N, _, _ = points.shape
        # Generate indices from Fiedler vector for sorting
        fiedler_vector = fiedler_vector.real
        _, sorted_indices = torch.sort(fiedler_vector, dim=1)

        # Expand the indices to work for x, y, z coordinates
        expanded_indices = sorted_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.group_size, 3)

        # Sort points based on the sorted indices
        sorted_points = torch.gather(points, 1, expanded_indices)

        return sorted_points

    def calc_logit_dist_reg(self, center: torch.Tensor, logits: torch.Tensor, sigma: float = None, eps: float = 1e-6):
        B, N, C = center.shape
        center = center.view(B, -1, self.num_group, C)
        logits = logits.view(B, -1, self.num_group)
        dist = torch.sum((center.unsqueeze(2) - center.unsqueeze(3)) ** 2, dim=-1)
        #dist, idx = torch.sort(dist, dim=-1)
        dist = torch.sqrt(dist)
        # dist, idx = self.knn(center, center)
        if sigma is None:
            sigma = torch.median(dist[..., -1], dim=-1)[0]  # robust bandwidth
        # dists = dist[..., 1:]#.reshape(-1)
        W = torch.exp(-(dist ** 2) / (2 * sigma[..., None, None] ** 2))

        # 2) build W with scatter_
        #W = torch.zeros((B, N, N), device=center.device, dtype=center.dtype)

        # scatter into the *column* dimension (dim=1).
        # For row i, idx[i, r] is the column and weights[i, r] the value.
        W = 0.5 * (W + W.transpose(-1, -2))
        W.diagonal(dim1=-2, dim2=-1).zero_()

        deg = W.sum(dim=-1, keepdim=True).clamp(min=eps)
        W_norm = W / deg
        I = torch.eye(self.num_group, device=center.device, dtype=center.dtype)
        L = I - W_norm

        theta_L = torch.matmul(L, logits.unsqueeze(-1)).squeeze(-1)
        reg = 0.5 * (logits * theta_L).sum(dim=(1, 2)).mean()
        return reg

    def forward(self, neighborhood, center, top_k_eigenvalues, top_k_eigenvectors, k_top_eigenvectors,
                reverse, noaug=False, tau:bool=None, orders=None, orders_soft=None, ret_only_policy=False):
        # generate mask
        if self.mask_type == 'rand':
            bool_masked_pos = self._mask_center_rand(center, noaug=noaug).cuda()  # B G
        else:
            bool_masked_pos = self._mask_center_block(center, noaug=noaug)

        group_input_tokens = self.encoder(neighborhood)  # B G C

        batch_size, seq_len, C = group_input_tokens.size()

        pos = self.pos_embed(center)

        sorted_bool_masked_pos_list = []

        sorted_group_input_tokens_t = []
        sorted_group_input_tokens_full_t = []
        sorted_pos_t = []
        sorted_pos_mask_t = []
        sorted_pos_full_t = []
        sorted_neighborhood_t = []
        sorted_center_t = []

        policy = torch.tensor(0.0, device=center.device, dtype=center.dtype)

        # if top_k_eigenvectors is not None:
        #     for i in range (k_top_eigenvectors):
        #
        #         sorted_group_input_tokens, sorted_bool_masked_pos = self.sort_points_by_fiedler(group_input_tokens, bool_masked_pos, top_k_eigenvectors[:, :, i])
        #         sorted_neighborhood = self.sort_points_by_fiedler_for_neighberhood(neighborhood, top_k_eigenvectors[:, :, i])
        #
        #         #TODO: optimize unneccessary gather and sort
        #         sorted_pos, _ = self.sort_points_by_fiedler(pos, bool_masked_pos, top_k_eigenvectors[:, :, i])
        #         sorted_center, _ = self.sort_points_by_fiedler(center, bool_masked_pos, top_k_eigenvectors[:, :, i])
        #
        #         sorted_pos_full = sorted_pos
        #         sorted_group_input_tokens_full = sorted_group_input_tokens
        #         sorted_group_input_tokens = sorted_group_input_tokens[~sorted_bool_masked_pos].reshape(batch_size, -1, C)
        #         sorted_pos = sorted_pos[~sorted_bool_masked_pos].reshape(batch_size, -1, C)
        #         sorted_pos_mask = sorted_pos_full[sorted_bool_masked_pos].reshape(batch_size, -1, C)
        #
        #         sorted_group_input_tokens_t.append(sorted_group_input_tokens)
        #         sorted_group_input_tokens_full_t.append(sorted_group_input_tokens_full)
        #         sorted_pos_t.append(sorted_pos)
        #         sorted_pos_mask_t.append(sorted_pos_mask)
        #         sorted_pos_full_t.append(sorted_pos_full)
        #         sorted_neighborhood_t.append(sorted_neighborhood)
        #         sorted_center_t.append(sorted_center)

                # if (i != 0):
                #     sorted_group_input_tokens = torch.cat((sorted_group_input_tokens_t, sorted_group_input_tokens), 1)
                #     sorted_group_input_tokens_full = torch.cat((sorted_group_input_tokens_full_t, sorted_group_input_tokens_full), 1)
                #     sorted_pos = torch.cat((sorted_pos_t, sorted_pos), 1)
                #     sorted_pos_mask = torch.cat((sorted_pos_mask_t, sorted_pos_mask), 1)
                #     sorted_pos_full = torch.cat((sorted_pos_full_t, sorted_pos_full), 1)
                #     sorted_neighborhood = torch.cat((sorted_neighborhood_t, sorted_neighborhood), 1)
                #
                #
                # sorted_group_input_tokens_t = sorted_group_input_tokens
                # sorted_group_input_tokens_full_t = sorted_group_input_tokens_full
                # sorted_pos_t = sorted_pos
                # sorted_pos_mask_t = sorted_pos_mask
                # sorted_pos_full_t = sorted_pos_full
                # sorted_neighborhood_t = sorted_neighborhood

            #     sorted_bool_masked_pos_list.append(sorted_bool_masked_pos)
            #
            # sorted_group_input_tokens_t = torch.cat(sorted_group_input_tokens_t, dim=1)
            # sorted_group_input_tokens_full_t = torch.cat(sorted_group_input_tokens_full_t, dim=1)
            # sorted_pos_t = torch.cat(sorted_pos_t, dim=1)
            # sorted_pos_mask_t = torch.cat(sorted_pos_mask_t, dim=1)
            # sorted_pos_full_t = torch.cat(sorted_pos_full_t, dim=1)
            # sorted_neighborhood_t = torch.cat(sorted_neighborhood_t, dim=1)
            # sorted_center_t = torch.cat(sorted_center_t, dim=1)

            #regularization = torch.tensor(0.0, device=center.device, dtype=center.dtype)
            # if tau is not None:
            #     if reverse:
            #         sorted_group_input_tokens_full_t_reverse = sorted_group_input_tokens_full_t.flip(1)
            #         sorted_group_input_tokens_full = torch.cat((sorted_group_input_tokens_full_t, sorted_group_input_tokens_full_t_reverse), 1)
            #         sorted_pos_full_t_reverse = sorted_pos_full_t.flip(1)
            #         sorted_pos_full = torch.cat((sorted_pos_full_t, sorted_pos_full_t_reverse), 1)
            #     else:
            #         sorted_group_input_tokens_full = sorted_group_input_tokens_full_t
            #         sorted_pos_full = sorted_pos_full_t
            #
            #     sorted_vecs, _ = torch.sort(top_k_eigenvectors.transpose(1, 2), dim=2)
            #     eigen_emb = self.eigen_embed(
            #         torch.stack((-sorted_vecs, top_k_eigenvalues.unsqueeze(-1).expand_as(sorted_vecs)), dim=-1)).flatten(1,
            #                                                                                                              2)
            #     eigen_emb = torch.cat((eigen_emb, eigen_emb.flip(1)), 1)
            #
            #     logits_feats = self.logit_blocks(sorted_group_input_tokens_full.detach(), sorted_pos_full.detach(), eigen_emb)
            #     logits_feats = self.logit_norm(logits_feats)
            #
            #     if reverse:
            #         logits_feats1, logits_feats2 = torch.tensor_split(logits_feats, 2, dim=1)
            #         logits_feats = logits_feats1 + logits_feats2.flip(1)
            #
            #     logits_inner = self.logit_head(logits_feats).view(batch_size, -1, self.num_group)
            #     logits_outer = self.logit_head2(
            #         logits_feats.view(batch_size, -1, self.num_group, self.trans_dim).mean(dim=2)).squeeze(-1)
            #
            #     seq_len = self.num_group
            #     P_inner = self.permuter(logits_inner.view(-1, seq_len), tau).view(batch_size, -1, seq_len, seq_len)
            #     P_outer = self.permuter(logits_outer, tau)
            #     perm_indices_outer = torch.argmax(P_outer, dim=2)
            #     perm_indices = torch.argmax(P_inner, dim=3) + perm_indices_outer.unsqueeze(-1) * seq_len
            #     perm_indices = perm_indices.view(batch_size, -1)
            #
            #     # sorted_bool_masked_pos = torch.cat(sorted_bool_masked_pos_list, -1)
            #     # sorted_bool_masked_pos = sorted_bool_masked_pos.gather(dim=1, index=perm_indices)
            #     # sorted_bool_masked_pos = torch.bmm(P, sorted_bool_masked_pos.unsqueeze(-1).to(P.dtype)).squeeze(-1)
            #     # sorted_bool_masked_pos = sorted_bool_masked_pos > 0.5
            #     # sorted_bool_masked_pos_list = list(torch.split(sorted_bool_masked_pos, seq_len, dim=1))
            #     logits_inner = logits_inner.flatten(1, 2).gather(dim=1, index=perm_indices)
            #     logits_outer = logits_outer.gather(dim=1, index=perm_indices_outer)
            #
            #     sorted_bool_masked_pos = torch.cat(sorted_bool_masked_pos_list, -1)
            #     sorted_bool_masked_pos = sorted_bool_masked_pos.gather(dim=1, index=perm_indices)
            #     #sorted_bool_masked_pos = torch.bmm(P, sorted_bool_masked_pos.unsqueeze(-1).to(P.dtype)).squeeze(-1)
            #     #sorted_bool_masked_pos = sorted_bool_masked_pos > 0.5
            #     sorted_bool_masked_pos_list = list(torch.split(sorted_bool_masked_pos, seq_len, dim=1))
            #     #logits_inner = logits_inner[~sorted_bool_masked_pos].reshape(batch_size, -1)
            #     sorted_pos_full_t = sorted_pos_full_t.gather(dim=1, index=perm_indices[..., None].expand_as(sorted_pos_full_t))
            #     sorted_pos_t = sorted_pos_full_t[~sorted_bool_masked_pos].reshape(batch_size, -1, C)
            #     sorted_pos_mask_t = sorted_pos_full_t[sorted_bool_masked_pos].reshape(batch_size, -1, C)
            #     sorted_neighborhood_t = sorted_neighborhood_t.gather(dim=1, index=perm_indices[..., None, None].expand_as(sorted_neighborhood_t)) #torch.einsum('bij,bjgd->bigd', P, sorted_neighborhood_t)
            #     sorted_center_t = sorted_center_t.gather(dim=1, index=perm_indices[..., None].expand_as(sorted_center_t))
            #
            #     policy = plackett_luce_dist(logits_inner.view(batch_size, -1, seq_len)).sum(dim=1) + plackett_luce_dist(logits_outer)
            #     #regularization = self.calc_logit_dist_reg(sorted_center_t, logits_inner)
            #
            #     if ret_only_policy:
            #         return None, None, None, None, None, None, None, policy

        if orders is not None:
            P_hard = orders.detach()
            if self.training:
                P = orders_soft
            else:
                P = P_hard
            # sorted_bool_masked_pos = bool_masked_pos.unsqueeze(1).expand(-1, orders.size(1), -1)
            # sorted_bool_masked_pos = sorted_bool_masked_pos.gather(dim=2, index=orders)
            #
            # orders = orders.unsqueeze(-1)
            #
            # sorted_group_input_tokens_t = group_input_tokens.unsqueeze(1).expand(-1, orders.size(1), -1, -1)
            # sorted_group_input_tokens_t = sorted_group_input_tokens_t.gather(dim=2, index=orders.expand_as(sorted_group_input_tokens_t))
            # sorted_group_input_tokens_full_t = sorted_group_input_tokens_t.flatten(1, 2)
            # sorted_group_input_tokens_t = sorted_group_input_tokens_t[~sorted_bool_masked_pos].reshape(batch_size, -1, C)
            #
            # sorted_pos_t = pos.unsqueeze(1).expand(-1, orders.size(1), -1, -1)
            # sorted_pos_t = sorted_pos_t.gather(dim=2, index=orders.expand_as(sorted_pos_t))
            # sorted_pos_full_t = sorted_pos_t
            # sorted_pos_t = sorted_pos_full_t[~sorted_bool_masked_pos].reshape(batch_size, -1, C)
            # sorted_pos_mask_t = sorted_pos_full_t[sorted_bool_masked_pos].reshape(batch_size, -1, C)
            # sorted_pos_full_t = sorted_pos_full_t.flatten(1, 2)
            #
            # sorted_neighborhood_t = neighborhood.unsqueeze(1).expand(-1, orders.size(1), -1, -1, -1)
            # sorted_neighborhood_t = sorted_neighborhood_t.gather(dim=2, index=orders.unsqueeze(-1).expand_as(sorted_neighborhood_t)).flatten(1, 2)
            #
            # sorted_center_t = center.unsqueeze(1).expand(-1, orders.size(1), -1, -1)
            # sorted_center_t = sorted_center_t.gather(dim=2, index=orders.expand_as(sorted_center_t)).flatten(1, 2)
            #
            # sorted_bool_masked_pos_list = list(torch.unbind(sorted_bool_masked_pos, 1))

            # 1) build the full [B,N,N] mask for every (head,tail)‐pair
            #    first reorder the 1D mask by P → [B, N], then broadcast out to [B,N,N]
            sorted_bool_masked_pos = torch.matmul(P_hard, bool_masked_pos.unsqueeze(1).unsqueeze(-1).float()).squeeze(-1).bool()  # [B, N]
            #sorted_bool_masked_pos = sorted_tail_mask.unsqueeze(1).expand(-1, N, -1)  # [B, N, N]

            # 2) head‐tokens repeated over tails
            #    reorder the head‐tokens by P → [B, N, C], then unsqueeze+expand → [B,N,N,C]
            sorted_group_input_tokens_full = torch.matmul(P, group_input_tokens.unsqueeze(1))  # [B, N, C]
            # sorted_group_input_tokens_full = sorted_heads.unsqueeze(2).expand(-1, -1, N,
            #                                                                   -1)  # [B, N(heads), N(tails), C]
            # flatten the head+tail dims
            sorted_group_input_tokens_full_t = sorted_group_input_tokens_full.flatten(1, 2)
            # pick only the unmasked pairs
            sorted_group_input_tokens_t = sorted_group_input_tokens_full[~sorted_bool_masked_pos].view(batch_size, -1, C)

            # 3) same for “pos”: these are your head‐positions repeated across tails
            sorted_pos_full = torch.matmul(P, pos.unsqueeze(1))  # [B, N, C]
            sorted_pos_full_t = sorted_pos_full.flatten(1, 2)
            sorted_pos_t = sorted_pos_full[~sorted_bool_masked_pos].view(batch_size, -1, C)
            sorted_pos_mask_t = sorted_pos_full[sorted_bool_masked_pos].view(batch_size, -1, C)

            # 4) neighborhood: you want to apply P on its 2nd dim (the “source” index)
            #    so that new_neighborhood[b,i,:,:] = sum_j P[b,i,j] * neighborhood[b,j,:,:]
            sorted_neighborhood_full = torch.einsum('bhij,bjkl->bhikl', P, neighborhood)  # [B, N(heads), N, D]
            sorted_neighborhood_t = sorted_neighborhood_full.flatten(1, 2)

            # 5) center also just gets permuted in its first (token) dim, then flattened
            sorted_center_full = torch.matmul(P, center.unsqueeze(1))  # [B, N, C]
            sorted_center_t = sorted_center_full.flatten(1, 2)

            # 6) if you still need a python list of per‐head masks:
            sorted_bool_masked_pos_list = list(torch.unbind(sorted_bool_masked_pos, dim=1))

        if (reverse == True):
            sorted_group_input_tokens_t_reverse = sorted_group_input_tokens_t.flip(1)
            sorted_pos_t_reverse = sorted_pos_t.flip(1)
            sorted_pos_mask_t_reverse = sorted_pos_mask_t.flip(1)
            sorted_pos_full_t_reverse = sorted_pos_full_t.flip(1)
            sorted_neighborhood_t_reverse = sorted_neighborhood_t.flip(1)
            sorted_center_t_reverse = sorted_center_t.flip(1)
            sorted_group_input_tokens = torch.cat((sorted_group_input_tokens_t, sorted_group_input_tokens_t_reverse), 1)
            sorted_pos = torch.cat((sorted_pos_t, sorted_pos_t_reverse), 1)
            sorted_pos_mask = torch.cat((sorted_pos_mask_t, sorted_pos_mask_t_reverse), 1)
            sorted_pos_full = torch.cat((sorted_pos_full_t, sorted_pos_full_t_reverse), 1)
            sorted_neighborhood = torch.cat((sorted_neighborhood_t, sorted_neighborhood_t_reverse), 1)

            x_vis = sorted_group_input_tokens.cuda()
            pos = sorted_pos.cuda()

            sorted_bool_masked_pos_tensor = torch.cat(sorted_bool_masked_pos_list, -1)
            sorted_bool_masked_pos_tensor = sorted_bool_masked_pos_tensor.flip(-1)

        # transformer
        x_vis = self.blocks(x_vis, pos)
        x_vis = self.norm(x_vis)

        return (x_vis, sorted_bool_masked_pos_list, sorted_pos_mask.cuda(), sorted_pos_full.cuda(),
                sorted_bool_masked_pos_tensor.cuda(), sorted_neighborhood.cuda(), self.mask_ratio, policy)


class MaskMamba_3(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.mask_ratio = config.transformer_config.mask_ratio
        self.trans_dim = config.transformer_config.trans_dim
        self.depth = config.transformer_config.depth
        self.num_heads = config.transformer_config.num_heads
        print_log(f'[args] {config.transformer_config}', logger='Mamba')
        # embedding
        self.encoder_dims = config.transformer_config.encoder_dims
        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        self.mask_type = config.transformer_config.mask_type
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )

        self.blocks = MixerModel(d_model=self.trans_dim,
                                 n_layer=self.depth,
                                 rms_norm=self.config.rms_norm)

        self.norm = nn.LayerNorm(self.trans_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _mask_center_block(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()
        # mask a continuous part
        mask_idx = []
        for points in center:
            # G 3
            points = points.unsqueeze(0)  # 1 G 3
            index = random.randint(0, points.size(1) - 1)
            distance_matrix = torch.norm(points[:, index].reshape(1, 1, 3) - points, p=2,
                                         dim=-1)  # 1 1 3 - 1 G 3 -> 1 G

            idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0]  # G
            ratio = self.mask_ratio
            mask_num = int(ratio * len(idx))
            mask = torch.zeros(len(idx))
            mask[idx[:mask_num]] = 1
            mask_idx.append(mask.bool())

        bool_masked_pos = torch.stack(mask_idx).to(center.device)  # B G

        return bool_masked_pos

    def _mask_center_rand(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        B, G, _ = center.shape
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()

        self.num_mask = int(self.mask_ratio * G)

        overall_mask = np.zeros([B, G])
        for i in range(B):
            mask = np.hstack([
                np.zeros(G - self.num_mask),
                np.ones(self.num_mask),
            ])
            np.random.shuffle(mask)
            overall_mask[i, :] = mask
        overall_mask = torch.from_numpy(overall_mask).to(torch.bool)

        return overall_mask.to(center.device)  # B G


    def sort_points_by_fiedler(self, points, bool_masked_pos, fiedler_vector, noaug= False):
        """
        Sorts points based on the Fiedler vector, optimized for GPU execution.

        Parameters:
        - points: Tensor of shape (B, N, 3) where B is the batch size,
                N is the number of points in the point cloud, and 3 are the xyz coordinates.
        - fiedler_vector: The Fiedler vector used for sorting, of shape (B, N).

        Returns:
        - sorted_points: Tensor of sorted points.
        """
        B, N, _ = points.shape
        # Generate indices from Fiedler vector for sorting
        #fiedler_vector = fiedler_vector.real
        _, sorted_indices = torch.sort(fiedler_vector, dim=1)

        # Expand the indices to work for x, y, z coordinates
        expanded_indices_mask = sorted_indices
        expanded_indices = sorted_indices.unsqueeze(-1).expand(-1, -1, 384)

        # Sort points based on the sorted indices
        sorted_points = torch.gather(points, 1, expanded_indices)
        sorted_bool_masked_pos = torch.gather(bool_masked_pos, 1, expanded_indices_mask)

        ############# find index for learnable
        if (noaug == False):
            sorted_indices_learnable_tokens = sorted_indices[sorted_bool_masked_pos].reshape(B, 38)
        else:
            sorted_indices_learnable_tokens = torch.zeros((B, 38)).cuda()

        return sorted_points, sorted_bool_masked_pos, sorted_indices_learnable_tokens, sorted_indices

    def sort_points_by_fiedler_for_neighberhood(self, points, fiedler_vector):
        """
        Sorts points based on the Fiedler vector, optimized for GPU execution.

        Parameters:
        - points: Tensor of shape (B, N, 3) where B is the batch size,
                N is the number of points in the point cloud, and 3 are the xyz coordinates.
        - fiedler_vector: The Fiedler vector used for sorting, of shape (B, N).

        Returns:
        - sorted_points: Tensor of sorted points.
        """
        B, N, _, _ = points.shape
        # Generate indices from Fiedler vector for sorting
        fiedler_vector = fiedler_vector.real
        _, sorted_indices = torch.sort(fiedler_vector, dim=1)

        # Expand the indices to work for x, y, z coordinates
        expanded_indices = sorted_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 32, 3)

        # Sort points based on the sorted indices
        sorted_points = torch.gather(points, 1, expanded_indices)

        return sorted_points

    def find_indices_vectorized(self, a, sorted_indices_learnable_tokens_2):
        B, N = a.shape
        T = sorted_indices_learnable_tokens_2.size(1)

        # Expand 'a' to compare with each element in 'sorted_indices_learnable_tokens_2'
        a_expanded = a.unsqueeze(2).expand(B, N, T)

        # Expand 'sorted_indices_learnable_tokens_2' for comparison
        sorted_indices_expanded = sorted_indices_learnable_tokens_2.unsqueeze(1).expand(B, N, T)

        # Compare and find indices where values match
        matches = (a_expanded == sorted_indices_expanded).nonzero(as_tuple=True)

        # Extract the third component of 'matches', which are the indices of matches
        # Since 'matches[2]' gives us the flat index positions across the tensor,
        # We need to reshape it to match the shape of 'a'
        output = matches[2].view(B, N)

        return output.cuda()

    def forward(self, neighborhood, center, top_k_eigenvectors, k_top_eigenvectors, reverse, noaug=False):
        # generate mask
        if self.mask_type == 'rand':
            bool_masked_pos = self._mask_center_rand(center, noaug=noaug).cuda()  # B G
        else:
            bool_masked_pos = self._mask_center_block(center, noaug=noaug)

        group_input_tokens = self.encoder(neighborhood)  # B G C

        batch_size, seq_len, C = group_input_tokens.size()

        pos = self.pos_embed(center)

        sorted_bool_masked_pos_list = []
        sorted_found_indices_list = []
        sorted_found_indices_list_reverse = []

        for i in range (k_top_eigenvectors):
            if (i == 0):
                sorted_group_input_tokens, sorted_bool_masked_pos, sorted_indices_learnable_tokens_, sorted_indices = self.sort_points_by_fiedler(group_input_tokens, bool_masked_pos, top_k_eigenvectors[:, :, i], noaug)
                sorted_indices_learnable_tokens = sorted_indices_learnable_tokens_
            else:
                sorted_group_input_tokens, sorted_bool_masked_pos, _, sorted_indices = self.sort_points_by_fiedler(group_input_tokens, bool_masked_pos, top_k_eigenvectors[:, :, i], noaug)

            sorted_found_indices_list.append(self.find_indices_vectorized(sorted_indices_learnable_tokens, sorted_indices)[..., None])
            sorted_found_indices_list_reverse.append(self.find_indices_vectorized(sorted_indices_learnable_tokens, sorted_indices.flip(-1))[..., None])

            sorted_neighborhood = self.sort_points_by_fiedler_for_neighberhood(neighborhood, top_k_eigenvectors[:, :, i])

            sorted_pos, _, _, _ = self.sort_points_by_fiedler(pos, bool_masked_pos, top_k_eigenvectors[:, :, i], noaug)

            sorted_pos_full = sorted_pos
            sorted_group_input_tokens = sorted_group_input_tokens[~sorted_bool_masked_pos].reshape(batch_size, -1, C)
            sorted_pos = sorted_pos[~sorted_bool_masked_pos].reshape(batch_size, -1, C)
            sorted_pos_mask = sorted_pos_full[sorted_bool_masked_pos].reshape(batch_size, -1, C)

            if (i != 0):
                sorted_group_input_tokens = torch.cat((sorted_group_input_tokens_t, sorted_group_input_tokens), 1)
                sorted_pos = torch.cat((sorted_pos_t, sorted_pos), 1)
                sorted_pos_mask = torch.cat((sorted_pos_mask_t, sorted_pos_mask), 1)
                sorted_pos_full = torch.cat((sorted_pos_full_t, sorted_pos_full), 1)
                sorted_neighborhood = torch.cat((sorted_neighborhood_t, sorted_neighborhood), 1)

            sorted_group_input_tokens_t = sorted_group_input_tokens
            sorted_pos_t = sorted_pos
            sorted_pos_mask_t = sorted_pos_mask
            sorted_pos_full_t = sorted_pos_full
            sorted_neighborhood_t = sorted_neighborhood

            sorted_group_input_tokens_t_reverse = sorted_group_input_tokens_t.flip(1)
            sorted_pos_t_reverse = sorted_pos_t.flip(1)
            sorted_pos_mask_t_reverse = sorted_pos_mask_t.flip(1)
            sorted_pos_full_t_reverse = sorted_pos_full_t.flip(1)
            sorted_neighborhood_t_reverse = sorted_neighborhood_t.flip(1)

            sorted_bool_masked_pos_list.append(sorted_bool_masked_pos)

        sorted_found_indices_tensor = torch.cat(sorted_found_indices_list, -1)
        sorted_found_indices_list_reverse.reverse()
        sorted_found_indices_tensor_reverse = torch.cat(sorted_found_indices_list_reverse, -1)

        sorted_found_indices_tensor_final = torch.cat((sorted_found_indices_tensor, sorted_found_indices_tensor_reverse), -1)

        if (reverse == True):
            sorted_group_input_tokens_t_reverse = sorted_group_input_tokens_t.flip(1)
            sorted_pos_t_reverse = sorted_pos_t.flip(1)
            sorted_pos_mask_t_reverse = sorted_pos_mask_t.flip(1)
            sorted_pos_full_t_reverse = sorted_pos_full_t.flip(1)
            sorted_neighborhood_t_reverse = sorted_neighborhood_t.flip(1)
            sorted_group_input_tokens = torch.cat((sorted_group_input_tokens_t, sorted_group_input_tokens_t_reverse), 1)
            sorted_pos = torch.cat((sorted_pos_t, sorted_pos_t_reverse), 1)
            sorted_pos_mask = torch.cat((sorted_pos_mask_t, sorted_pos_mask_t_reverse), 1)
            sorted_pos_full = torch.cat((sorted_pos_full_t, sorted_pos_full_t_reverse), 1)
            sorted_neighborhood = torch.cat((sorted_neighborhood_t, sorted_neighborhood_t_reverse), 1)

            x_vis = sorted_group_input_tokens.cuda()
            pos = sorted_pos.cuda()

            sorted_bool_masked_pos_tensor = torch.cat(sorted_bool_masked_pos_list, -1)
            sorted_bool_masked_pos_tensor = sorted_bool_masked_pos_tensor.flip(-1)

        # transformer
        x_vis = self.blocks(x_vis, pos)
        x_vis = self.norm(x_vis)

        return x_vis, sorted_bool_masked_pos_list, sorted_pos_mask.cuda(), sorted_pos_full.cuda(), sorted_bool_masked_pos_tensor.cuda(), sorted_neighborhood.cuda(), sorted_found_indices_tensor_final.cuda()


class MambaDecoder(nn.Module):
    def __init__(self, embed_dim=384, depth=4, norm_layer=nn.LayerNorm, config=None):
        super().__init__()
        if hasattr(config, "use_external_dwconv_at_last"):
            self.use_external_dwconv_at_last = config.use_external_dwconv_at_last
        else:
            self.use_external_dwconv_at_last = False
        self.blocks = MixerModel(d_model=embed_dim,
                                 n_layer=depth,
                                 rms_norm=config.rms_norm,
                                 drop_path=config.drop_path)
        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, pos, return_token_num):
        x = self.blocks(x, pos)

        x = self.head(self.norm(x[:, -return_token_num:]))  # only return the mask tokens predict pixel
        return x


class MambaDecoder_SST(nn.Module):
    def __init__(self, embed_dim=384, depth=4, norm_layer=nn.LayerNorm, config=None):
        super().__init__()
        if hasattr(config, "use_external_dwconv_at_last"):
            self.use_external_dwconv_at_last = config.use_external_dwconv_at_last
        else:
            self.use_external_dwconv_at_last = False
        self.blocks = MixerModel(d_model=embed_dim,
                                 n_layer=depth,
                                 rms_norm=config.rms_norm,
                                 drop_path=config.drop_path)
        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, pos, return_token_num):
        x = self.blocks(x, pos)

        x = self.head(self.norm(x))
        return x


@MODELS.register_module()
class Point_MAE_Mamba(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[Point_MAE] ', logger='Point_MAE')
        self.config = config
        self.trans_dim = config.transformer_config.trans_dim
        if (config.transformer_config.method == "MAMBA"):
            self.MAE_encoder = MaskMamba(config)
        elif (config.transformer_config.method == "smallest_eigenvectors_seperate_learnable_tokens"):
            self.MAE_encoder = MaskMamba_2(config)
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        self.decoder_depth = config.transformer_config.decoder_depth
        if (config.transformer_config.method == "MAMBA"):
            self.MAE_decoder = MambaDecoder(
                embed_dim=self.trans_dim,
                depth=self.decoder_depth,
                config=config,
            )

        elif (config.transformer_config.method == "smallest_eigenvectors_seperate_learnable_tokens"):
            self.MAE_decoder = MambaDecoder_SST(
                embed_dim=self.trans_dim,
                depth=self.decoder_depth,
                config=config,
            )

        print_log(f'[Point_MAE] divide point cloud into G{self.num_group} x S{self.group_size} points ...',
                  logger='Point_MAE')
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        # prediction head
        self.increase_dim = nn.Sequential(
            nn.Conv1d(self.trans_dim, 3 * self.group_size, 1)
        )

        trunc_normal_(self.mask_token, std=.02)
        self.loss = config.loss
        # loss
        self.build_loss_func(self.loss)

        self.method = config.transformer_config.method
        self.reverse = config.transformer_config.reverse
        self.k_top_eigenvectors = config.transformer_config.k_top_eigenvectors
        self.smallest = config.transformer_config.smallest
        self.knn_graph = config.transformer_config.knn_graph
        self.alpha = config.transformer_config.alpha
        self.symmetric = config.transformer_config.symmetric
        self.self_loop = config.transformer_config.self_loop
        self.binary = config.transformer_config.binary

        self.beta = 0.99
        baseline = torch.tensor(torch.inf)
        self.register_buffer('baseline', baseline)

        #scales = [0.02, 0.15, 0.5, 1.0]
        #self.sgwt = GraphWaveletTransform(scales=scales, K=25, tight_frame=True, J=4)
        #self.sgwt = ComplexMeyerSGWT(J=4, K=25, use_complex=True, use_delta=False)
        #sgwt = ComplexMeyerSGWT(J=4, K=30, use_complex=True, use_delta=False, jackson=False).cuda()
        self.diff_sgwt = DiffusionWaveletSGWT(J=3, in_features=3, num_group=self.num_group)
        #self.scatt = GraphScattering(diff_sgwt, act='abs')

        #self.sgwt = WaveGCSGWT(scales=scales, K=25)
        # for n, p in self.named_parameters():
        #     if 'logit_' in n:
        #         p.requires_grad = True
        #     else:
        #         p.requires_grad = False


    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            # self.loss_func = ChamferDistanceL1().cuda()
            self.loss_func = chamfer_distance
        elif loss_type == 'cdl2':
            # self.loss_func = ChamferDistanceL2().cuda()
            self.loss_func = chamfer_distance
        else:
            raise NotImplementedError
            # self.loss_func = emd().cuda()

    def create_graph_from_centers(self, points, k=5, alpha=1, symmetric = False, self_loop = False, binary = False):

        points = points.to('cuda')
        B, N, _ = points.shape

        # Compute pairwise Euclidean distances, shape (B, N, N), on GPU
        dist_matrix = torch.sqrt(torch.sum((points.unsqueeze(2) - points.unsqueeze(1)) ** 2, dim=-1))

        sigma = torch.mean(dist_matrix)

        # Find the k-nearest neighbors for each point, including itself
        distances, indices = torch.topk(-dist_matrix, k=k+1, largest=True, dim=-1)
        distances, indices = -distances[:, :, :], indices[:, :, :]

        if (self_loop):
            indices = indices[:, :, :]  # Remove self-loops
            distances = distances[:, :, :]  # Remove self-loops
        else:
            indices = indices[:, :, 1:]  # Remove self-loops
            distances = distances[:, :, 1:]  # Remove self-loops

        # Create a weighted adjacency matrix on GPU
        adjacency_matrix = torch.zeros(B, N, N, device='cuda')
        b_idx = torch.arange(B, device='cuda')[:, None, None]
        n_idx = torch.arange(N, device='cuda')[:, None]

        # Use gathered distances as weights
        if (self.alpha == 0):
            distances_new = torch.exp(-distances ** 2 / (2 * sigma ** 2))
        else:
            distances_new = torch.exp((-1) * alpha * (distances)**2)

        if (binary):
            adjacency_matrix[b_idx, n_idx, indices] = 1.
            if (symmetric):
                adjacency_matrix[b_idx, indices, n_idx] = 1.  # Ensure symmetry
        else:
            adjacency_matrix[b_idx, n_idx, indices] = distances_new
            if (symmetric):
                adjacency_matrix[b_idx, indices, n_idx] = distances_new  # Ensure symmetry

        return adjacency_matrix

    def calc_top_k_eigenvalues_eigenvectors(
            self,
            adj_matrices: torch.Tensor,  # (B, N, N)
            k: int,
            smallest: bool = False,
            eps: float = 1e-12):
        """
        Batched computation of the k largest (or smallest) eigenvalues / eigenvectors
        of the random-walk normalised Laplacian  L_rw = I â€“ Dâ»Â¹A.

        Returns
        -------
        top_k_eigenvalues  : (B, k)
        top_k_eigenvectors : (B, N, k)
        eigenvalues_full   : (B, N)
        eigenvectors_full  : (B, N, N)
        """
        B, N, _ = adj_matrices.shape
        device = adj_matrices.device
        dtype = adj_matrices.dtype  # keep FP32/FP64 unchanged

        # 1. Symmetrise A  â€¦ in batch
        A = 0.5 * (adj_matrices + adj_matrices.transpose(-1, -2))

        # 2. Row degrees and Dâ»Â¹A      (no explicit diag() needed)
        deg = A.sum(-1).clamp(min=eps)  # (B, N)
        DinvA = A / deg.unsqueeze(-1)  # broadcast divide

        # 3. Random-walk Laplacian  L_rw = I â€“ Dâ»Â¹A
        I = torch.eye(N, dtype=dtype, device=device)
        L_rw = I - DinvA  # (B, N, N)  batched

        # 4. Batched Hermitian eigendecomposition
        #    torch.linalg.eigh guarantees ascending eigenvalues
        eigenvalues_full, eigenvectors_full = torch.linalg.eigh(L_rw)

        # 5. Pick k extremal eigenpairs
        largest = not smallest
        top_vals, top_idx = torch.topk(
            eigenvalues_full, k, dim=-1,
            largest=largest, sorted=True)  # (B, k)

        # gather the matching eigenvectors â†’ (B, N, k)
        top_vecs = torch.gather(
            eigenvectors_full,  # (B, N, N)
            dim=-1,
            index=top_idx.unsqueeze(-2)  # (B, 1, k)
            .expand(-1, N, -1))

        return top_vals, top_vecs, eigenvalues_full, eigenvectors_full


    def forward(self, pts, noaug = False, vis=False, tau=None, use_wavelets:bool = False, use_diff_sort: bool = False,
                ret_policy: bool = False, ret_only_policy: bool = False, save_pts_dir: str = None, epoch: int = None, **kwargs):
        neighborhood, center, neighborhood_org = self.group_divider(pts)

        if (self.method == "MAMBA"):

            x_vis, mask, logits, P = self.MAE_encoder(neighborhood, center)
            B, _, C = x_vis.shape  # B VIS C

            pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)

            pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)

            _, N, _ = pos_emd_mask.shape
            mask_token = self.mask_token.expand(B, N, -1)
            x_full = torch.cat([x_vis, mask_token], dim=1)
            pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)

            x_rec = self.MAE_decoder(x_full, pos_full, N)

            B, M, C = x_rec.shape
            rebuild_points = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)  # B M 1024

            gt_points = neighborhood[mask].reshape(B * M, -1, 3)
            loss1 = self.loss_func(rebuild_points, gt_points)

            if vis:  # visualization
                vis_points = neighborhood[~mask].reshape(B * (self.num_group - M), -1, 3)
                full_vis = vis_points + center[~mask].unsqueeze(1)
                full_rebuild = rebuild_points + center[mask].unsqueeze(1)
                full = torch.cat([full_vis, full_rebuild], dim=0)
                # full_points = torch.cat([rebuild_points,vis_points], dim=0)
                full_center = torch.cat([center[mask], center[~mask]], dim=0)
                # full = full_points + full_center.unsqueeze(1)
                ret2 = full_vis.reshape(-1, 3).unsqueeze(0)
                ret1 = full.reshape(-1, 3).unsqueeze(0)
                # return ret1, ret2
                return ret1, ret2, full_center
            else:
                # return loss1
                return loss1[0]

        elif(self.method == "smallest_eigenvectors_seperate_learnable_tokens"):

            adjacency_matrix = self.create_graph_from_centers(center, self.knn_graph, self.alpha, self.symmetric, self.self_loop, self.binary)
            orders = None
            if use_wavelets:
                laplacian = build_rw_laplacian(adjacency_matrix)
                #coeffs = self.sgwt(center, laplacian)
                coeffs = self.diff_sgwt(center, laplacian, tau=tau)
                orders, orders_soft = traversal_order_from_coeffs_perm(coeffs, use_diff_sort=True, return_soft_perm=False)
                if orders_soft is None:
                    orders_soft = orders

                if save_pts_dir is not None:
                    sorted_neighborhood = torch.einsum('bhij,bjkl->bhikl', orders, neighborhood)  # [B, N(heads), N, D]
                    sorted_neighborhood = sorted_neighborhood

                    # 5) center also just gets permuted in its first (token) dim, then flattened
                    sorted_center = torch.matmul(orders, center.unsqueeze(1))  # [B, N, C]
                    sorted_center = sorted_center

                    # sorted_neighborhood = neighborhood.unsqueeze(1).expand(-1, orders.size(1), -1, -1, -1)
                    # sorted_neighborhood = sorted_neighborhood.gather(dim=2, index=orders.unsqueeze(-1).unsqueeze(-1).expand_as(sorted_neighborhood))
                    #
                    # sorted_center = center.unsqueeze(1).expand(-1, orders.size(1), -1, -1)
                    # sorted_center = sorted_center.gather(dim=2, index=orders.unsqueeze(-1).expand_as(sorted_center))

                    np.savez_compressed(os.path.join(save_pts_dir, '{}.npz'.format(epoch)),
                                        center=sorted_center.detach().cpu().numpy(),
                                        neighborhood=sorted_neighborhood.detach().cpu().numpy(),
                                        orders=orders.detach().cpu().numpy())

            # top_k_eigenvalues, top_k_eigenvectors, eigenvalues, eigenvectors = (
            #     self.calc_top_k_eigenvalues_eigenvectors(adjacency_matrix, k=self.k_top_eigenvectors, smallest = self.smallest))
            top_k_eigenvalues = None
            top_k_eigenvectors = None

            (x_vis, sorted_bool_masked_pos_list, sorted_pos_mask, sorted_pos_full, sorted_bool_masked_pos_tensor,
             sorted_neighborhood, mask_ratio, policy) = self.MAE_encoder(neighborhood, center, top_k_eigenvalues, top_k_eigenvectors,
                                                                         self.k_top_eigenvectors, self.reverse, noaug, tau=None, orders=orders,
                                                                         orders_soft=orders_soft, ret_only_policy=ret_only_policy)
            # if use_wavelets:
            #     policy = sum([plackett_luce_dist(sig_i) for sig_i in sig])
            if ret_only_policy:
                return policy
            B, _, C = x_vis.shape  # B VIS C
            n_masked_tokens_after_masking = int(mask_ratio * neighborhood_org.shape[1])
            n_visible_tokens_after_masking = neighborhood_org.shape[1] - n_masked_tokens_after_masking


            if (noaug == True):
                return x_vis

            _, N, _ = sorted_pos_mask.shape
            mask_token = self.mask_token.expand(B, N, -1)

            x_full_list = []

            # Put learnable tokens in real position
            for cnt, i in enumerate(sorted_bool_masked_pos_list):

                x_full = torch.zeros((x_vis.shape[0], center.shape[1], x_vis.shape[-1])).cuda()

                B, T, D = x_full.shape

                mask_token_part = mask_token[:, (cnt*n_masked_tokens_after_masking):(cnt+1)*n_masked_tokens_after_masking, :]
                x_vis_part = x_vis[:, (cnt*n_visible_tokens_after_masking):(cnt+1)*n_visible_tokens_after_masking, :]

                # Generate indices where i == 1 and i == 0
                mask_indices = torch.where(i == 1)
                vis_indices = torch.where(i == 0)

                # Update x_full where i is 1
                x_full[mask_indices] = mask_token_part.view(-1, D)[0:len(mask_indices[0])]

                # Update x_full where i is 0
                x_full[vis_indices] = x_vis_part.reshape(-1, D)[0:len(vis_indices[0])]

                x_full_list.append(x_full)

            cnt = 4
            x_full_tensor_1 = torch.cat(x_full_list, 1)
            x_full_tensor_2 = torch.zeros((x_full_tensor_1.shape[0], x_full_tensor_1.shape[1], x_full_tensor_1.shape[-1])).cuda()
            mask_token_part = mask_token[:, (cnt*n_masked_tokens_after_masking): , :]
            x_vis_part = x_vis[:, (cnt*n_visible_tokens_after_masking): , :]

            # Generate indices where i == 1 and i == 0
            mask_indices = torch.where(sorted_bool_masked_pos_tensor == 1)
            vis_indices = torch.where(sorted_bool_masked_pos_tensor == 0)

            # Update x_full where i is 1
            x_full_tensor_2[mask_indices] = mask_token_part.view(-1, D)[0:len(mask_indices[0])]

            # Update x_full where i is 0
            x_full_tensor_2[vis_indices] = x_vis_part.reshape(-1, D)[0:len(vis_indices[0])]

            x_full_tensor = torch.cat((x_full_tensor_1, x_full_tensor_2), 1)

            x_rec = self.MAE_decoder(x_full_tensor, sorted_pos_full, N)

            sorted_bool_masked_pos_tensor_1 = torch.cat(sorted_bool_masked_pos_list, 1)
            sorted_bool_masked_pos_tensor_final = torch.cat((sorted_bool_masked_pos_tensor_1, sorted_bool_masked_pos_tensor), 1)

            x_rec = x_rec[sorted_bool_masked_pos_tensor_final].reshape(B, -1, C)

            B, M, C = x_rec.shape
            rebuild_points = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)  # B M 1024

            gt_points = sorted_neighborhood[sorted_bool_masked_pos_tensor_final].reshape(B * M, -1, 3)
            loss1 = self.loss_func(rebuild_points.float(), gt_points.float(), batch_reduction=None)
            loss = loss1[0]

            #if tau is not None:
                #if torch.isinf(self.baseline):
                #    self.baseline = -loss.mean().detach()
                #reward = -loss.detach()
                #self.baseline = self.baseline*self.beta + (1-self.beta)*reward.mean()
                #advantage = reward.view(B, -1).mean(1) - self.baseline
                #advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-5)
            loss = loss.mean() #- 0.1*(advantage*policy).mean() + 1e-3*wavegc_smoothness_loss(self.sgwt)

            if ret_policy:
                return loss, policy
            else:
                return loss
            #return loss1           
