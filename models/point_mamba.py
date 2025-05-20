from typing import Union, Optional
import math
import random
from functools import partial

import numpy as np
import torch
import torch.nn as nn

from utils import misc
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *
from timm.models.layers import trunc_normal_
from timm.models.layers import DropPath

# from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from pytorch3d.loss import chamfer_distance
from mamba_ssm.modules.mamba_simple import Mamba

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

# from knn_cuda import KNN
from .block import Block
from .build import MODELS

from pytorch3d.ops import knn_points, sample_farthest_points    


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
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
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


    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
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
        # top_k_eigenvalues = torch.zeros((B, k+1)).cuda()
        # top_k_eigenvectors = torch.zeros((B, N, k+1)).cuda()
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
            ####normalized_A = torch.matmul(D_inv, A)
            I = torch.eye(N).cuda()    
            normalized_A = I - torch.matmul(D_inv, A)             

            eigenvalues, eigenvectors = torch.linalg.eigh(normalized_A)
            # eigenvalues, eigenvectors = torch.linalg.eig(normalized_A)        
            # eigenvalues = eigenvalues.real
            # eigenvectors = eigenvectors.real     

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

        # return top_k_eigenvalues[:, 1:], top_k_eigenvectors[:, :, 1:], eigenvalues_l, eigenvectors_l       
        return top_k_eigenvalues, top_k_eigenvectors, eigenvalues_l, eigenvectors_l     

    def calc_top_k_eigenvalues_eigenvectors_parallel(self, adj_matrices, k, smallest):
        B, N, _ = adj_matrices.shape

        # Compute degree matrix D for the entire batch
        D = torch.diag_embed(torch.sum(adj_matrices, dim=2))  # Shape: (B, N, N)

        # Compute D^-1 safely by adding a small epsilon to avoid division by zero
        D_inv = torch.diag_embed(1.0 / (torch.diagonal(D, dim1=1, dim2=2) + 1e-6))  # Shape: (B, N, N)

        # Identity matrix for the entire batch
        I = torch.eye(N, device=adj_matrices.device).unsqueeze(0).expand(B, N, N)   

        # Perform random-walk normalization: I - D^-1 * A
        normalized_A = I - torch.bmm(D_inv, adj_matrices)  # Batch matrix multiply

        # Compute eigenvalues and eigenvectors for the batch
        eigenvalues, eigenvectors = torch.linalg.eig(normalized_A)
        eigenvalues = eigenvalues.real  # Take real part
        eigenvectors = eigenvectors.real  # Take real part

        # Select the top k+1 eigenvalues and corresponding eigenvectors in parallel
        if not smallest:
            top_vals, top_indices = torch.topk(eigenvalues, k+1, largest=True, sorted=True, dim=1)
        else:
            top_vals, top_indices = torch.topk(eigenvalues, k+1, largest=False, sorted=True, dim=1)

        # top_indices is of shape (B, k+1), but we need it to match (B, N, k+1) for gathering eigenvectors
        # To gather along the third dimension (N), use top_indices with unsqueeze and expand appropriately
        top_vecs = torch.gather(eigenvectors, 2, top_indices.unsqueeze(1).expand(B, N, k+1))

        # Returning top k eigenvalues and eigenvectors, excluding the first one (index 0)
        return top_vals[:, 1:], top_vecs[:, :, 1:], eigenvalues, eigenvectors

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

    def forward(self, pts):
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
        return ret


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


class MaskMamba_2(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.mask_ratio = config.transformer_config.mask_ratio
        self.group_size = config.group_size
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


    def sort_points_by_fiedler(self, points, bool_masked_pos, fiedler_vector):

        B, N, _ = points.shape
        _, sorted_indices = torch.sort(fiedler_vector, dim=1)   
        
        expanded_indices_mask = sorted_indices
        expanded_indices = sorted_indices.unsqueeze(-1).expand(-1, -1, 384) 
        
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

        for i in range (k_top_eigenvectors):      
            sorted_group_input_tokens, sorted_bool_masked_pos = self.sort_points_by_fiedler(group_input_tokens, bool_masked_pos, top_k_eigenvectors[:, :, i]) 
            sorted_neighborhood = self.sort_points_by_fiedler_for_neighberhood(neighborhood, top_k_eigenvectors[:, :, i]) 

            sorted_pos, _ = self.sort_points_by_fiedler(pos, bool_masked_pos, top_k_eigenvectors[:, :, i])

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

            sorted_bool_masked_pos_list.append(sorted_bool_masked_pos)

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

        return x_vis, sorted_bool_masked_pos_list, sorted_pos_mask.cuda(), sorted_pos_full.cuda(), sorted_bool_masked_pos_tensor.cuda(), sorted_neighborhood.cuda(), self.mask_ratio


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

    def calc_top_k_eigenvalues_eigenvectors(self, adj_matrices, k, smallest):   
        
        B, N, _ = adj_matrices.shape
        # top_k_eigenvalues = torch.zeros((B, k+1)).cuda()
        # top_k_eigenvectors = torch.zeros((B, N, k+1)).cuda()    
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
            D_inv = torch.diag(1.0 / torch.diag(D))

            # Perform Random-walk normalization: D^-1 * A
            ####normalized_A = torch.matmul(D_inv, A)
            I = torch.eye(N).cuda()    
            normalized_A = I - torch.matmul(D_inv, A)      
 
            eigenvalues, eigenvectors = torch.linalg.eigh(normalized_A)     
            # eigenvalues, eigenvectors = torch.linalg.eig(normalized_A)        
            # eigenvalues = eigenvalues.real
            # eigenvectors = eigenvectors.real

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

        # return top_k_eigenvalues[:, 1:], top_k_eigenvectors[:, :, 1:], eigenvalues_l, eigenvectors_l  
        return top_k_eigenvalues, top_k_eigenvectors, eigenvalues_l, eigenvectors_l  

    def forward(self, pts, noaug = False, vis=False, **kwargs):
        neighborhood, center, neighborhood_org = self.group_divider(pts)   

        if (self.method == "MAMBA"):

            x_vis, mask = self.MAE_encoder(neighborhood, center)
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
            top_k_eigenvalues, top_k_eigenvectors, eigenvalues, eigenvectors = self.calc_top_k_eigenvalues_eigenvectors(adjacency_matrix, k=self.k_top_eigenvectors, smallest = self.smallest)
            
            x_vis, sorted_bool_masked_pos_list, sorted_pos_mask, sorted_pos_full, sorted_bool_masked_pos_tensor, sorted_neighborhood, mask_ratio = self.MAE_encoder(neighborhood, center, top_k_eigenvectors, self.k_top_eigenvectors, self.reverse, noaug)
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
            loss1 = self.loss_func(rebuild_points, gt_points) 

            return loss1[0]         
            #return loss1           
