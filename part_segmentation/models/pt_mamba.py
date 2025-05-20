from typing import Optional, Tuple
import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from timm.models.layers import DropPath, trunc_normal_
from logger import get_missing_parameters_message, get_unexpected_parameters_message

from pointnet2_ops import pointnet2_utils
# from knn_cuda import KNN
from pytorch3d.ops import knn_points, sample_farthest_points    
from pointnet2_utils import PointNetFeaturePropagation

from mamba_ssm.modules.mamba_simple import Mamba

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


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
                # Special Scaled Initialization --> There are 2 Layer Norms per Mamba Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class Block(nn.Module):
    def __init__(
            self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False, drop_path=0.
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Mamba block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)

        # drop path 
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (self.drop_path(hidden_states) + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


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
        dtype=None, ):
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


def fps(data, number):
    '''
        data B N 3
        number int
    '''
    fps_idx = pointnet2_utils.furthest_point_sample(data, number)
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()
    return fps_data


class Group(nn.Module):
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
        # center = fps(xyz, self.num_group)  # B G 3
        center = sample_farthest_points(points = xyz, K = self.num_group)     
        center = center[0] 
        # knn to get the neighborhood
        # _, idx = self.knn(xyz, center)  # B G M
        idx = knn_points(center.cuda(), xyz.cuda(), K=self.group_size, return_sorted=False)
        idx = idx.idx
        idx = idx.long()
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center


class Encoder(nn.Module):
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
        feature = self.first_conv(point_groups.transpose(2, 1))
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)
        feature = self.second_conv(feature)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]
        return feature_global.reshape(bs, g, self.encoder_channel)


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


class MixerModelForSegmentation(MixerModel):
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
            drop_path: int = 0.1,
            fetch_idx: Tuple[int] = [3, 7, 11],
            device=None,
            dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(MixerModel, self).__init__()
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

        self.fetch_idx = fetch_idx

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


    def forward(self, input_ids, pos, inference_params=None):
        hidden_states = input_ids  # + pos
        residual = None
        hidden_states = hidden_states + pos
        feature_list = []
        for idx, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
            if idx in self.fetch_idx:
                if not self.fused_add_norm:
                    residual_output = (hidden_states + residual) if residual is not None else hidden_states
                    hidden_states_output = self.norm_f(residual_output.to(dtype=self.norm_f.weight.dtype))
                else:
                    # Set prenorm=False here since we don't need the residual
                    fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
                    hidden_states_output = fused_add_norm_fn(
                        hidden_states,
                        self.norm_f.weight,
                        self.norm_f.bias,
                        eps=self.norm_f.eps,
                        residual=residual,
                        prenorm=False,
                        residual_in_fp32=self.residual_in_fp32,
                    )
                feature_list.append(hidden_states_output)
        return feature_list


class get_model(nn.Module):
    def __init__(self, cls_dim, config=None):
        super().__init__()

        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.cls_dim = cls_dim

        self.group_size = 32
        self.num_group = 128
        # grouper
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        # define the encoder
        self.encoder_dims = 384
        self.encoder = Encoder(encoder_channel=self.encoder_dims)
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )
        self.blocks = MixerModelForSegmentation(d_model=self.trans_dim,
                                                n_layer=self.depth,
                                                rms_norm=config.rms_norm,
                                                drop_path=config.drop_path,
                                                fetch_idx=config.fetch_idx)

        self.drop_out = nn.Dropout(config.drop_out) if "drop_out" in config else nn.Dropout(0)
        self.drop_path_rate = config.drop_path_rate
        self.drop_path_block = DropPath(self.drop_path_rate) if self.drop_path_rate > 0. else nn.Identity()

        self.norm = nn.LayerNorm(self.trans_dim)

        self.label_conv = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                        nn.BatchNorm1d(64),
                                        nn.LeakyReLU(0.2))

        self.propagation_0 = PointNetFeaturePropagation(in_channel=1152 + 3, mlp=[self.trans_dim * 4, 1024])

        self.convs1 = nn.Conv1d(3392, 512, 1)
        self.dp1 = nn.Dropout(0.5)
        self.convs2 = nn.Conv1d(512, 256, 1)
        self.convs3 = nn.Conv1d(256, self.cls_dim, 1)
        self.bns1 = nn.BatchNorm1d(512)
        self.bns2 = nn.BatchNorm1d(256)

        self.relu = nn.ReLU()

        ########## Settings for our method
        self.method = config.method
        self.reverse = config.reverse
        self.k_top_eigenvectors = config.k_top_eigenvectors
        self.smallest = config.smallest 
        self.knn_graph = config.knn_graph 
        self.symmetric = config.symmetric
        self.self_loop = config.self_loop 
        self.alpha = config.alpha 
        self.binary = config.binary    

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
                print('missing_keys')
                print(get_missing_parameters_message(incompatible.missing_keys))
            if incompatible.unexpected_keys:
                print('unexpected_keys')
                print(get_unexpected_parameters_message(incompatible.unexpected_keys))
            print(f'[Mamba] Successful Loading the ckpt from {bert_ckpt_path}')
        else:
            print(f'[Mamba] No ckpt is loaded, training from scratch!')

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

    def sort_points_by_fiedler(self, points, fiedler_vector):

            B, N, _ = points.shape
            _, sorted_indices = torch.sort(fiedler_vector, dim=1)   
            
            expanded_indices = sorted_indices.unsqueeze(-1).expand(-1, -1, 384) 
            
            sorted_points = torch.gather(points, 1, expanded_indices) 
            
            return sorted_points
    
    def sort_points_by_center(self, points, fiedler_vector):

            B, N, _ = points.shape
            _, sorted_indices = torch.sort(fiedler_vector, dim=1)   
            
            expanded_indices = sorted_indices.unsqueeze(-1).expand(-1, -1, 3) 
            
            sorted_points = torch.gather(points, 1, expanded_indices) 
            
            return sorted_points 

    def forward(self, pts, cls_label):
        B, C, N = pts.shape
        pts = pts.transpose(-1, -2)  # B N 3
        # divide the point cloud in the same form. This is important
        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)  # B G N

        pos = self.pos_embed(center)

        if (self.method == "Point_MAMBA"):
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
            center_xx = center.gather(dim=1, index=torch.tile(center_x, (1, 1, center.shape[-1])))
            center_yy = center.gather(dim=1, index=torch.tile(center_y, (1, 1, center.shape[-1])))
            center_zz = center.gather(dim=1, index=torch.tile(center_z, (1, 1, center.shape[-1])))
            group_input_tokens = torch.cat([group_input_tokens_x, group_input_tokens_y, group_input_tokens_z], dim=1)
            pos = torch.cat([pos_x, pos_y, pos_z], dim=1)
            center = torch.cat([center_xx, center_yy, center_zz], dim=1)

            sorted_group_input_tokens = group_input_tokens
            sorted_pos = pos
            sorted_center = center

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

 
            sorted_group_input_tokens = sorted_group_input_tokens_zeros 
            sorted_pos = sorted_pos_zeros 
            sorted_center = sorted_center_zeros

        elif(self.method == "SAST"):    
                
            # create graph
            adjacency_matrix = self.create_graph_from_centers(center, self.knn_graph, self.alpha, self.symmetric, self.self_loop, self.binary)

            # calculate eigenvectors and eigenvalues
            top_k_eigenvalues, top_k_eigenvectors, eigenvalues, eigenvectors = self.calc_top_k_eigenvalues_eigenvectors(adjacency_matrix, k=self.k_top_eigenvectors, smallest = self.smallest)

            # ordering tokens   
            for i in range (self.k_top_eigenvectors):         
                sorted_group_input_tokens = self.sort_points_by_fiedler(group_input_tokens, top_k_eigenvectors[:, :, i]) 
                sorted_pos = self.sort_points_by_fiedler(pos, top_k_eigenvectors[:, :, i])
                sorted_center = self.sort_points_by_center(center, top_k_eigenvectors[:, :, i])

                if (i != 0): 
                    sorted_group_input_tokens = torch.cat((sorted_group_input_tokens_t, sorted_group_input_tokens), 1)         
                    sorted_pos = torch.cat((sorted_pos_t, sorted_pos), 1)  
                    sorted_center = torch.cat((sorted_center_t, sorted_center), 1)  

                sorted_group_input_tokens_t = sorted_group_input_tokens      
                sorted_pos_t = sorted_pos
                sorted_center_t = sorted_center


            if (self.reverse == True):         
                sorted_group_input_tokens_t_reverse = sorted_group_input_tokens_t.flip(1) 
                sorted_pos_t_reverse = sorted_pos_t.flip(1)  
                sorted_center_t_reverse = sorted_center_t.flip(1)  
                sorted_group_input_tokens = torch.cat((sorted_group_input_tokens_t, sorted_group_input_tokens_t_reverse), 1)           
                sorted_pos = torch.cat((sorted_pos_t, sorted_pos_t_reverse), 1)       
                sorted_center = torch.cat((sorted_center_t, sorted_center_t_reverse), 1)       
                
                sorted_group_input_tokens = sorted_group_input_tokens.cuda()
                sorted_pos = sorted_pos.cuda()
                sorted_center = sorted_center.cuda() 


        # final input
        # x = group_input_tokens
        x = sorted_group_input_tokens.cuda()

        feature_list = self.blocks(x, sorted_pos.cuda())

        feature_list = [self.norm(x).transpose(-1, -2).contiguous() for x in feature_list]
        x = torch.cat((feature_list), dim=1)  # 1152
        x_max = torch.max(x, 2)[0]
        x_avg = torch.mean(x, 2)
        x_max_feature = x_max.view(B, -1).unsqueeze(-1).repeat(1, 1, N)
        x_avg_feature = x_avg.view(B, -1).unsqueeze(-1).repeat(1, 1, N)
        cls_label_one_hot = cls_label.view(B, 16, 1)
        cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N)
        x_global_feature = torch.cat((x_max_feature, x_avg_feature, cls_label_feature), 1)

        f_level_0 = self.propagation_0(pts.transpose(-1, -2), sorted_center.transpose(-1, -2), pts.transpose(-1, -2), x)

        x = torch.cat((f_level_0, x_global_feature), 1)
        x = self.relu(self.bns1(self.convs1(x)))
        x = self.dp1(x)
        x = self.relu(self.bns2(self.convs2(x)))
        x = self.convs3(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x  


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        total_loss = F.nll_loss(pred, target)
        return total_loss
