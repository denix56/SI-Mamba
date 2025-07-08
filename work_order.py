import torch.linalg

from tools import builder
from utils.config import *

from models.point_mamba import *


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

        # Precompute eigenvalues if spectral schedule is desired
        #eigvals = torch.linalg.eigvalsh(L)  # (B,N)

        # Determine rank schedule if not provided: default dyadic
        # if rank_schedule is None:
        #     rank_schedule = [max(1, (N + (1<<j) - 1)//(1<<j)) for j in range(J+1)]
        #
        # rank_schedule = []
        # for j in range(J + 1):
        #     cutoff = self.lam_cutoff if self.lam_cutoff is not None else (2.0 / (2 ** j))
        #     # collect count of eigenvalues <= cutoff
        #     rj = int((eigvals <= cutoff).sum().item())
        #     rj = max(1, rj)
        #     rank_schedule.append(rj)



        # Initialize V0 as identity basis for each batch
        V_prev = torch.eye(N, device=device, dtype=dtype).unsqueeze(0).expand(B, N, N)
        V_bases = [V_prev]

        # Chebyshev-like coarse hop: power of T
        lam_max = 2.0
        jj = torch.arange(1, J+1, device=device, dtype=dtype)
        cutoff_j = lam_max / (2 ** jj)
        t_j = math.log(2.0) / cutoff_j
        T = torch.matrix_exp(-t_j[:, None, None, None] * (2 ** (jj[:, None, None, None] - 1)) * L.unsqueeze(0).expand(J, -1, -1, -1))

        eigvals = torch.linalg.eigvalsh(L)
        rank_schedule = (eigvals[:, None, :].expand(-1, J, -1) <= cutoff_j[None, :, None].expand(B, -1, N)).sum(dim=2)
        rank_schedule = torch.clamp(torch.max(rank_schedule, dim=0)[0], min=1)

        assert len(rank_schedule) == J, "rank_schedule must have length J"

        # Build V_j for j=1..J via batched operations
        for j in range(1, J+1):
            k = rank_schedule[j-1]
            M  = torch.matmul(T[j-1], V_prev)              # (B,N,r_prev)
            # Batched SVD
            # 1) form Gram matrix
            C = M.transpose(1, 2) @ M  # (B, r, r)

            # 2) eigh returns ascending
            evals, V = torch.linalg.eigh(C)  # evals[:, 0] is smallest

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
            W.append(Qj)

        # Final scaling basis V_J
        VJ = V_bases[-1]  # (B,N,r_J)
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

    def forward(self, x: Tensor, L: Tensor) -> Tensor:
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

        coeffs /= torch.sqrt((coeffs ** 2).mean(1, keepdim=True) + 1e-6)

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



def create_graph_from_centers(points, k=5, alpha=1, symmetric=False, self_loop=False, binary=False):
    points = points.to('cuda')
    B, N, _ = points.shape

    # Compute pairwise Euclidean distances, shape (B, N, N), on GPU
    dist_matrix = torch.sqrt(torch.sum((points.unsqueeze(2) - points.unsqueeze(1)) ** 2, dim=-1))

    sigma = torch.mean(dist_matrix)

    # Find the k-nearest neighbors for each point, including itself
    distances, indices = torch.topk(-dist_matrix, k=k + 1, largest=True, dim=-1)
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
    if (alpha == 0):
        distances_new = torch.exp(-distances ** 2 / (2 * sigma ** 2))
    else:
        distances_new = torch.exp((-1) * alpha * (distances) ** 2)

    if (binary):
        adjacency_matrix[b_idx, n_idx, indices] = 1.
        if (symmetric):
            adjacency_matrix[b_idx, indices, n_idx] = 1.  # Ensure symmetry
    else:
        adjacency_matrix[b_idx, n_idx, indices] = distances_new
        if (symmetric):
            adjacency_matrix[b_idx, indices, n_idx] = distances_new  # Ensure symmetry

    return adjacency_matrix


if __name__ == '__main__':
    config = cfg_from_yaml_file('cfgs/pretrain.yaml')
    extra_train_dataloader, extra_test_dataloader = builder.dataset_builder_svm(config.dataset.svm)
    config = config.model

    points, label = next(iter(extra_test_dataloader))
    points = points.cuda()
    group_divider = Group(num_group=config.num_group, group_size=config.group_size)
    neighborhood, center, neighborhood_org = group_divider(points)



    save_pts_dir = 'tmp'
    os.makedirs(save_pts_dir, exist_ok=True)

    knn_graph = config.transformer_config.knn_graph
    alpha = config.transformer_config.alpha
    symmetric = config.transformer_config.symmetric
    self_loop = config.transformer_config.self_loop
    binary = config.transformer_config.binary

    adjacency_matrix = create_graph_from_centers(center, knn_graph, alpha, symmetric,
                                                      self_loop, binary)
    orders = None
    laplacian = build_rw_laplacian(adjacency_matrix)
    sgwt = ComplexMeyerSGWT(J=4, K=30, use_complex=True, use_delta=False, jackson=False).cuda()
    coeffs = sgwt(center, laplacian)
    J = sgwt.J  # len(self.sgwt.scales) if self.sgwt.scales is not None else self.sgwt.J
    orders = traversal_order_from_coeffs(coeffs)

    eigvals, eigvecs = torch.linalg.eigh(laplacian)  # eigvals: (B,N), eigvecs: (B,N,N)

    # 2) Extract the Fiedler vector (second column)
    phi1 = eigvecs[:, :, :4]  # (B, N)

    # 3) Canonicalize its sign so phi1[:,0] >= 0
    sign0 = torch.sign(phi1[:, :1])  # (B, 1)

    # 4) Build your first traversal
    order_phi1 = torch.argsort(phi1, dim=1).transpose(1, 2)
    order_sign_phi1 = torch.argsort(phi1 * sign0, dim=1).transpose(1, 2)  # list of point‐indices per batch

    diff_sgwt = DiffusionWaveletSGWT(J=3)
    scatt = GraphScattering(diff_sgwt, act='abs')
    coeffs = scatt(center, laplacian, level=0)
    diff_orders = traversal_order_from_coeffs(coeffs)
    # orders = multi_scale_traversals(coeffs, F=3, J=J+1, k=J+1, tau=tau, aggregate='signed_mean')
    # orders = torch.cat((#order_phi1, order_sign_phi1, torch.stack(orders, dim=1),
    #                        torch.stack(diff_orders, dim=1)), dim=1)
    #orders = torch.cat((order_sign_phi1, ), dim=1)
    orders = diff_orders

    if save_pts_dir is not None:
        sorted_neighborhood = neighborhood.unsqueeze(1).expand(-1, orders.size(1), -1, -1, -1)
        sorted_neighborhood = sorted_neighborhood.gather(dim=2, index=orders.unsqueeze(-1).unsqueeze(-1).expand_as(
            sorted_neighborhood))

        sorted_center = center.unsqueeze(1).expand(-1, orders.size(1), -1, -1)
        sorted_center = sorted_center.gather(dim=2, index=orders.unsqueeze(-1).expand_as(sorted_center))

        np.savez_compressed(os.path.join(save_pts_dir, '0.npz'),
                            center=sorted_center.cpu().numpy(),
                            neighborhood=sorted_neighborhood.cpu().numpy(),
                            orders=orders.cpu().numpy())