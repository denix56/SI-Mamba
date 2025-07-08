import torch
from torch import nn
from torch import Tensor

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


def _sinkhorn_uv(C: Tensor, tau: float, n_iters: int) -> (Tensor, Tensor, Tensor):
    """
    Run Sinkhorn iterations to convergence, returning kernel K and scaling vectors u, v.
    """
    K = torch.exp(-C / tau)
    B, N, _ = C.shape
    u = torch.full((B, N), 1.0 / N, device=C.device, dtype=C.dtype)
    v = torch.full((B, N), 1.0 / N, device=C.device, dtype=C.dtype)
    for _ in range(n_iters):
        u = 1.0 / (K.bmm(v.unsqueeze(-1)).squeeze(-1))
        v = 1.0 / (K.transpose(-2, -1).bmm(u.unsqueeze(-1)).squeeze(-1))
    return K, u, v


def _run_sinkhorn(C: Tensor, tau: float, n_iters: int) -> Tensor:
    """Compute soft permutation P = diag(u) K diag(v)."""
    K, u, v = _sinkhorn_uv(C, tau, n_iters)
    return u.unsqueeze(-1) * K * v.unsqueeze(-2)


def implicit_sinkhorn_grad(K: Tensor, u: Tensor, v: Tensor, gradP: Tensor, tau: float) -> Tensor:
    """
    Compute implicit gradient dC using the implicit function theorem.
    """
    B, N, _ = K.shape
    # Marginals: a = K v, b = K^T u
    a = torch.bmm(K, v.unsqueeze(-1)).squeeze(-1)
    b = torch.bmm(K.transpose(-2, -1), u.unsqueeze(-1)).squeeze(-1)

    # Gradients of loss w.r.t. u, v
    g_u = (gradP * K * v.unsqueeze(1)).sum(dim=2)
    g_v = (gradP * K * u.unsqueeze(2)).sum(dim=1)

    # Build blocks of F_x^T
    F11 = torch.diag_embed(a)                   # diag(Kv)
    F12 = K * u.unsqueeze(-1)                   # diag(u)K
    F21 = K.transpose(-2, -1) * v.unsqueeze(-1) # diag(v)K^T
    F22 = torch.diag_embed(b)                   # diag(K^T u)

    # Assemble F_x^T (B, 2N, 2N)
    top = torch.cat([F11, F12], dim=2)
    bottom = torch.cat([F21, F22], dim=2)
    F_T = torch.cat([top, bottom], dim=1)

    # Assemble g = [g_u; g_v]
    g = torch.cat([g_u, g_v], dim=1)  # (B, 2N)

    # Solve F_x^T lambda = g
    lam = torch.linalg.solve(F_T, g.unsqueeze(-1)).squeeze(-1)
    lam_r, lam_c = lam[:, :N], lam[:, N:]

    # Compute gradC
    factor = lam_r.unsqueeze(2) + lam_c.unsqueeze(1)  # (B, N, N)
    gradC = (u.unsqueeze(2) * K * v.unsqueeze(1)) * factor / tau
    return gradC


class SinkhornFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C: Tensor, tau: float, n_iters: int) -> Tensor:
        ctx.tau = tau
        ctx.n_iters = n_iters
        # Compute K, u, v without building a graph
        with torch.no_grad():
            K, u, v = _sinkhorn_uv(C, tau, n_iters)
        ctx.save_for_backward(K, u, v)
        return u.unsqueeze(-1) * K * v.unsqueeze(-2)

    @staticmethod
    def backward(ctx, gradP: Tensor) -> tuple:
        K, u, v = ctx.saved_tensors
        tau = ctx.tau
        gradC = implicit_sinkhorn_grad(K, u, v, gradP, tau)
        return gradC, None, None


def sinkhorn_perm(C: Tensor, tau: float = 1.0, n_iters: int = 20) -> Tensor:
    """
    Compute soft permutation P from cost matrix C.
    """
    return SinkhornFunction.apply(C, tau, n_iters)


def sinkhorn_sort(z: Tensor, tau: float = 1.0, n_iters: int = 20) -> Tensor:
    """
    Differentiable sorting of input z via Sinkhorn.

    Args:
        z: input tensor of shape (B, N)
        tau: temperature
        n_iters: Sinkhorn iterations

    Returns:
        P: soft permutation matrix of shape (B, N, N)
        z_sorted: approximated sorted tensor of shape (B, N)
    """
    if z.dim() != 2:
        raise ValueError(f"Expected 2D input (B, N), got {z.shape}")
    B, N = z.shape
    # Hard sort each row
    y_sorted, _ = torch.sort(z, dim=1)
    # Build cost matrix C_{b,i,j} = |z_{b,i} - y_{b,j}|
    z_i = z.unsqueeze(-1)       # (B, N, 1)
    y_j = y_sorted.unsqueeze(1) # (B, 1, N)
    C = torch.abs(z_i - y_j)    # (B, N, N)
    # Compute soft permutation
    P = sinkhorn_perm(C, tau, n_iters)
    # Apply to inputs
    return P


class StochasticNeuralSortPermuter(nn.Module):
    def __init__(self, tau_2: float = 4.0):
        super().__init__()
        self.tau_2 = tau_2

    def forward(self, z: torch.Tensor, tau: float, hard: bool = True) -> torch.Tensor:
        """
        z: [B, N]  — learned log-scores for each patch
        returns:
          P: [B, N, N]  — a soft permutation matrix, stochastically sampled
        """
        B, dim = z.shape
        eps = torch.finfo(z.dtype).eps
        # 1) sample Gumbel noise
        g = -torch.log(-torch.log(torch.rand_like(z) + eps) + eps)
        z_tilde = z + tau * g  # [B, N]

        pi = torch.argsort(z_tilde, dim=1)
        P_hat = torch.zeros((B, dim, dim), device=z.device, dtype=z.dtype).scatter_(2, index=pi.unsqueeze(-1), value=1)

        # P_hat = sinkhorn_sort(z_tilde, self.tau_2)
        #
        # # 2) build pairwise absolute-diff matrix
        # #    shape [B, N, N]: |z_tilde_i - z_tilde_j|
        # # A = torch.abs(z_tilde.unsqueeze(2) - z_tilde.unsqueeze(1))
        # # B = A.sum(dim=2, keepdim=True).expand(-1, -1, A.size(1))
        # # scaling = (dim + 1 - 2 * torch.arange(1, dim + 1, device=z.device, dtype=z.dtype))
        # # C = torch.matmul(z_tilde.unsqueeze(-1), scaling.unsqueeze(0))
        # #
        # # P_max = (C - B).transpose(1, 2)
        # # P_hat = torch.softmax(P_max / self.tau_2, dim=-1)
        # if hard:
        #     P_indices = P_hat.argsort(dim=-1, descending=True) + 1
        #     indices = []
        #     for i in range(P_hat.size(-1)):
        #         mask = (P_indices[:, i] != 0).int()
        #         idx = mask.argmax(dim=1, keepdim=True)
        #         real_idx = torch.gather(P_indices[:, i], dim=1, index=idx)
        #         P_indices[:, i+1:][P_indices[:, i+1:] == real_idx.unsqueeze(-1)] = 0
        #         indices.append(real_idx)
        #     indices = torch.stack(indices, dim=1) - 1
        #     P_hard = torch.zeros_like(P_hat).scatter_(-1, indices, 1.0)
        #     #P_hard = torch.zeros_like(P_hat).scatter(-1, P_hat.argmax(dim=-1, keepdim=True), 1.0)
        #     P_hat = P_hard + (P_hat - P_hat.detach())
        return P_hat
