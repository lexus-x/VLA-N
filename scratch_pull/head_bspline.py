"""BEAST-style B-spline / compact-coefficient action head.

BEAST (refs/beast, intuitive-robots/beast_calvin) represents an action chunk
with far fewer numbers than raw timesteps by fitting each DOF's trajectory
with a cubic B-spline and tokenizing its K coefficients -- see
refs/beast/beast/models/beast_florence.py, `action_tokenizer` config
(num_dof/num_basis/seq_len) and the `zhouhongyi/beast` HF processor it loads
for `encode_discrete`/`decode_discrete`. That custom processor's code is
hosted on the HF Hub and is not vendored in the local refs/beast checkout, so
the B-spline basis itself (uniform-knot cubic Cox-de Boor basis) is
reimplemented here from the standard formula -- this is the "BEAST spline
basis functions" helper piece the task anticipated needing to be ported.

Simplification vs. native BEAST: BEAST's actual training loss is
cross-entropy over *discretized* B-spline-coefficient tokens decoded by a
Florence-2 LM head (mirroring VLA-0's AR-token recipe, see
`compute_llm_outputs` in beast_florence.py: `masked_lm_loss`). Per the
matched-protocol spec for this smoke-test phase, we keep only the core
"compact coefficient" idea -- predict K continuous B-spline coefficients per
DOF -- and train with a direct regression (L1) loss against coefficients
obtained by least-squares-fitting the ground-truth action chunk onto the
same basis, instead of adding a second discretization+CE step on top.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from heads.backbone import ACTION_DIM, CHUNK_LEN

NUM_BASIS = 4  # K compact coefficients per DOF (<< CHUNK_LEN)


def cubic_bspline_basis(num_points: int, num_basis: int, degree: int = 3) -> torch.Tensor:
    """Uniform-knot B-spline basis matrix of shape (num_points, num_basis),
    evaluated via the standard Cox-de Boor recursion (de Boor, 1978).
    """
    num_knots = num_basis + degree + 1
    inner = torch.linspace(0.0, 1.0, num_knots - 2 * degree)
    knots = torch.cat([inner[0].repeat(degree), inner, inner[-1].repeat(degree)])
    t = torch.linspace(0.0, 1.0, num_points)

    def basis(i, k, t):
        if k == 0:
            left = knots[i] <= t
            right = t < knots[i + 1]
            right = right | ((t == knots[-1]) & (knots[i + 1] >= knots[-1]))
            return (left & right).float()
        term1 = torch.zeros_like(t)
        denom1 = knots[i + k] - knots[i]
        if denom1 > 0:
            term1 = (t - knots[i]) / denom1 * basis(i, k - 1, t)
        term2 = torch.zeros_like(t)
        denom2 = knots[i + k + 1] - knots[i + 1]
        if denom2 > 0:
            term2 = (knots[i + k + 1] - t) / denom2 * basis(i + 1, k - 1, t)
        return term1 + term2

    cols = [basis(i, degree, t) for i in range(num_basis)]
    return torch.stack(cols, dim=1)  # (num_points, num_basis)


class BSplineHead(nn.Module):
    def __init__(self, hidden_size, num_basis=NUM_BASIS, chunk_len=CHUNK_LEN, action_dim=ACTION_DIM):
        super().__init__()
        self.num_basis = num_basis
        self.chunk_len = chunk_len
        self.action_dim = action_dim

        basis = cubic_bspline_basis(chunk_len, num_basis)  # (T, K)
        self.register_buffer("basis", basis)
        self.register_buffer("basis_pinv", torch.linalg.pinv(basis))  # (K, T)
        self.proj = nn.Linear(hidden_size, action_dim * num_basis)

    def forward(self, pooled: torch.Tensor, target_actions: torch.Tensor):
        B = pooled.shape[0]
        pred_coeffs = self.proj(pooled).reshape(B, self.action_dim, self.num_basis)

        # Target coefficients: least-squares fit of the GT action chunk onto
        # the same fixed spline basis (BEAST's "compact coefficient" target).
        target_t = target_actions.transpose(1, 2)  # (B, D, T)
        target_coeffs = target_t @ self.basis_pinv.t()  # (B, D, K)

        loss = F.l1_loss(pred_coeffs, target_coeffs)

        pred_action = (pred_coeffs @ self.basis.t()).transpose(1, 2)  # (B, T, D)
        return pred_action, loss

    @torch.no_grad()
    def generate(self, pooled: torch.Tensor) -> torch.Tensor:
        """pred_action above is a direct regression off `pooled` alone (never
        reads target_actions), so eval-time generation is just forward() with
        a dummy target -- no autoregression needed for this head."""
        B = pooled.shape[0]
        dummy = torch.zeros(B, self.chunk_len, self.action_dim, device=pooled.device)
        return self.forward(pooled, dummy)[0]
