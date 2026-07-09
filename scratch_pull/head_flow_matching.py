"""SmolVLA-style flow-matching action head.

Reuses lerobot's own flow-matching machinery directly:
`create_sinusoidal_pos_embedding` (imported straight from
lerobot/policies/smolvla/modeling_smolvla.py) for the diffusion-timestep
embedding, and the exact flow-matching recipe from `VLAFlowMatching.forward`
in that same file: sample noise + Beta(1.5, 1.0) time in [0.001, 1.0],
x_t = t*noise + (1-t)*actions, target velocity u_t = noise - actions,
predict v_t, loss = MSE(u_t, v_t). Sampling mirrors
`VLAFlowMatching.sample_actions`'s Euler integration (num_steps steps,
dt = -1/num_steps, x_t += dt * v_t from t=1 down to t=0).

Simplification vs. native SmolVLA: SmolVLA's action expert is a full
Gemma-style transformer (`SmolVLMWithExpertModel`) that cross-attends into
the VLM's OWN KV cache -- i.e. it embeds its own image/language prefix
rather than consuming another backbone's hidden state. Instantiating that
here would mean loading a second ~500M-param VLM copy, defeating the "one
shared frozen backbone" requirement of this harness. So the action-expert
*transformer* is replaced with a small MLP (mirroring VLAFlowMatching's own
action_in_proj / action_time_mlp_in / action_time_mlp_out / action_out_proj
projections almost verbatim) conditioned on the shared backbone's pooled
hidden state instead of a cross-attended VLM prefix -- the flow-matching
math itself (noise schedule, loss, sampler) is untouched.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from lerobot.policies.smolvla.modeling_smolvla import create_sinusoidal_pos_embedding

from heads.backbone import ACTION_DIM, CHUNK_LEN

MIN_PERIOD, MAX_PERIOD = 4e-3, 4.0


class FlowMatchingHead(nn.Module):
    def __init__(self, hidden_size, expert_hidden=256, action_dim=ACTION_DIM, chunk_len=CHUNK_LEN):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_len = chunk_len
        self.expert_hidden = expert_hidden

        self.cond_proj = nn.Linear(hidden_size, expert_hidden)
        self.action_in_proj = nn.Linear(action_dim, expert_hidden)
        self.action_time_mlp_in = nn.Linear(expert_hidden * 2, expert_hidden)
        self.action_time_mlp_out = nn.Linear(expert_hidden, expert_hidden)
        self.action_out_proj = nn.Linear(expert_hidden, action_dim)

    def _velocity(self, pooled: torch.Tensor, x_t: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        action_emb = self.action_in_proj(x_t)  # (B, T, H)
        time_emb = create_sinusoidal_pos_embedding(
            time, self.expert_hidden, MIN_PERIOD, MAX_PERIOD, device=x_t.device
        ).to(action_emb.dtype)
        time_emb = time_emb[:, None, :].expand_as(action_emb)
        cond = self.cond_proj(pooled)[:, None, :].expand_as(action_emb)

        h = self.action_time_mlp_in(torch.cat([action_emb, time_emb], dim=-1))
        h = F.silu(h) + cond  # fuse shared-backbone conditioning
        h = self.action_time_mlp_out(h)
        return self.action_out_proj(h)

    def forward(self, pooled: torch.Tensor, target_actions: torch.Tensor):
        B = target_actions.shape[0]
        device = target_actions.device

        noise = torch.randn_like(target_actions)
        time = torch.distributions.Beta(1.5, 1.0).sample((B,)).to(device) * 0.999 + 0.001
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * target_actions
        u_t = noise - target_actions

        v_t = self._velocity(pooled, x_t, time)
        loss = F.mse_loss(u_t, v_t)

        with torch.no_grad():
            pred_action = self.sample(pooled, batch_size=B, device=device)
        return pred_action, loss

    @torch.no_grad()
    def sample(self, pooled: torch.Tensor, batch_size=1, device="cpu", num_steps=10) -> torch.Tensor:
        """Euler-integrate the flow from noise to actions, mirroring
        VLAFlowMatching.sample_actions.
        """
        x_t = torch.randn(batch_size, self.chunk_len, self.action_dim, device=device)
        dt = -1.0 / num_steps
        for step in range(num_steps):
            t = torch.full((batch_size,), 1.0 + step * dt, device=device)
            v_t = self._velocity(pooled, x_t, t)
            x_t = x_t + dt * v_t
        return x_t

    @torch.no_grad()
    def generate(self, pooled: torch.Tensor) -> torch.Tensor:
        """`sample` above already never reads target_actions -- alias for the
        uniform head.generate(pooled) eval-loop interface."""
        return self.sample(pooled, batch_size=pooled.shape[0], device=pooled.device)
