"""
cross_mamba_cmi.py

Cross Mamba Interaction (CMI) implementation.

Usage:
    cmi = CrossMambaInteraction(in_ch=2048, D=256, N=128)
    F_rgb_hat, F_x_hat = cmi(F_rgb, F_x)
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossMambaInteraction(nn.Module):
    """
    Cross Mamba Interaction (CMI).

    Args:
        in_ch: channel dimension of stage-5 feature maps (C)
        D: latent projection dimension for sequence tokens (default 256)
        N: SSM hidden/state dimension (default 128)
        eps: small eps used for numerical stability
        clamp_logA_min: lower clamp for log(A) to avoid underflow/overflow when exponentiating
    """

    def __init__(
        self,
        in_ch: int,
        D: int = 256,
        N: int = 128,
        eps: float = 1e-6,
        clamp_logA_min: float = -20.0,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.D = D
        self.N = N
        self.eps = eps
        self.clamp_logA_min = clamp_logA_min

        # Preprocessing: project input channels -> D, with small spatial mixer (1x1 + DW conv)
        self.rgb_pre = nn.Sequential(
            nn.Conv2d(in_ch, D, kernel_size=1, bias=True),
            nn.LayerNorm([D, 1, 1]),  # LayerNorm applied to channels; we'll adapt during forward
            nn.SiLU(),
            nn.Conv2d(D, D, kernel_size=3, padding=1, groups=1, bias=True),  # optional small spatial mixing
        )
        self.x_pre = nn.Sequential(
            nn.Conv2d(in_ch, D, kernel_size=1, bias=True),
            nn.LayerNorm([D, 1, 1]),
            nn.SiLU(),
            nn.Conv2d(D, D, kernel_size=3, padding=1, groups=1, bias=True),
        )

        # Input projection from D -> N (for SSM input)
        self.W_in_rgb = nn.Linear(D, N, bias=True)
        self.W_in_x = nn.Linear(D, N, bias=True)

        # Learnable template parameters A_bar and B_bar (vectors of size N).
        # They are initialized small negative for A_bar (so exp(A_bar*delta) initially < 1).
        self.A_bar = nn.Parameter(torch.randn(N) * -0.05)  # small negative mean
        self.B_bar = nn.Parameter(torch.randn(N) * 0.05)

        # Heads to produce delta_A and delta_B from pooled content per modality
        # Use small MLPs mapping pooled (D) -> N
        self.delta_rgb_A = nn.Sequential(nn.Linear(D, N), nn.SiLU(), nn.Linear(N, N))
        self.delta_rgb_B = nn.Sequential(nn.Linear(D, N), nn.SiLU(), nn.Linear(N, N))
        self.delta_x_A = nn.Sequential(nn.Linear(D, N), nn.SiLU(), nn.Linear(N, N))
        self.delta_x_B = nn.Sequential(nn.Linear(D, N), nn.SiLU(), nn.Linear(N, N))

        # Cross-decoders: decode hidden states of one modality using other modality's decoder
        # C_rgb decodes states into D (but will be used to decode the other's hidden states)
        self.C_rgb = nn.Linear(N, D, bias=True)  # decodes states to D
        self.C_x = nn.Linear(N, D, bias=True)

        # D_* projections (residual from token z_t -> output)
        self.D_rgb = nn.Linear(D, D, bias=True)
        self.D_x = nn.Linear(D, D, bias=True)

        # Output projection: D -> in_ch (project back to original channel dimension)
        self.out_rgb = nn.Conv2d(D, in_ch, kernel_size=1, bias=True)
        self.out_x = nn.Conv2d(D, in_channels=in_ch, kernel_size=1, bias=True)  # same as rgb

        # small layer norms for stability
        self.ln_out_rgb = nn.LayerNorm([in_ch, 1, 1])
        self.ln_out_x = nn.LayerNorm([in_ch, 1, 1])

        # init weights for stability
        self._init_weights()

    def _init_weights(self):
        # Orthogonal-ish or kaiming initializations where appropriate, tiny gains
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ---------------------------
    # Utility helpers
    # ---------------------------
    def _spatial_flatten(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        # x: (B, C, H, W) -> (B, T, C) where T = H*W
        B, C, H, W = x.shape
        T = H * W
        z = x.view(B, C, T).permute(0, 2, 1).contiguous()  # (B, T, C)
        return z, H, W

    def _spatial_unflatten(self, seq: torch.Tensor, H: int, W: int) -> torch.Tensor:
        # seq: (B, T, D) -> (B, D, H, W)
        B, T, D = seq.shape
        assert T == H * W
        x = seq.permute(0, 2, 1).contiguous().view(B, D, H, W)
        return x

    # ---------------------------
    # Core recurrence
    # ---------------------------
    def _compute_A_B(self, pooled: torch.Tensor, delta_A_head: nn.Module, delta_B_head: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        pooled: (B, D)
        Returns:
            A: (B, N) -> values in (0, 1] via exp(clamped(logA))
            B: (B, N) -> can be positive or negative
        Follows: A = exp( delta * A_bar ), B = delta * B_bar
        """
        # delta heads -> (B, N)
        deltaA = delta_A_head(pooled)  # (B, N)
        deltaB = delta_B_head(pooled)  # (B, N)

        # compute logA safely: logA = deltaA * A_bar
        # A_bar shape (N,), broadcast to (B,N)
        logA = deltaA * self.A_bar.unsqueeze(0)  # (B, N)
        # clamp to avoid overflow when exp
        logA = torch.clamp(logA, min=self.clamp_logA_min, max=0.0)
        A = torch.exp(logA)  # (B, N) in (0, 1]

        # B: deltaB * B_bar
        B = deltaB * self.B_bar.unsqueeze(0)  # (B, N)

        return A, B

    def _run_recurrence(self, u: torch.Tensor, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        u: (B, T, N)  -> inputs already projected via W_in
        A: (B, N)      -> recurrence multiplicative coefficients
        B: (B, N)      -> input mixing coefficients
        Returns:
            h: (B, T, N)
        Recurrence:
            h[:,0] = B * u[:,0]
            h[:,t] = A * h[:,t-1] + B * u[:,t]
        Implemented with a python loop over T (T small for top-level features).
        """
        Bsz, T, N = u.shape
        device = u.device

        # prepare output
        h = torch.zeros((Bsz, T, N), dtype=u.dtype, device=device)

        # initial step
        h_prev = B * u[:, 0, :]  # (B, N) elementwise
        h[:, 0, :] = h_prev

        # recurrence loop
        # Note: vectorized broadcasting: A (B,N) * h_prev (B,N)
        for t in range(1, T):
            ut = u[:, t, :]         # (B, N)
            h_t = A * h_prev + B * ut
            h[:, t, :] = h_t
            h_prev = h_t

        return h

    # ---------------------------
    # Forward
    # ---------------------------
    def forward(self, F_rgb: torch.Tensor, F_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[dict]]:
        """
        Args:
            F_rgb: (B, in_ch, H, W)
            F_x:   (B, in_ch, H, W)
        Returns:
            F_rgb_hat: (B, in_ch, H, W)
            F_x_hat:   (B, in_ch, H, W)
            optional_info: dict with intermediate tensors (for debugging/analysis)
        """
        assert F_rgb.dim() == 4 and F_x.dim() == 4
        Bsz, C, H, W = F_rgb.shape
        assert F_x.shape == (Bsz, C, H, W), "Both inputs must share the same shape"

        # ---- Preprocess -> project to D ----
        # Use conv pipeline and then flatten to (B, T, D)
        z_rgb_map = self.rgb_pre(F_rgb)  # (B, D, H, W)
        z_x_map = self.x_pre(F_x)

        z_rgb_flat, H_, W_ = self._spatial_flatten(z_rgb_map)  # (B, T, D)
        z_x_flat, _, _ = self._spatial_flatten(z_x_map)

        T = z_rgb_flat.shape[1]

        # ---- pooled content to produce deltas ----
        # Use global average pool over spatial -> (B, D)
        pooled_rgb = z_rgb_map.mean(dim=(2, 3))  # (B, D)
        pooled_x = z_x_map.mean(dim=(2, 3))

        # compute A and B per modality (B, N)
        A_rgb, B_rgb = self._compute_A_B(pooled_rgb, self.delta_rgb_A, self.delta_rgb_B)
        A_x, B_x = self._compute_A_B(pooled_x, self.delta_x_A, self.delta_x_B)

        # ---- project token embeddings to SSM input dim N: u_t = W_in(z_t) ----
        u_rgb = self.W_in_rgb(z_rgb_flat)  # (B, T, N)
        u_x = self.W_in_x(z_x_flat)

        # ---- run recurrences h_rgb and h_x ----
        h_rgb = self._run_recurrence(u_rgb, A_rgb, B_rgb)  # (B, T, N)
        h_x = self._run_recurrence(u_x, A_x, B_x)

        # ---- cross-decoding: decode hidden states of each modality with the *other* modality's decoder ----
        # y_rgb_t = C_x(h_rgb_t) + D_rgb(z_rgb_t)
        # y_x_t   = C_rgb(h_x_t)   + D_x(z_x_t)
        # shapes: C_*(N->D), D_*(D->D)
        y_rgb_seq = self.C_x(h_rgb.view(-1, self.N)).view(Bsz, T, self.D)  # (B, T, D)
        y_rgb_seq = y_rgb_seq + self.D_rgb(z_rgb_flat)  # broadcast (B,T,D)

        y_x_seq = self.C_rgb(h_x.view(-1, self.N)).view(Bsz, T, self.D)
        y_x_seq = y_x_seq + self.D_x(z_x_flat)

        # ---- reshape back to (B, D, H, W) ----
        y_rgb_map = self._spatial_unflatten(y_rgb_seq, H, W)  # (B, D, H, W)
        y_x_map = self._spatial_unflatten(y_x_seq, H, W)

        # ---- project back to in_ch channels ----
        out_rgb = self.out_rgb(y_rgb_map)  # (B, in_ch, H, W)
        out_x = self.out_x(y_x_map)

        # optional layernorm (channel-wise). Use 1x1 LN wrapper (LayerNorm expects last dims but we give shape)
        # Reshape to (B, in_ch, 1, 1) style LayerNorm expecting [C,1,1] param format used in init.
        out_rgb = self.ln_out_rgb(out_rgb)
        out_x = self.ln_out_x(out_x)

        # Return optionally useful debug info
        info = {
            "z_rgb_map": z_rgb_map.detach(),
            "z_x_map": z_x_map.detach(),
            "A_rgb": A_rgb.detach(),
            "B_rgb": B_rgb.detach(),
            "A_x": A_x.detach(),
            "B_x": B_x.detach(),
            "h_rgb": h_rgb.detach(),
            "h_x": h_x.detach(),
        }

        return out_rgb, out_x, info


# -------------------------
# Quick smoke test
# -------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B = 2
    C = 2048
    H = 8
    W = 8

    model = CrossMambaInteraction(in_ch=C, D=256, N=128).to(device)
    F_rgb = torch.randn(B, C, H, W, device=device)
    F_x = torch.randn(B, C, H, W, device=device)

    with torch.no_grad():
        out_rgb, out_x, info = model(F_rgb, F_x)

    print("in:", F_rgb.shape, "out_rgb:", out_rgb.shape, "out_x:", out_x.shape)
    assert out_rgb.shape == F_rgb.shape
    assert out_x.shape == F_x.shape
    print("A_rgb shape:", info["A_rgb"].shape, "h_rgb shape:", info["h_rgb"].shape)
    print("Smoke test passed.")
