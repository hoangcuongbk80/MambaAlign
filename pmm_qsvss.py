"""

Per-Modal Mamba Module (PMM) with QuadSnake Visual State-Space Stack (QSVSS)
- QuadSnake indexing: row/row-reverse/diagonal/diagonal-reverse serpentine orders
- QSVSSBlock: LayerNorm -> 1x1 projection -> DepthwiseConv -> split into 4 routes ->
             SSM per-route on sequences -> fold -> 1x1 project back -> residual + FFN
- PerModalMamba: stacks multiple QSVSSBlocks

Expectations:
- Input feature shape: (B, C, H, W)
- SSM API expected: module(seq: Tensor[B, T, C]) -> Tensor[B, T, C]
    (i.e., takes sequences, returns same shape)
- If you have an S6/S4 implementation, pass a factory that constructs an SSM instance.
"""

from typing import Callable, Dict, List, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# QuadSnake Index utilities
# -------------------------
class QuadSnakeIndex:
    """
    Build and cache four QuadSnake index permutations for a given H,W.
    Each index is a LongTensor of length T = H*W containing flattened indices in the chosen scan order.

    Routes:
      0: row-serpentine forward  (left->right on even rows; serpentine)
      1: row-serpentine reverse  (reverse of route 0)
      2: diagonal-serpentine forward (anti-diagonals, serpentine)
      3: diagonal-serpentine reverse (reverse of route 2)
    """

    _CACHE: Dict[Tuple[int, int], List[torch.LongTensor]] = {}

    @classmethod
    def get_indices(cls, H: int, W: int, device: Optional[torch.device] = None) -> List[torch.LongTensor]:
        key = (H, W)
        if key in cls._CACHE:
            idxs = cls._CACHE[key]
        else:
            T = H * W
            # route 0: row-serpentine forward
            r0 = []
            for i in range(H):
                if i % 2 == 0:
                    cols = range(0, W)
                else:
                    cols = range(W - 1, -1, -1)
                for j in cols:
                    r0.append(i * W + j)

            # route 1: reverse of route0
            r1 = list(reversed(r0))

            # route 2: diagonal-serpentine forward (anti-diagonals)
            r2 = []
            for s in range(H + W - 1):  # s = i + j
                # i ranges
                i_min = max(0, s - (W - 1))
                i_max = min(H - 1, s)
                coords = []
                for i in range(i_min, i_max + 1):
                    j = s - i
                    coords.append((i, j))
                # serpentine direction alternates with s
                if s % 2 == 0:
                    coords_iter = coords
                else:
                    coords_iter = reversed(coords)
                for (i, j) in coords_iter:
                    r2.append(i * W + j)

            # route 3: reverse of route2
            r3 = list(reversed(r2))

            # convert to tensors
            idxs = [torch.LongTensor(r).contiguous() for r in (r0, r1, r2, r3)]
            cls._CACHE[key] = idxs

        # move to device if provided
        if device is not None:
            return [t.to(device) for t in idxs]
        return idxs


# -------------------------
# SSM Interface & fallback
# -------------------------
class SSMInterface(nn.Module):
    """
    Minimal interface expected for an SSM module:
    - forward(seq: Tensor[B, T, C]) -> Tensor[B, T, C]
    - implementers: S4/S6 modules, or the fallback below
    """
    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("SSMInterface must implement forward(seq: Tensor[B,T,C])")


class FallbackCausalConv1D(SSMInterface):
    """
    Small causal Conv1D used as a fallback SSM-like operator for testing.
    - Applies a depthwise (per-channel) causal conv along T.
    - Maintains input dims.
    """

    def __init__(self, dim: int, kernel_size: int = 9, groups: int = None):
        super().__init__()
        groups = dim if groups is None else groups
        self.dim = dim
        self.kernel_size = kernel_size
        # conv1d expects shape (B, C, T)
        self.conv = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=kernel_size,
                              padding=kernel_size - 1, groups=groups, bias=True)
        # we'll ensure causality by slicing off the extra right padding
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity="relu")

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        # seq: (B, T, C) -> (B, C, T)
        B, T, C = seq.shape
        x = seq.permute(0, 2, 1)  # (B, C, T)
        x = self.conv(x)  # (B, C, T + padding)
        # causal trim: keep last T timesteps produced by conv with padding at left
        x = x[:, :, :T]
        x = x.permute(0, 2, 1)  # (B, T, C)
        return x


# -------------------------
# QSVSS Block
# -------------------------
class QSVSSBlock(nn.Module):
    """
    One QuadSnake Visual State-Space Stack block.

    Args:
      in_ch: input channels (C)
      route_factor: how many channels per route rC = in_ch // route_factor (paper uses 4 routes -> route_factor = 4)
      ssm_factory: Callable[[int], SSMInterface] -> constructs an SSM for a given channel dimension
                   If None, a fallback small causal conv is used.
      use_ln: whether to use LayerNorm over channels before projection
      dw_kernel: depthwise conv kernel size (spatial local mixer)
      sequential_routes: if True, process routes sequentially to reduce peak memory
    """

    def __init__(
        self,
        in_ch: int,
        route_factor: int = 4,
        ssm_factory: Optional[Callable[[int], SSMInterface]] = None,
        use_ln: bool = True,
        dw_kernel: int = 3,
        sequential_routes: bool = False,
    ):
        super().__init__()
        assert in_ch % route_factor == 0, "in_ch must be divisible by route_factor"
        self.in_ch = in_ch
        self.route_factor = route_factor
        self.rC = in_ch // route_factor
        self.use_ln = use_ln
        self.sequential_routes = sequential_routes

        # LayerNorm applied per spatial token over channels; we operate on (B,H,W,C) then permute back
        if use_ln:
            self.ln = nn.LayerNorm(in_ch, elementwise_affine=True)
        else:
            self.ln = nn.Identity()

        # project channels down to rC (we will then apply depthwise spatial conv)
        self.proj1 = nn.Conv2d(in_ch, self.rC, kernel_size=1, bias=True)
        # depthwise conv for local spatial mixing
        self.dw = nn.Conv2d(self.rC, self.rC, kernel_size=dw_kernel, padding=dw_kernel // 2,
                            groups=self.rC, bias=True)
        # nonlinearity
        self.act = nn.SiLU()

        # SSM modules per route: create using factory when needed
        self.ssm_factory = ssm_factory
        # we will lazily create SSM modules per route when forward sees a sequence length
        self._ssm_modules: List[Optional[SSMInterface]] = [None] * route_factor

        # project back from 4*rC -> in_ch
        self.proj_back = nn.Conv2d(self.rC * route_factor, in_ch, kernel_size=1, bias=True)

        # small FFN (pointwise MLP) as residual
        ffn_hidden = max(4 * in_ch, in_ch * 2)
        self.ffn = nn.Sequential(
            nn.LayerNorm(in_ch),
            nn.Conv2d(in_ch, ffn_hidden, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(ffn_hidden, in_ch, kernel_size=1),
        )

        # caching for indices (device-specific handled during forward)
        self._index_cache: Dict[Tuple[int, int, torch.device], List[torch.LongTensor]] = {}

    def _ensure_ssm(self, idx: int, dim: int, device: torch.device):
        """
        Ensure ssm module for a route exists and on correct device.
        If ssm_factory is provided, call it with `dim` to construct SSM; else use fallback.
        """
        if self._ssm_modules[idx] is None or next(self._ssm_modules[idx].parameters(), None) is None:
            # create new
            if self.ssm_factory is None:
                ssm = FallbackCausalConv1D(dim)
            else:
                ssm = self.ssm_factory(dim)
            ssm.to(device)
            self._ssm_modules[idx] = ssm
        else:
            # ensure device
            self._ssm_modules[idx].to(device)

    def _get_indices(self, H: int, W: int, device: torch.device) -> List[torch.LongTensor]:
        key = (H, W, device)
        if key not in self._index_cache:
            idxs = QuadSnakeIndex.get_indices(H, W, device=device)
            self._index_cache[key] = idxs
        return self._index_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x: Tensor[B, C, H, W]
        Returns:
          out: Tensor[B, C, H, W]  (residual-style)
        """
        B, C, H, W = x.shape
        device = x.device
        # 1) Pre-norm (LayerNorm over channel dim); LayerNorm expects (..., C)
        # Permute to (B, H, W, C)
        x_perm = x.permute(0, 2, 3, 1).contiguous()
        x_ln = self.ln(x_perm)
        # back to (B, C, H, W) for convs
        x_ln_conv = x_ln.permute(0, 3, 1, 2).contiguous()

        # 2) Project to rC channels and local spatial mixing
        u = self.proj1(x_ln_conv)          # (B, rC, H, W)
        u = self.dw(u)                     # (B, rC, H, W)
        u = self.act(u)

        # 3) Unfold to sequence tokens and apply SSM per route
        # convert to (B, T, rC) with T = H*W
        T = H * W
        u_flat = u.permute(0, 2, 3, 1).contiguous().view(B, T, self.rC)  # (B, T, rC)

        idxs = self._get_indices(H, W, device)
        route_outs = []
        # process each route either sequentially or in parallel
        if self.sequential_routes:
            # process routes one-by-one to reduce peak memory
            for r, idx in enumerate(idxs):
                self._ensure_ssm(r, self.rC, device)
                # reorder tokens according to route's permutation
                seq = u_flat[:, idx, :]        # (B, T, rC)
                out_seq = self._ssm_modules[r](seq)  # (B, T, rC)
                # fold back: we need inverse permutation mapping flat_pos -> route_position
                # compute inverse on the CPU once (but here small)
                inv = torch.empty_like(idx)
                inv[idx] = torch.arange(T, device=device, dtype=idx.dtype)
                folded = out_seq[:, inv, :].view(B, H, W, self.rC).permute(0, 3, 1, 2).contiguous()
                route_outs.append(folded)
        else:
            # parallel processing: build a batch with stacked sequences for all routes
            seqs = []
            for r, idx in enumerate(idxs):
                self._ensure_ssm(r, self.rC, device)
                seqs.append(u_flat[:, idx, :])  # (B, T, rC)
            # apply SSMs
            out_seqs = []
            for r, seq in enumerate(seqs):
                out_seq = self._ssm_modules[r](seq)
                out_seqs.append(out_seq)
            # fold each
            for r, out_seq in enumerate(out_seqs):
                idx = idxs[r]
                inv = torch.empty_like(idx)
                inv[idx] = torch.arange(T, device=device, dtype=idx.dtype)
                folded = out_seq[:, inv, :].view(B, H, W, self.rC).permute(0, 3, 1, 2).contiguous()
                route_outs.append(folded)

        # 4) Concat routes along channels and project back
        cat = torch.cat(route_outs, dim=1)  # (B, 4*rC, H, W)
        sproj = self.proj_back(cat)         # (B, C, H, W)

        # 5) Residual + FFN
        x_res = x + sproj
        x_out = x_res + self.ffn(x_res)

        return x_out


# -------------------------
# PerModalMamba (stack)
# -------------------------
class PerModalMamba(nn.Module):
    """
    Stack of QSVSS blocks applied to a per-modality feature map.

    Args:
      in_ch: channel dimension of incoming feature map
      num_blocks: how many QSVSS blocks to stack
      route_factor: channels split factor (default 4)
      ssm_factory: optional factory to build SSM modules
      other args pass to QSVSSBlock
    """

    def __init__(
        self,
        in_ch: int,
        num_blocks: int = 4,
        route_factor: int = 4,
        ssm_factory: Optional[Callable[[int], SSMInterface]] = None,
        use_ln: bool = True,
        dw_kernel: int = 3,
        sequential_routes: bool = False,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            QSVSSBlock(
                in_ch=in_ch,
                route_factor=route_factor,
                ssm_factory=ssm_factory,
                use_ln=use_ln,
                dw_kernel=dw_kernel,
                sequential_routes=sequential_routes,
            )
            for _ in range(num_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply stacked QSVSS blocks.

        Input: x (B, C, H, W)
        Output: same shape
        """
        out = x
        for blk in self.blocks:
            out = blk(out)
        return out


# -------------------------
# Quick Smoke Test
# -------------------------
if __name__ == "__main__":
    # smoke test - ensure shapes are preserved and routes work
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B = 2
    C = 256
    H = 16
    W = 16

    # simple SSM factory that creates small causal conv SSMs
    def ssm_factory(dim: int) -> SSMInterface:
        return FallbackCausalConv1D(dim, kernel_size=9)

    pmm = PerModalMamba(in_ch=C, num_blocks=2, route_factor=4, ssm_factory=ssm_factory, sequential_routes=False)
    pmm.to(device)
    x = torch.randn(B, C, H, W, device=device)

    with torch.no_grad():
        y = pmm(x)
    print("input:", x.shape, "output:", y.shape)
    assert y.shape == x.shape
    print("Smoke test passed.")
