"""
Implements a two-branch backbone for RGB + auxiliary modality X:
- Uses torchvision.models.wide_resnet50_2 as base
- Shares Conv2d weight parameters across the two branches
- Keeps separate BatchNorm (and other affine) parameters per modality
- Returns multi-scale features: stage3, stage4, stage5
- Provides helper utilities to get unique parameter groups and LR multipliers
"""

from typing import Tuple, Iterable, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


def _pairwise_modules(m1: nn.Module, m2: nn.Module):
    """
    Generator yielding module pairs in traversal order.
    This relies on both models having identical topology (true for two wide_resnet50_2 instances).
    """
    yield from zip(m1.modules(), m2.modules())


def _is_conv(module: nn.Module) -> bool:
    return isinstance(module, nn.Conv2d)


def _is_bn(module: nn.Module) -> bool:
    return isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm))


class RGBXBackbone(nn.Module):
    """
    RGB + X backbone wrapper.

    - rgb_in_ch: number of channels for RGB input (usually 3)
    - x_in_ch: number of channels for auxiliary modality (e.g., depth=1, thermal=1, normals=3)
    - pretrained: whether to load torchvision pretrained weights for wide_resnet50_2 (imagenet)
    - share_first_conv_when_inch_equal: if True and x_in_ch == rgb_in_ch, the first conv is tied.
    """

    def __init__(
        self,
        x_in_ch: int = 1,
        pretrained: bool = True,
        backbone_name: str = "wide_resnet50_2",
        freeze_bn_stats: bool = False,
    ):
        super().__init__()
        assert backbone_name in {"wide_resnet50_2"}, "Currently implemented only for wide_resnet50_2"

        # create two backbone instances (RGB and X)
        self.rgb_net = tvm.__dict__[backbone_name](pretrained=pretrained, progress=True)
        self.x_net = tvm.__dict__[backbone_name](pretrained=pretrained, progress=True)

        # store input channel for X
        self.x_in_ch = x_in_ch
        self.rgb_in_ch = 3

        # adapt first conv of x_net if input channels differ
        if x_in_ch != 3:
            self._adapt_first_conv_for_x(self.x_net, x_in_ch, self.rgb_net)

        # Tie CONV weights across the two nets (share parameters).
        # We'll iterate through modules and make conv params the same Parameter object.
        self._tie_conv_weights(self.rgb_net, self.x_net, tie_first_conv=(x_in_ch == 3))

        # Optionally freeze batchnorm running stats (useful if you want frozen BN at finetune)
        if freeze_bn_stats:
            self._freeze_bn_running_stats(self.rgb_net)
            self._freeze_bn_running_stats(self.x_net)

    # --------------------------
    # Forward / feature outputs
    # --------------------------
    def forward(self, rgb: torch.Tensor, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for both modalities.

        Returns dict:
            {
                "rgb_stage3": Tensor (B, C3, H3, W3),
                "rgb_stage4": Tensor (B, C4, H4, W4),
                "rgb_stage5": Tensor (B, C5, H5, W5),
                "x_stage3":   Tensor ...
            }

        Mapping from ResNet blocks:
            stage3 -> layer2
            stage4 -> layer3
            stage5 -> layer4
        """
        assert rgb.dim() == 4 and x.dim() == 4, "Inputs must be (B,C,H,W)"
        # forward rgb branch, manually extract features
        rgb_feats = self._forward_single(self.rgb_net, rgb)
        x_feats = self._forward_single(self.x_net, x)

        out = {
            "rgb_stage3": rgb_feats["layer2"],
            "rgb_stage4": rgb_feats["layer3"],
            "rgb_stage5": rgb_feats["layer4"],
            "x_stage3": x_feats["layer2"],
            "x_stage4": x_feats["layer3"],
            "x_stage5": x_feats["layer4"],
        }
        return out

    def _forward_single(self, net: nn.Module, inp: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Run the forward pass of a torchvision ResNet-like model but return intermediate layer outputs.
        Uses the standard torchvision ResNet implementation fields (conv1, bn1, relu, maxpool, layer1..layer4).
        """
        x = net.conv1(inp)
        x = net.bn1(x)
        x = net.relu(x)
        x = net.maxpool(x)

        x = net.layer1(x)
        layer1 = x
        x = net.layer2(x)
        layer2 = x
        x = net.layer3(x)
        layer3 = x
        x = net.layer4(x)
        layer4 = x

        return {"layer1": layer1, "layer2": layer2, "layer3": layer3, "layer4": layer4}

    # --------------------------
    # Utilities for weight tying and initialization
    # --------------------------
    def _adapt_first_conv_for_x(self, x_net: nn.Module, in_ch: int, rgb_net: nn.Module):
        """
        Replace x_net.conv1 to accept `in_ch` channels.
        Initialization strategy: if in_ch < 3 -> average RGB conv weights across channels to reduce to fewer channels.
        If in_ch > 3 -> replicate or tile RGB conv weights to initialize extra channels.
        We do NOT tie this first conv when in_ch != 3 (can't share shape).
        """
        old_conv = x_net.conv1
        new_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None),
        )
        # init new_conv.weight from rgb conv weights intelligently
        with torch.no_grad():
            w_rgb = rgb_net.conv1.weight.data  # shape (out_ch, 3, k, k)
            if in_ch == 1:
                # average RGB channels
                w_new = w_rgb.mean(dim=1, keepdim=True)  # (out_ch,1,k,k)
            elif in_ch < 3:
                # average to reduce
                w_new = w_rgb[:, :in_ch, :, :].mean(dim=1, keepdim=True)
            else:
                # replicate channels to match
                reps = int((in_ch + 2) // 3)  # how many copies to make
                w_rep = w_rgb.repeat(1, reps, 1, 1)[:, :in_ch, :, :]
                w_new = w_rep
            new_conv.weight.data.copy_(w_new)
            if old_conv.bias is not None:
                new_conv.bias.data.copy_(old_conv.bias.data)
        x_net.conv1 = new_conv

    def _tie_conv_weights(self, rgb_net: nn.Module, x_net: nn.Module, tie_first_conv: bool = True):
        """
        Tie (make identical Parameter references) all Conv2d weights between rgb_net and x_net.
        If tie_first_conv is False we skip tying conv1 (useful when x_in_ch != 3).
        """
        # iterate modules in parallel; this depends on identical module traversal order
        for m_rgb, m_x in _pairwise_modules(rgb_net, x_net):
            if _is_conv(m_rgb) and _is_conv(m_x):
                # check whether this is conv1; mapping:
                # if conv1 tied should be allowed only when shapes match
                if m_rgb is rgb_net.conv1 and not tie_first_conv:
                    # skip first conv
                    continue
                # replace m_x weight and bias with references to m_rgb's
                # delete existing nn.Parameter in m_x then assign reference to the rgb Parameter
                # safe way: set attribute to reference the same Parameter
                m_x.weight = m_rgb.weight
                if hasattr(m_rgb, "bias") and m_rgb.bias is not None:
                    m_x.bias = m_rgb.bias

    def _freeze_bn_running_stats(self, net: nn.Module):
        """
        Freeze running_mean/var of all BatchNorm layers (still keep affine learnable).
        """
        for m in net.modules():
            if _is_bn(m):
                m.track_running_stats = False

    # --------------------------
    # Parameter helpers
    # --------------------------
    def parameters(self, recurse: bool = True) -> Iterable[nn.Parameter]:
        """
        Override to return unique parameters (de-duplicated), because conv weights are shared and would otherwise appear twice.
        """
        seen = set()
        for name, p in self.named_parameters(recurse=recurse):
            pid = id(p)
            if pid not in seen:
                seen.add(pid)
                yield p

    def named_parameters(self, prefix: str = "", recurse: bool = True):
        """
        Yield named parameters (unique only once).
        Note: parameter names correspond to the wrapper module (rgb_net and x_net).
        If a Parameter is shared we only yield the first occurrence.
        """
        seen = set()
        # iterate over both submodules to preserve names like 'rgb_net.layer3.0.conv1.weight', etc.
        for module_name in ["rgb_net", "x_net"]:
            module = getattr(self, module_name)
            for name, p in module.named_parameters(prefix=module_name, recurse=recurse):
                pid = id(p)
                if pid in seen:
                    continue
                seen.add(pid)
                yield name, p

    def get_param_groups(self, base_lr: float = 1e-4, backbone_lr_mult: float = 0.25, weight_decay: float = 1e-5):
        """
        Convenient method returning optimizer parameter groups (deduplicated).
        Returns a list of dicts suitable for passing to optimizers (torch.optim.AdamW(..., params=...)).

        - Shared conv params get grouped under 'shared_conv' and receive backbone_lr_mult * base_lr
        - All other backbone params get backbone_lr_mult * base_lr
        - If you add new heads on top, you can pass their params separately to the optimizer
        """
        param_groups = [
            {"params": [], "lr": base_lr * backbone_lr_mult, "weight_decay": weight_decay},  # backbone
        ]
        # collect params from both nets, but ensure unique
        for name, p in self.named_parameters():
            # simple heuristic: keep everything in backbone group; if you'd like to separate conv vs bn, refine here
            param_groups[0]["params"].append(p)
        return param_groups

    # --------------------------
    # Convenience: move both nets to device
    # --------------------------
    def to(self, *args, **kwargs):
        # preserve the standard Module.to behavior
        return super().to(*args, **kwargs)


# --------------------------
# Example usage
# --------------------------
if __name__ == "__main__":
    # quick local shape test
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RGBXBackbone(x_in_ch=1, pretrained=False).to(device)

    B = 2
    H = 512
    W = 512
    rgb = torch.randn(B, 3, H, W, device=device)
    x = torch.randn(B, 1, H, W, device=device)  # e.g., depth/thermal

    feats = model(rgb, x)
    for k, v in feats.items():
        print(f"{k}: {v.shape}")

    # optimizer example
    import torch.optim as optim
    param_groups = model.get_param_groups(base_lr=1e-4, backbone_lr_mult=0.25)
    opt = optim.AdamW(param_groups)
    # run forward + backward to ensure params are usable
    seg_loss = feats["rgb_stage5"].abs().mean()
    seg_loss.backward()
    opt.step()
    print("Quick smoke test passed.")
