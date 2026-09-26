"""
Real AI Super-Resolution / Restoration Architectures.

Implements the RRDBNet and SRVGGNetCompact network architectures exactly as
published by the Real-ESRGAN authors (xinntao/Real-ESRGAN, BSD-3-Clause),
so that the official pretrained checkpoints load with strict=True. These are
proven, widely deployed generative models that perform real super-resolution,
denoising, and detail reconstruction (unlike a randomly initialized network).

Three registered models cover the practical range needed for video:
  - general_v3: SRVGGNetCompact (realesr-general-x4v3) - fast, all-purpose,
    with a genuine tunable denoise strength (weight interpolation with the
    "wdn" checkpoint, the same technique used by Real-ESRGAN's own CLI).
  - x4plus:     RRDBNet 23-block - higher quality / heavier, for fine detail.
  - anime6b:    RRDBNet 6-block anime-optimized checkpoint.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def pixel_unshuffle(x: torch.Tensor, scale: int) -> torch.Tensor:
    b, c, hh, hw = x.size()
    h, w = hh // scale, hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, c * scale * scale, h, w)


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block (5 convs) used inside every RRDB, per ESRGAN."""

    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block."""

    def __init__(self, num_feat: int, num_grow_ch: int = 32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """
    Real-ESRGAN generator network (RRDBNet), matching the official
    xinntao/Real-ESRGAN `rrdbnet_arch.py` layer names and shapes exactly so
    pretrained checkpoints load without any key remapping.
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
        super().__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x

        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


class SRVGGNetCompact(nn.Module):
    """
    Compact VGG-style super-resolution network used by realesr-general-x4v3.
    Much lighter than RRDBNet, tuned for fast general-purpose video
    restoration while remaining a genuine trained generative model.
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type="prelu"):
        super().__init__()
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.upscale = upscale

        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        self.body.append(self._activation(act_type, num_feat))

        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(self._activation(act_type, num_feat))

        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    @staticmethod
    def _activation(act_type: str, num_feat: int) -> nn.Module:
        if act_type == "relu":
            return nn.ReLU(inplace=True)
        if act_type == "leakyrelu":
            return nn.LeakyReLU(negative_slope=0.1, inplace=True)
        return nn.PReLU(num_parameters=num_feat)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        base = F.interpolate(x, scale_factor=self.upscale, mode="nearest")
        return out + base


# ---------------------------------------------------------------------------
# Model registry: maps a user-facing model name to its architecture, native
# upscale factor, and the checkpoint file(s) it needs (auto-downloaded).
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    "general_v3": {
        "label": "Universal Video Restoration (Real-ESRGAN general v3)",
        "arch": "srvgg",
        "native_scale": 4,
        "weights": "realesr-general-x4v3.pth",
        "weights_denoise": "realesr-general-wdn-x4v3.pth",
        "supports_native_denoise": True,
        "arch_kwargs": dict(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type="prelu"),
    },
    "x4plus": {
        "label": "Fine Details & Texture SR (RealESRGAN x4plus)",
        "arch": "rrdb",
        "native_scale": 4,
        "weights": "RealESRGAN_x4plus.pth",
        "supports_native_denoise": False,
        "arch_kwargs": dict(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
    },
    "x2plus": {
        "label": "Native 2x High Fidelity (RealESRGAN x2plus)",
        "arch": "rrdb",
        "native_scale": 2,
        "weights": "RealESRGAN_x2plus.pth",
        "supports_native_denoise": False,
        "arch_kwargs": dict(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2),
    },
    "anime6b": {
        "label": "Clean Animation & CG (RealESRGAN anime 6B)",
        "arch": "rrdb",
        "native_scale": 4,
        "weights": "RealESRGAN_x4plus_anime_6B.pth",
        "supports_native_denoise": False,
        "arch_kwargs": dict(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4),
    },
    "bsrgan": {
        "label": "Realistic Photo/Video Restoration (BSRGAN)",
        "arch": "rrdb",
        "native_scale": 4,
        "weights": "BSRGAN.pth",
        "supports_native_denoise": False,
        "arch_kwargs": dict(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
    },
    "ultrasharp": {
        "label": "Community Ultra Sharp (4x-UltraSharp — solo uso no comercial)",
        "arch": "rrdb",
        "native_scale": 4,
        "weights": "4x-UltraSharp.pth",
        "supports_native_denoise": False,
        "arch_kwargs": dict(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
    },
}


def build_torch_arch(model_name: str) -> nn.Module:
    cfg = MODEL_REGISTRY[model_name]
    if cfg["arch"] == "rrdb":
        return RRDBNet(**cfg["arch_kwargs"])
    if cfg["arch"] == "srvgg":
        return SRVGGNetCompact(**cfg["arch_kwargs"])
    raise ValueError(f"Arquitectura desconocida para el modelo: {model_name}")


def extract_state_dict(checkpoint: dict) -> dict:
    """Real-ESRGAN checkpoints wrap the weights under 'params_ema' or 'params'."""
    if "params_ema" in checkpoint:
        return checkpoint["params_ema"]
    if "params" in checkpoint:
        return checkpoint["params"]
    return checkpoint


def remap_sequential_old_esrgan_state_dict(sd: dict) -> dict:
    """
    A second, even older ESRGAN serialization (seen in several community
    fine-tunes, e.g. 4x-UltraSharp) saves the whole network as one flat
    nn.Sequential named "model", so keys look like "model.0.weight" or
    "model.1.sub.0.RDB1.conv1.0.weight" instead of named submodules. This is
    the same RRDBNet architecture, just serialized differently.

    The layout (verified against actual checkpoint tensor shapes, not
    assumed from memory) is: model.0=conv_first, model.1.sub.{i}.RDBn.convM
    ={i-th RRDB's dense-block conv M}, model.1.sub.{last}=conv_body
    (trunk_conv), model.3=conv_up1, model.6=conv_up2, model.8=conv_hr,
    model.10=conv_last. The trunk_conv index is detected dynamically (the
    one "sub" entry with no "RDB" in its key) so this works regardless of
    how many RRDB blocks the checkpoint has.
    """
    if not any(k.startswith("model.0.") for k in sd.keys()):
        return sd

    sub_prefix = "model.1.sub."
    indices_without_rdb = set()
    for k in sd.keys():
        if k.startswith(sub_prefix):
            rest = k[len(sub_prefix):]
            if "RDB" not in rest:
                indices_without_rdb.add(int(rest.split(".")[0]))
    trunk_conv_idx = next(iter(indices_without_rdb), None)

    outer_map = {
        "model.0.": "conv_first.",
        "model.3.": "conv_up1.",
        "model.6.": "conv_up2.",
        "model.8.": "conv_hr.",
        "model.10.": "conv_last.",
    }

    remapped = {}
    for k, v in sd.items():
        matched_prefix = next((p for p in outer_map if k.startswith(p)), None)
        if matched_prefix:
            remapped[k.replace(matched_prefix, outer_map[matched_prefix])] = v
            continue

        if k.startswith(sub_prefix):
            parts = k[len(sub_prefix):].split(".")
            idx = int(parts[0])
            if idx == trunk_conv_idx:
                new_k = "conv_body." + ".".join(parts[1:])
            else:
                rdb, conv, wb = parts[1].lower(), parts[2], parts[-1]
                new_k = f"body.{idx}.{rdb}.{conv}.{wb}"
            remapped[new_k] = v
        else:
            remapped[k] = v

    return remapped


def remap_old_esrgan_state_dict(sd: dict) -> dict:
    """
    Many community/academic RRDBNet checkpoints (e.g. BSRGAN, the original
    ESRGAN, and several "old-arch" fine-tunes distributed as plain ESRGAN
    .pth files) use the original ESRGAN layer names instead of the renamed
    ones BasicSR/Real-ESRGAN's rrdbnet_arch.py uses. Both are the exact same
    architecture; only the key names differ. This remaps old -> new so any
    such checkpoint can be loaded with our single RRDBNet implementation.
    """
    if not any(k.startswith("RRDB_trunk.") for k in sd.keys()):
        return sd  # already in the new (Real-ESRGAN) naming convention

    remapped = {}
    for k, v in sd.items():
        new_k = k
        new_k = new_k.replace("RRDB_trunk.", "body.")
        new_k = new_k.replace(".RDB1.", ".rdb1.")
        new_k = new_k.replace(".RDB2.", ".rdb2.")
        new_k = new_k.replace(".RDB3.", ".rdb3.")
        new_k = new_k.replace("trunk_conv.", "conv_body.")
        new_k = new_k.replace("upconv1.", "conv_up1.")
        new_k = new_k.replace("upconv2.", "conv_up2.")
        new_k = new_k.replace("HRconv.", "conv_hr.")
        remapped[new_k] = v
    return remapped
