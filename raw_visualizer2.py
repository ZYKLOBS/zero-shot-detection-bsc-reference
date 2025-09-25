"""
Thesis Visualization of NLL & Entropy Maps (Direct 4th Pixel)
-------------------------------------------------------------
- NLL: linear normalization
- Entropy: gamma correction to stretch small values
- Last-resolution entropy figure uses actual 4th pixel (H^(3))
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
import os

filepath = "./data/pt/misc/gatze2.pt"
data = torch.load(filepath)

nll_list = data["nll"]         # List of 12 tensors for NLL
entropy_list = data["entropy"] # List of 12 tensors for Entropy
print(data["nll"][3])
print(data["nll"][0])

def prepare_img_for_interleave(tensor):
    """Convert torch tensor (1,C,H,W) → numpy (H,W)."""
    return tensor.mean(dim=1).squeeze(0).cpu().numpy()

def pad_to_shape(img, target_shape):
    """Pad an image to target shape with reflect mode."""
    h, w = img.shape
    th, tw = target_shape
    pad_h, pad_w = th - h, tw - w
    return np.pad(img, ((0, pad_h), (0, pad_w)), mode="reflect")

def interleave(tl, tr, bl, br):
    """Interleave 4 subpixel maps into one full map."""
    shapes = [tl.shape, tr.shape, bl.shape, br.shape]
    max_h, max_w = max(s[0] for s in shapes), max(s[1] for s in shapes)
    tl, tr = pad_to_shape(tl, (max_h, max_w)), pad_to_shape(tr, (max_h, max_w))
    bl, br = pad_to_shape(bl, (max_h, max_w)), pad_to_shape(br, (max_h, max_w))
    full = np.zeros((max_h*2, max_w*2), dtype=np.float32)
    full[0::2, 0::2], full[0::2, 1::2] = tl, tr
    full[1::2, 0::2], full[1::2, 1::2] = bl, br
    return full

def compute_global_minmax(data_list):
    """Compute global 1st and 99th percentile for normalization."""
    vals = np.concatenate([prepare_img_for_interleave(t).ravel() for t in data_list])
    return np.percentile(vals, 1), np.percentile(vals, 99)

def normalize_linear(img, min_val, max_val):
    """Linear normalization (for NLL)."""
    img_clipped = np.clip(img, min_val, max_val)
    return (img_clipped - min_val) / (max_val - min_val + 1e-8)

def normalize_gamma(img, min_val, max_val, gamma=0.5):
    """Gamma correction normalization (for Entropy)."""
    img_clipped = np.clip(img, min_val, max_val)
    img_norm = (img_clipped - min_val) / (max_val - min_val + 1e-8)
    return np.power(img_norm, gamma)


def plot_per_pixel_maps(data_list, label, vmin, vmax, normalize_fn, save_dir=None):
    """Show all 4 pixels for each resolution (2x2 grid)."""
    for res in range(3):
        fig, axes = plt.subplots(2, 2, figsize=(8, 8), constrained_layout=True)
        if label == "Entropy":
            label= "H"
        fig.suptitle(fr"${label}^{{{2-res}}}$ per pixel", fontsize=16)

        for i in range(4):
            row, col = divmod(i, 2)  # map 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
            idx = res * 4 + i
            img = prepare_img_for_interleave(data_list[idx])
            axes[row, col].imshow(normalize_fn(img, vmin, vmax), cmap="gray", interpolation="nearest")
            axes[row, col].set_title(f"Pixel {i}")
            axes[row, col].axis("off")

        if save_dir:
            fname = os.path.join(save_dir, f"{label}_perpixel_res{res + 1}.png")
            plt.savefig(fname, dpi=200, bbox_inches="tight")
        plt.show()


def plot_interleaved_maps(data_list, label, vmin, vmax, normalize_fn, save_dir=None):
    """Show interleaved raw + smoothed maps per resolution using actual 4th pixel."""
    for res in range(3):
        tl = prepare_img_for_interleave(data_list[res*4 + 0])
        tr = prepare_img_for_interleave(data_list[res*4 + 1])
        bl = prepare_img_for_interleave(data_list[res*4 + 2])
        br = prepare_img_for_interleave(data_list[res*4 + 3])  # Use actual 4th pixel
        full_map = interleave(tl, tr, bl, br)
        smoothed_map = gaussian_filter(full_map, sigma=1)
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
        if label == "Entropy":
            label= "H"
        fig.suptitle(fr"${label}^{{{2-res}}}$ interleaved", fontsize=16)
        axes[0].imshow(normalize_fn(full_map, vmin, vmax), cmap="gray", interpolation="nearest")
        axes[0].set_title("Raw")
        axes[0].axis("off")
        axes[1].imshow(normalize_fn(smoothed_map, vmin, vmax), cmap="gray", interpolation="bilinear")
        axes[1].set_title("Smoothed")
        axes[1].axis("off")
        if save_dir:
            fname = os.path.join(save_dir, f"{label}_interleaved_res{res+1}.png")
            plt.savefig(fname, dpi=200, bbox_inches="tight")
        plt.show()

def plot_last_entropy_full(entropy_list, vmin, vmax, normalize_fn, save_dir=None):
    """Plot only the last resolution of entropy using actual 4th pixel (raw)."""
    res = 2  # Last resolution
    tl = prepare_img_for_interleave(entropy_list[res*4 + 0])
    tr = prepare_img_for_interleave(entropy_list[res*4 + 1])
    bl = prepare_img_for_interleave(entropy_list[res*4 + 2])
    br = prepare_img_for_interleave(entropy_list[res*4 + 3])  # Actual 4th pixel
    full_map = interleave(tl, tr, bl, br)
    plt.figure(figsize=(8, 8), constrained_layout=True)
    plt.imshow(normalize_fn(full_map, vmin, vmax), cmap="gray", interpolation="nearest")
    plt.title(r"$H^{(3)}$ — Last Resolution (Full Raw)", fontsize=16)
    plt.axis("off")
    if save_dir:
        fname = os.path.join(save_dir, "Entropy_last_res_full_raw.png")
        plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.show()

# --- Plot ---
save_dir = "./data/figures/thesis"
os.makedirs(save_dir, exist_ok=True)

# Global normalization
nll_min, nll_max = compute_global_minmax(nll_list)
ent_min, ent_max = compute_global_minmax(entropy_list)

print(f"NLL range: {nll_min:.4f} – {nll_max:.4f}")
print(f"Entropy range: {ent_min:.4f} – {ent_max:.4f}")

# NLL
plot_per_pixel_maps(nll_list, "NLL", nll_min, nll_max, normalize_linear, save_dir)
plot_interleaved_maps(nll_list, "NLL", nll_min, nll_max, normalize_linear, save_dir)

# Entropy
plot_per_pixel_maps(entropy_list, "Entropy", ent_min, ent_max,
                    lambda img, mn, mx: normalize_gamma(img, mn, mx, gamma=0.25), save_dir)
plot_interleaved_maps(entropy_list, "Entropy", ent_min, ent_max,
                      lambda img, mn, mx: normalize_gamma(img, mn, mx, gamma=0.25), save_dir)

# Last-resolution entropy figure (4th pixel)
#plot_last_entropy_full(entropy_list, ent_min, ent_max,
#                       lambda img, mn, mx: normalize_gamma(img, mn, mx, gamma=0.25),
#                       save_dir)
