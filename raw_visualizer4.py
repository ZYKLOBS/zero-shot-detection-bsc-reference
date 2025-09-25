"""
Thesis Visualization of D = NLL - H
Comparison of one real and one fake image
------------------------------------------
- Uses same global scale (so darker values = lower D in both)
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import os






# -------- Load Data --------
real_path = "./data/pt/raise1k_pt/r19fe3107t.srec_stats_20250810_151010_035055.pt"
fake_path = "./data/pt/dalle2/r19fe3107t.srec_stats_20250815_152046_559186.pt"

data_real = torch.load(real_path)
data_fake = torch.load(fake_path)

nll_list_real = data_real["nll"]
entropy_list_real = data_real["entropy"]

nll_list_fake = data_fake["nll"]
entropy_list_fake = data_fake["entropy"]


def prepare_img_for_interleave(tensor):
    """Convert torch tensor (1,C,H,W) → numpy (H,W)."""
    return tensor.mean(dim=1).squeeze(0).cpu().numpy()

def pad_to_shape(img, target_shape):
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

    full = np.zeros((max_h * 2, max_w * 2), dtype=np.float32)
    full[0::2, 0::2], full[0::2, 1::2] = tl, tr
    full[1::2, 0::2], full[1::2, 1::2] = bl, br
    return full

def normalize_linear(img, min_val, max_val):
    img_clipped = np.clip(img, min_val, max_val)
    return (img_clipped - min_val) / (max_val - min_val + 1e-8)

def save_clean(img, path):
    """Save image with no border, no axis, no title — only the array itself."""
    plt.figure(figsize=(6, 6), dpi=200)
    plt.imshow(img, cmap="gray", interpolation="nearest")
    plt.axis("off")
    plt.margins(0, 0)
    plt.gca().set_position([0, 0, 1, 1])
    plt.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()


def compare_real_fake_independent(nll_real, h_real, nll_fake, h_fake, save_dir):
    """Compute and save clean D images for both real and fake."""
    for res in range(3):

        nll_tl_r = prepare_img_for_interleave(nll_real[res * 4 + 0])
        nll_tr_r = prepare_img_for_interleave(nll_real[res * 4 + 1])
        nll_bl_r = prepare_img_for_interleave(nll_real[res * 4 + 2])
        nll_br_r = np.mean([nll_tl_r, nll_tr_r, nll_bl_r], axis=0)

        nll_tl_f = prepare_img_for_interleave(nll_fake[res * 4 + 0])
        nll_tr_f = prepare_img_for_interleave(nll_fake[res * 4 + 1])
        nll_bl_f = prepare_img_for_interleave(nll_fake[res * 4 + 2])
        nll_br_f = np.mean([nll_tl_f, nll_tr_f, nll_bl_f], axis=0)


        h_tl_r = prepare_img_for_interleave(h_real[res * 4 + 0])
        h_tr_r = prepare_img_for_interleave(h_real[res * 4 + 1])
        h_bl_r = prepare_img_for_interleave(h_real[res * 4 + 2])
        h_br_r = np.mean([h_tl_r, h_tr_r, h_bl_r], axis=0)

        h_tl_f = prepare_img_for_interleave(h_fake[res * 4 + 0])
        h_tr_f = prepare_img_for_interleave(h_fake[res * 4 + 1])
        h_bl_f = prepare_img_for_interleave(h_fake[res * 4 + 2])
        h_br_f = np.mean([h_tl_f, h_tr_f, h_bl_f], axis=0)

        # Interleave full maps
        nll_full_r = interleave(nll_tl_r, nll_tr_r, nll_bl_r, nll_br_r)
        h_full_r   = interleave(h_tl_r, h_tr_r, h_bl_r, h_br_r)
        nll_full_f = interleave(nll_tl_f, nll_tr_f, nll_bl_f, nll_br_f)
        h_full_f   = interleave(h_tl_f, h_tr_f, h_bl_f, h_br_f)

        # Compute D
        D_real = nll_full_r - h_full_r
        D_fake = nll_full_f - h_full_f

        # Normalize each independently
        d_min_r, d_max_r = np.percentile(D_real, 1), np.percentile(D_real, 99)
        d_min_f, d_max_f = np.percentile(D_fake, 1), np.percentile(D_fake, 99)

        D_real_norm = normalize_linear(D_real, d_min_r, d_max_r)
        D_fake_norm = normalize_linear(D_fake, d_min_f, d_max_f)


        save_clean(D_real_norm, os.path.join(save_dir, f"D_real_res{res+1}.png"))
        save_clean(D_fake_norm, os.path.join(save_dir, f"D_fake_res{res+1}.png"))


save_dir = "./data/figures/thesis_clean"
os.makedirs(save_dir, exist_ok=True)

compare_real_fake_independent(
    nll_list_real, entropy_list_real,
    nll_list_fake, entropy_list_fake,
    save_dir
)
