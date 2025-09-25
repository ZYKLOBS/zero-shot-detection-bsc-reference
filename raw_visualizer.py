"""
Thesis Visualization of D = NLL - H
Real vs Fake image comparison
------------------------------------------
This version normalizes each image independently
(i.e. separate scales for real and fake).
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import os


real_path = "./data/pt/raise1k_pt/r0ea0825ft.srec_stats_20250810_142206_116979.pt"
fake_path = "./data/pt/mj5_pt/r0ea0825ft.srec_stats_20250810_193851_103223.pt"

real_data = torch.load(real_path)
fake_data = torch.load(fake_path)

real_nll, real_entropy = real_data["nll"], real_data["entropy"]
fake_nll, fake_entropy = fake_data["nll"], fake_data["entropy"]



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



def compute_D(nll_list, entropy_list, res):
    """Compute interleaved D = NLL - H map for a given resolution."""
    nll_tl = prepare_img_for_interleave(nll_list[res * 4 + 0])
    nll_tr = prepare_img_for_interleave(nll_list[res * 4 + 1])
    nll_bl = prepare_img_for_interleave(nll_list[res * 4 + 2])
    nll_br = np.mean([nll_tl, nll_tr, nll_bl], axis=0)

    h_tl = prepare_img_for_interleave(entropy_list[res * 4 + 0])
    h_tr = prepare_img_for_interleave(entropy_list[res * 4 + 1])
    h_bl = prepare_img_for_interleave(entropy_list[res * 4 + 2])
    h_br = np.mean([h_tl, h_tr, h_bl], axis=0)

    nll_full = interleave(nll_tl, nll_tr, nll_bl, nll_br)
    h_full   = interleave(h_tl, h_tr, h_bl, h_br)

    return nll_full - h_full



def compare_real_fake_independent(real_nll, real_entropy, fake_nll, fake_entropy, save_dir=None):
    for res in range(3):
        D_real = compute_D(real_nll, real_entropy, res)
        D_fake = compute_D(fake_nll, fake_entropy, res)

        # independent scaling
        real_min, real_max = np.percentile(D_real, 1), np.percentile(D_real, 99)
        fake_min, fake_max = np.percentile(D_fake, 1), np.percentile(D_fake, 99)

        D_real_norm = normalize_linear(D_real, real_min, real_max)
        D_fake_norm = normalize_linear(D_fake, fake_min, fake_max)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
        fig.suptitle(fr"$D^{{({2-res})}}$ Comparison (independent scales)", fontsize=16)

        axes[0].imshow(D_real_norm, cmap="gray", interpolation="nearest")
        axes[0].set_title("Real Image")
        axes[0].axis("off")

        axes[1].imshow(D_fake_norm, cmap="gray", interpolation="nearest")
        axes[1].set_title("Fake Image")
        axes[1].axis("off")

        if save_dir:
            fname = os.path.join(save_dir, f"D_compare_independent_res{res+1}.png")
            plt.savefig(fname, dpi=200, bbox_inches="tight")

        plt.show()


# ---Plot ---
save_dir = "./data/figures/thesis"
os.makedirs(save_dir, exist_ok=True)

compare_real_fake_independent(real_nll, real_entropy, fake_nll, fake_entropy, save_dir)
