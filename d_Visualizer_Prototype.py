"""
NLL - Entropy
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter  # For smoothing

filepath = "./pt_data/original.srec_stats_20250803_122630_571372.pt"
data = torch.load(filepath)

nll_list = data['nll']        # List of 12 tensors for NLL
entropy_list = data['entropy'] # List of 12 tensors for Entropy

def prepare_img_for_interleave(tensor):
    # Average over channels (dim=1) and remove batch dim, to get (H, W)
    return tensor.mean(dim=1).squeeze(0).cpu().numpy()

def pad_to_shape(img, target_shape):
    h, w = img.shape
    th, tw = target_shape
    pad_h = th - h
    pad_w = tw - w
    if pad_h < 0 or pad_w < 0:
        raise ValueError(f"Target shape {target_shape} smaller than image shape {img.shape}")
    return np.pad(img, ((0, pad_h), (0, pad_w)), mode='reflect')

def interleave(tl, tr, bl, br):
    shapes = [tl.shape, tr.shape, bl.shape, br.shape]
    max_h = max(s[0] for s in shapes)
    max_w = max(s[1] for s in shapes)

    tl_pad = pad_to_shape(tl, (max_h, max_w))
    tr_pad = pad_to_shape(tr, (max_h, max_w))
    bl_pad = pad_to_shape(bl, (max_h, max_w))
    br_pad = pad_to_shape(br, (max_h, max_w))

    full = np.zeros((max_h * 2, max_w * 2), dtype=np.float32)
    full[0::2, 0::2] = tl_pad
    full[0::2, 1::2] = tr_pad
    full[1::2, 0::2] = bl_pad
    full[1::2, 1::2] = br_pad
    return full

def normalize_img(img):
    min_val = np.percentile(img, 1)
    max_val = np.percentile(img, 99)
    img_clipped = np.clip(img, min_val, max_val)
    return (img_clipped - min_val) / (max_val - min_val + 1e-8)

for res in range(3):
    fig, axes = plt.subplots(2, 2, figsize=(8, 8), constrained_layout=True)
    fig.suptitle(f"NLL - Entropy Maps — Resolution {res+1} (Subpixel Groups)", fontsize=16)

    for i in range(4):
        idx = res * 4 + i
        nll_img = prepare_img_for_interleave(nll_list[idx])
        ent_img = prepare_img_for_interleave(entropy_list[idx])
        diff_img = nll_img - ent_img

        # Optional: shift to make all positive (uncomment if needed)
        # diff_img = diff_img - diff_img.min()

        ax = axes[i // 2, i % 2]
        ax.imshow(normalize_img(diff_img), cmap='gray', interpolation='nearest', aspect='equal')
        ax.set_title(f"Pixel {i}")
        ax.axis('off')

    plt.show()


for res in range(3):
    tl_diff = prepare_img_for_interleave(nll_list[res * 4 + 0]) - prepare_img_for_interleave(entropy_list[res * 4 + 0])
    tr_diff = prepare_img_for_interleave(nll_list[res * 4 + 1]) - prepare_img_for_interleave(entropy_list[res * 4 + 1])
    bl_diff = prepare_img_for_interleave(nll_list[res * 4 + 2]) - prepare_img_for_interleave(entropy_list[res * 4 + 2])
    br_diff = prepare_img_for_interleave(nll_list[res * 4 + 3]) - prepare_img_for_interleave(entropy_list[res * 4 + 3])

    full_diff = interleave(tl_diff, tr_diff, bl_diff, br_diff)
    smoothed_diff = gaussian_filter(full_diff, sigma=1)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(normalize_img(full_diff), cmap='gray', interpolation='nearest', aspect='equal')
    plt.title(f"Interleaved NLL - Entropy Map — Resolution {res+1} (Raw)")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(normalize_img(smoothed_diff), cmap='gray', interpolation='bilinear', aspect='equal')
    plt.title(f"Interleaved NLL - Entropy Map — Resolution {res+1} (Smoothed)")
    plt.axis('off')

    plt.show()
