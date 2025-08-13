"""
NLL or Entropy Visualization
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter  # For smoothing


filepath = "./pt_data/original.srec_stats_20250803_122630_571372.pt"
data = torch.load(filepath)
nll_list = data['nll']  # List of 12 tensors, call nll and entropy :)
print(nll_list[2])
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
    # Pad bottom and right edges with reflect mode for smoother transition ? ok -> chatgpt
    return np.pad(img, ((0, pad_h), (0, pad_w)), mode='reflect')

def interleave(tl, tr, bl, br):
    # Find max height and width to pad smaller images
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
    # Normalize to [0,1] using 1st and 99th percentile for robust contrast scaling
    min_val = np.percentile(img, 1)
    max_val = np.percentile(img, 99)
    img_clipped = np.clip(img, min_val, max_val)
    return (img_clipped - min_val) / (max_val - min_val + 1e-8)

# Visualize subpixel maps separately per resolution (the pixel groups)
for res in range(3):
    fig, axes = plt.subplots(2, 2, figsize=(6, 6), constrained_layout=True)
    fig.suptitle(f"NLL Map — Resolution {res+1} (Subpixel Groups)", fontsize=14)
    for i in range(4):
        idx = res * 4 + i
        img = prepare_img_for_interleave(nll_list[idx])
        # if idx in [3, 7, 11]:
        # Pixel 4 looks flipped therefore we invert it optionally?
        #     img = -img  # Invert the raw values (only useful if pixel4 is deterministically lower/higher)
        ax = axes[i // 2, i % 2]
        ax.imshow(normalize_img(img), cmap='gray', interpolation='nearest', aspect='equal')
        ax.set_title(f"Pixel {i}")
        ax.axis('off')
    plt.show()

# Visualize interleaved full images per resolution with smoothing side-by-side with raw
for res in range(3):
    tl = prepare_img_for_interleave(nll_list[res * 4 + 0])
    tr = prepare_img_for_interleave(nll_list[res * 4 + 1])
    bl = prepare_img_for_interleave(nll_list[res * 4 + 2])
    br = prepare_img_for_interleave(nll_list[res * 4 + 3])
    full_img = interleave(tl, tr, bl, br)

    # Smooth the interleaved image to reduce checkerboard effect
    smoothed_img = gaussian_filter(full_img, sigma=1)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(normalize_img(full_img), cmap='gray', interpolation='nearest', aspect='equal')
    plt.title(f"Interleaved NLL Map — Resolution {res+1} (Raw)")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(normalize_img(smoothed_img), cmap='gray', interpolation='bilinear', aspect='equal')
    plt.title(f"Interleaved NLL Map — Resolution {res+1} (Smoothed)")
    plt.axis('off')

    plt.show()



#OLD
# import torch
# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.ndimage import gaussian_filter  # For smoothing
#
# # Load your .pt file
# filepath = "./pt_data/0001eeaf4aed83f9.srec_stats_20250802_215656_513946.pt"
# data = torch.load(filepath)
# nll_list = data['nll']  # List of 12 tensors
#
# def prepare_img_for_interleave(tensor):
#     # Average over channels (dim=1) and remove batch dim, to get (H, W)
#     return tensor.mean(dim=1).squeeze(0).cpu().numpy()
#
# def pad_to_shape(img, target_shape):
#     h, w = img.shape
#     th, tw = target_shape
#     pad_h = th - h
#     pad_w = tw - w
#     if pad_h < 0 or pad_w < 0:
#         raise ValueError(f"Target shape {target_shape} smaller than image shape {img.shape}")
#     # Pad bottom and right edges by repeating the edge pixels
#     return np.pad(img, ((0, pad_h), (0, pad_w)), mode='edge')
#
# def interleave(tl, tr, bl, br):
#     # Find max height and width to pad smaller images
#     shapes = [tl.shape, tr.shape, bl.shape, br.shape]
#     max_h = max(s[0] for s in shapes)
#     max_w = max(s[1] for s in shapes)
#
#     tl_pad = pad_to_shape(tl, (max_h, max_w))
#     tr_pad = pad_to_shape(tr, (max_h, max_w))
#     bl_pad = pad_to_shape(bl, (max_h, max_w))
#     br_pad = pad_to_shape(br, (max_h, max_w))
#
#     full = np.zeros((max_h * 2, max_w * 2), dtype=np.float32)
#     full[0::2, 0::2] = tl_pad
#     full[0::2, 1::2] = tr_pad
#     full[1::2, 0::2] = bl_pad
#     full[1::2, 1::2] = br_pad
#     return full
#
# # Visualize subpixel maps separately per resolution (the pixel groups)
# for res in range(3):
#     fig, axes = plt.subplots(2, 2, figsize=(6, 6))
#     fig.suptitle(f"NLL Map — Resolution {res+1} (Subpixel Groups)", fontsize=14)
#     for i in range(4):
#         idx = res * 4 + i
#         img = prepare_img_for_interleave(nll_list[idx])
#         ax = axes[i // 2, i % 2]
#         ax.imshow(img, cmap='gray', interpolation='nearest')
#         ax.set_title(f"Pixel {i}")
#         ax.axis('off')
#     plt.tight_layout()
#     plt.subplots_adjust(top=0.88)
#     plt.show()
#
# # Visualize interleaved full images per resolution with smoothing side-by-side with raw
# for res in range(3):
#     tl = prepare_img_for_interleave(nll_list[res * 4 + 0])
#     tr = prepare_img_for_interleave(nll_list[res * 4 + 1])
#     bl = prepare_img_for_interleave(nll_list[res * 4 + 2])
#     br = prepare_img_for_interleave(nll_list[res * 4 + 3])
#     full_img = interleave(tl, tr, bl, br)
#
#     # Smooth the interleaved image to reduce checkerboard effect
#     smoothed_img = gaussian_filter(full_img, sigma=1)
#
#     plt.figure(figsize=(12, 6))
#     plt.subplot(1, 2, 1)
#     plt.imshow(full_img, cmap='gray', interpolation='nearest')
#     plt.title(f"Interleaved NLL Map — Resolution {res+1} (Raw)")
#     plt.axis('off')
#
#     plt.subplot(1, 2, 2)
#     plt.imshow(smoothed_img, cmap='gray', interpolation='nearest')
#     plt.title(f"Interleaved NLL Map — Resolution {res+1} (Smoothed)")
#     plt.axis('off')
#
#     plt.show()
