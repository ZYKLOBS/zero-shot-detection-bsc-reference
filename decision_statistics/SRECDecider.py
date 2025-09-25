from typing import List
from .base import BaseDecider
import numpy as np
import torch

class SRECDecider(BaseDecider):
    def __init__(self, pixel4_method: str = "direct"):
        """
        :param pixel4_method: How to handle the 4th pixel ("direct", "avg", "ignore")
        """
        super().__init__()
        self.name = f"SRECDecider (pixel4_method={pixel4_method})"
        self.pixel4_method = pixel4_method



    def get_D(self, nll: List, entropy: List) -> List[float]:
        """
        Compute D = NLL - H with handling of the 4th pixel depending on pixel4_method.
        Each entry in nll/entropy is a tensor corresponding to a pixel.
        Returns a list of 3 floats (one per resolution) after spatial averaging.
        """

        # Ensure correct lengths for SReC, see https://github.com/ZYKLOBS/SREC-nll-entropy
        assert len(nll) == len(entropy) == 12, "Expected 12 values per image (4 per pixel, 3 pixels per resolution)"
        assert nll[0].shape == entropy[0].shape, "Shape mismatch between nll and entropy entry"

        # Handle pixel4_method, see paper or https://github.com/ZYKLOBS/SREC-nll-entropy for details
        if self.pixel4_method == "direct":
            # Use values directly, assume they are correct
            indices_per_res = [(0, 4), (4, 8), (8, 12)]

        elif self.pixel4_method == "avg":
            # Replace 4th pixel in each resolution with mean of first 3




            nll[3] = sum(nll[0:3]) / 3
            nll[7] = sum(nll[4:7]) / 3
            nll[11] = sum(nll[8:11]) / 3

            # ------------------------------- FOR ODD DIMENSION ERROR USE THIS -------------------------------
            # This code block drops one element in case of odd and even dimensions (odd resolution leads to problem with
            # 2x2 grouping since that will yield 168, 167 for 335). I use this approach here with commenting so you notice the error!
            #You may think of better ways to handle this, like cropping the image beforehand etc... (Plus time constraints of thesis :)
            # min_h = min(t.shape[0] for t in nll[0:3])
            # min_w = min(t.shape[1] for t in nll[0:3])
            # nll_sliced = [t[:min_h, :min_w] for t in nll[0:3]]
            # nll[3] = sum(nll_sliced) / 3
            #
            # min_h = min(t.shape[0] for t in nll[4:7])
            # min_w = min(t.shape[1] for t in nll[4:7])
            # nll_sliced = [t[:min_h, :min_w] for t in nll[4:7]]
            # nll[7] = sum(nll_sliced) / 3
            #
            # min_h = min(t.shape[0] for t in nll[8:11])
            # min_w = min(t.shape[1] for t in nll[8:11])
            # nll_sliced = [t[:min_h, :min_w] for t in nll[8:11]]
            # nll[11] = sum(nll_sliced) / 3
            # ------------------------------- FOR ODD DIMENSION ERROR USE THIS -------------------------------

            indices_per_res = [(0, 4), (4, 8), (8, 12)]

        elif self.pixel4_method == "ignore":
            # Drop the 4th pixel in each resolution
            indices_per_res = [(0, 3), (4, 7), (8, 11)]

        else:
            raise ValueError(f"Unknown pixel4_method: {self.pixel4_method}")

        D_list = []
        for start, end in indices_per_res:
            # Element-wise subtraction
            D_block = [nll[i] - entropy[i] for i in range(start, end)]

            # ------------------------------- FOR ODD DIMENSION ERROR USE THIS -------------------------------
            # This code block drops one element in case of odd and even dimensions (odd resolution leads to problem with
            # 2x2 grouping since that will yield 168, 167 for 335). I use this approach here with commenting so you notice this case!
            # You may think of better ways to handle this, like cropping the image beforehand etc...

            # D_block = [nll[i] - entropy[i] for i in range(start, end)]
            #
            # min_h = min(t.shape[0] for t in D_block)
            # min_w = min(t.shape[1] for t in D_block)
            #
            # D_block_cropped = [t[:min_h, :min_w] for t in D_block]
            #
            # D_avg = torch.stack(D_block_cropped).mean().item()
            # D_list.append(D_avg)
            # ------------------------------- FOR ODD DIMENSION ERROR USE THIS -------------------------------

            # Spatial averaging
            D_avg = torch.stack(D_block).mean().item()  # .item() converts to float
            D_list.append(D_avg)

        return D_list

    def get_delta(self, D: List[float]) -> List[float]:
        """
        Compute difference between consecutive elements: delta_n = D_(n+1) - D_n
        """
        return [D[i + 1] - D[i] for i in range(len(D) - 1)]
