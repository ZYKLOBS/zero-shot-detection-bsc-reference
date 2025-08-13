from typing import List
from .base import BaseDecider  # adjust import to your file structure
import numpy as np

#LOOK INTO THIS CODE SOME MORE, CONSIDER SPACIAL AVERAGING FOR D!
class SRECDecider(BaseDecider):
    def __init__(self, pixel4_method: str = "direct"):
        """
        :param pixel4_method: How to handle the 4th pixel ("direct", "avg", "ignore")
        """
        super().__init__()
        self.name = f"SRECDecider (pixel4_method={pixel4_method})"
        self.pixel4_method = pixel4_method

    def get_D(self, nll: List, entropy: List) -> List:
        """
        Compute D = NLL - H with handling of the 4th pixel depending on pixel4_method.
        """
        nll = np.array(nll, dtype=np.float32)
        entropy = np.array(entropy, dtype=np.float32)

        # Ensure shape compatibility
        assert nll.shape == entropy.shape, "NLL and entropy arrays must have the same shape"

        if self.pixel4_method == "direct":
            # Use values as-is
            D = nll - entropy

        elif self.pixel4_method == "avg":
            # Average over each 2×2 block including 4th pixel
            D = nll - entropy
            D = self._avg_2x2_blocks(D)

        elif self.pixel4_method == "ignore":
            # Mask out 4th pixels (assume they are last in every 4-pixel group)
            mask = np.ones_like(nll, dtype=bool)
            mask[3::4] = False  # every 4th pixel ignored
            D = (nll - entropy) * mask

        else:
            raise ValueError(f"Unknown pixel4_method: {self.pixel4_method}")

        return D.tolist()

    def get_delta(self, D: List) -> List:
        """
        Compute difference between consecutive elements: ΔD_n = D_(n+1) - D_n.
        """
        D = np.array(D, dtype=np.float32)
        delta = np.diff(D, axis=0)
        return delta.tolist()

    @staticmethod
    def _avg_2x2_blocks(arr: np.ndarray) -> np.ndarray:
        """
        Average over non-overlapping 2×2 blocks.
        This assumes arr is 1D in pixel order; adjust if you have (H, W) shape.
        """
        if arr.ndim == 1:
            arr = arr.reshape(-1, 4)  # group into 4 pixels
            arr = arr.mean(axis=1)    # average per block
            return arr
        else:
            raise ValueError("avg_2x2_blocks currently supports only flat pixel arrays")
