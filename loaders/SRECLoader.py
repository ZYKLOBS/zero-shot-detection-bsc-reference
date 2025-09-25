import torch
from pathlib import Path
from typing import List, Any
from .base import BaseLoader  # assuming your base class is in base.py

def prepare_img_for_interleave(tensor: torch.Tensor) -> Any:
    """
    Average over channels (dim=1) and remove batch dim, to get (H, W) numpy array.
    """
    return tensor.mean(dim=1).squeeze(0).cpu()

class SRECDataLoader(BaseLoader):
    def __init__(self):
        super().__init__()
        self.name = "SRECDataLoader"

    def get_nll(self, file: Path) -> List[Any]:
        """
        In this case: Array of size 12 with Res 0, pixel 0 at idx=0, Res 0, pixel 1 at idx=1, ..., Res 2, pixel 4 at idx=11
        Res 0 being lowest resolution
        """
        data = torch.load(file, map_location="cpu")
        nll_list: List[torch.Tensor] = data["nll"]
        return [prepare_img_for_interleave(t) for t in nll_list]

    def get_entropy(self, file: Path) -> List[Any]:
        """
        In this case: Array of size 12 with Res 0, pixel 0 at idx=0, Res 0, pixel 1 at idx=1, ..., Res 2, pixel 4 at idx=11
        Res 0 being lowest resolution
        """
        data = torch.load(file, map_location="cpu")
        entropy_list: List[torch.Tensor] = data["entropy"]
        return [prepare_img_for_interleave(t) for t in entropy_list]
