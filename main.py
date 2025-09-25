"""
Calculates decision metrics, in thesis this would be the decision metric module
"""

import argparse
from pathlib import Path
from typing import List, Type

from decision_statistics.SRECDecider import SRECDecider
from decision_statistics.base import BaseDecider
from loaders.SRECLoader import SRECDataLoader
from loaders.base import BaseLoader
from datetime import datetime

import numpy as np

# Loader registry
LOADER_CLASSES = {
    "srec": SRECDataLoader,
}

DECIDER_CLASSES = {
    "srec": SRECDecider,
}

def parse_args():
    parser = argparse.ArgumentParser(description="Computes if images are synthetic or real")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Path to dir that contains files that contain nll & entropy ")
    parser.add_argument("--file-type", type=str, default="*.pt",
                        help="File Type, i.e. *.pt, *.pth, *.npz, *.json")
    parser.add_argument(
        "--loader", choices=LOADER_CLASSES.keys(), required=True,
        help="Which loader to use for reading files"
    )
    parser.add_argument(
        "--decider", choices=DECIDER_CLASSES.keys(),
        help="Which decider to use (defaults to loader name)"
    )
    #Due to SREC for example treating the 4th pixel differently we have to add a flag here
    parser.add_argument(
        "--pixel4-method",
        choices=["direct", "avg", "ignore"],
        default="direct",
        help="Method to handle the 4th pixel"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # Pick loader
    loader_cls: BaseLoader = LOADER_CLASSES[args.loader]
    loader = loader_cls()

    # Pick decider (default to loader name if not given)
    decider_key = args.decider or args.loader
    decider_cls: BaseDecider = DECIDER_CLASSES[decider_key]
    decider = decider_cls(pixel4_method=args.pixel4_method)

    print(f"Loading '{args.file_type}' in '{args.data_dir}'\n"
          f"with loader '{loader.name}' and decider '{decider.name}'"
          f" (pixel4 method: '{args.pixel4_method}')")

    D_vals: List[List] = [] #List of D vals, N_images x D_values, i.e. [[d_vals for image 1], [d_vals for image 2], ...]
    delta_vals: List[List] = [] #List of deltas, same as above

    for file in args.data_dir.glob(args.file_type):
        nll = loader.get_nll(file)
        entropy = loader.get_entropy(file)

        D = decider.get_D(nll, entropy)
        D_vals.append(D)

        delta = decider.get_delta(D)
        delta_vals.append(delta)



    # Ensure results folder exists
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Create timestamped filename, rename appropriately or fix to better name if time left after thesis
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = results_dir / f"results_{timestamp}.npz"

    D_vals_arr = np.array(D_vals)
    delta_vals_arr = np.array(delta_vals)

    # Save
    np.savez(filename, D_vals=D_vals_arr, delta_vals=delta_vals_arr)

    print(f"Finished '{args.file_type}' in '{args.data_dir}'\n"
          f"with loader '{loader.name}' and decider '{decider.name}'"
          f" (pixel4 method: '{args.pixel4_method}')")

if __name__ == "__main__":
    main()