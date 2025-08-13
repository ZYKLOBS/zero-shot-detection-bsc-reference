import argparse
from pathlib import Path

from decision_statistics.SRECDecider import SRECDecider
from decision_statistics.base import BaseDecider
from loaders.SRECLoader import SRECDataLoader
from loaders.base import BaseLoader

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
    decider = decider_cls()

    print(f"Loading '{args.file_type}' in '{args.data_dir}'\n"
          f"with loader '{loader.name}' and decider '{decider.name}'"
          f" (pixel4 method: '{args.pixel4_method}')")

    for file in args.data_dir.glob(args.file_type):
        nll = loader.get_nll(file)
        entropy = loader.get_entropy(file)

        D = decider.get_D(nll, entropy)
        delta = decider.get_delta(D)
        print(f"File: {file}, D: {D}, delta: {delta}")
        break


if __name__ == "__main__":
    main()