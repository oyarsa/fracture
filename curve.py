from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

GEOMETRY_NAMES: dict[str, str] = {
    "t": "45 graus",
    "h": "hexagonal",
    "s": "quadrada",
}


def sort_by_size(path: Path) -> int:
    return int(path.stem.split(".")[0])


def sort_by_disorder(path: Path) -> float:
    return float(path.stem.split(".")[2].replace(",", "."))


def parse_args() -> argparse.Namespace:
    opt = argparse.ArgumentParser(
        description="Plot fracture V-I curves.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    opt.add_argument("-s", "--source", type=Path, required=True)
    opt.add_argument(
        "-m",
        "--mode",
        choices=["size", "disorder"],
        required=True,
        help="Vary over lattice size or disorder",
    )
    opt.add_argument("-g", "--geometry", choices=["t", "h", "s"])
    opt.add_argument(
        "-d",
        "--disorder",
        type=float,
        help="Fixed disorder value (required for mode=size)",
    )
    opt.add_argument(
        "-l",
        "--length",
        type=int,
        help="Fixed length value (required for mode=disorder)",
    )
    return opt.parse_args()


def main() -> None:
    args = parse_args()
    source: Path = args.source

    if args.mode == "size":
        files = sorted(source.glob("*.csv"), key=sort_by_size)
        sizes = [7, 14, 20, 28]
        data = [
            np.genfromtxt(path, delimiter=",", skip_header=1, names=["V", "I"])
            for path in files
        ]

        for L, d in zip(sizes, data):
            plt.plot(d["V"], d["I"], label=f"L = {L}")

        geometry = GEOMETRY_NAMES[args.geometry]
        plt.title(f"$D = {args.disorder}, G = ${geometry}")

    else:
        files = sorted(source.glob("*.csv"), key=sort_by_disorder)
        disorders = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        data = [
            np.genfromtxt(path, delimiter=",", skip_header=1, names=["V", "I"])
            for path in files
        ]

        for D, d in zip(disorders, data):
            plt.plot(d["V"], d["I"], label=f"D = {D}")

        geometry = GEOMETRY_NAMES[args.geometry]
        plt.title(f"$L = {args.length}$, G = {geometry}")

    plt.legend(loc="upper left")
    plt.grid(True)
    plt.axis("equal")
    plt.xlabel(r"$V$")
    plt.ylabel(r"$I$")
    plt.show()


if __name__ == "__main__":
    main()
