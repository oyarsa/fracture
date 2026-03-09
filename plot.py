"""
fracture: Simulation of a scalar fracture model based on Freitas (2007).
Copyright (C) 2017 Italo Silva

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    opt = argparse.ArgumentParser(
        description="Plot folder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    opt.add_argument("-f", "--folder", type=Path, required=True)
    opt.add_argument("--noshow", action="store_true")
    opt.add_argument("-g", "--geometry", choices=["t", "h", "s"], required=True)
    opt.add_argument("-l", "--length", type=int, required=True)
    return opt.parse_args()


def main() -> None:
    args = parse_args()
    folder: Path = args.folder

    plt.style.use("ggplot")

    print()
    plt.figure()
    plt.suptitle(f"L = {args.length}, G = {args.geometry}")

    for path in sorted(folder.glob("*.csv")):
        out = np.genfromtxt(path, delimiter=",", skip_header=1, names=["V", "I"])
        D = ".".join(path.stem.split(".")[2:])
        plt.plot(out["V"], out["I"], label=f"D = {D}")

    plt.legend(loc="upper left")
    plt.savefig(folder, bbox_inches="tight")

    if not args.noshow:
        plt.show()


if __name__ == "__main__":
    main()
