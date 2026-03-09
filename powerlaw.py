from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

GEOMETRY_NAMES: dict[str, str] = {
    "t": "45 graus",
    "h": "hexagonal",
    "s": "quadrada",
}


def fit_size(
    data: npt.NDArray, alfa: float, beta: float, L: int
) -> tuple[npt.NDArray, npt.NDArray]:
    vlb = data["V"] / (L**beta)
    ilb = data["I"] / (L**alfa)
    return ilb, vlb


def fit_disorder(
    data: npt.NDArray, eta: float, nu: float, D: float
) -> tuple[npt.NDArray, npt.NDArray]:
    vlb = data["V"] * (1 + D) ** nu
    ilb = data["I"] * (1 + D) ** eta
    return ilb, vlb


def sort_by_size(path: Path) -> int:
    return int(path.stem.split(".")[0])


def sort_by_disorder(path: Path) -> float:
    return float(path.stem.split(".")[2].replace(",", "."))


def parse_args() -> argparse.Namespace:
    opt = argparse.ArgumentParser(
        description="Evaluate fracture power law.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    opt.add_argument("-s", "--source", type=Path, required=True)
    opt.add_argument(
        "-m", "--mode", choices=["size", "disorder"], required=True,
        help="Vary over lattice size or disorder",
    )
    opt.add_argument(
        "-g", "--geometry", choices=["t", "h", "s"],
        help="Geometry type (required for mode=disorder)",
    )
    opt.add_argument(
        "-l", "--length", type=int,
        help="Fixed length value (required for mode=disorder)",
    )
    return opt.parse_args()


def main() -> None:
    args = parse_args()
    source: Path = args.source

    if args.mode == "size":
        files = sorted(source.glob("*.csv"), key=sort_by_size)
        print("\n".join(str(f) for f in files))
        sizes = [7, 14, 20, 28]
        data = [
            np.genfromtxt(path, delimiter=",", skip_header=1, names=["V", "I"])
            for path in files
        ]

        while True:
            try:
                alfa = float(input("\nalfa: "))
                beta = float(input("beta: "))
            except ValueError:
                break
            fitresult = [
                (L, fit_size(d, alfa, beta, L)) for L, d in zip(sizes, data)
            ]

            for L, (ilb, vlb) in fitresult:
                plt.plot(vlb, ilb, label=f"L = {L}")

            plt.legend(loc="upper left")
            plt.grid(True)
            plt.axis("equal")
            plt.xlabel(r"$V/L^\beta$")
            plt.ylabel(r"$I/L^\alpha$")
            plt.title(f"$D = 0$, $\\alpha = {alfa}$ e $\\beta = {beta}$")
            plt.show()

    else:
        files = sorted(source.glob("*.csv"), key=sort_by_disorder)
        print("\n".join(str(f) for f in files))
        disorders = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        data = [
            np.genfromtxt(path, delimiter=",", skip_header=1, names=["V", "I"])
            for path in files
        ]

        while True:
            try:
                eta = float(input("\neta: "))
                nu = float(input("nu: "))
            except ValueError:
                break
            fitresult = [
                (D, fit_disorder(d, eta, nu, D))
                for D, d in zip(disorders, data)
            ]

            for D, (ilb, vlb) in fitresult:
                plt.plot(vlb, ilb, label=f"D = {D}")

            plt.legend(loc="upper left")
            plt.grid(True)
            plt.xlabel(r"$V(1+D)^{\nu}$")
            plt.ylabel(r"$I(1+D)^{\eta}$")

            geometry = GEOMETRY_NAMES[args.geometry]
            plt.title(
                f"$L = {args.length}$, G = {geometry}, $\\eta = {eta}$ e $\\nu = {nu}$"
            )
            plt.show()


if __name__ == "__main__":
    main()
