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
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

EXE = Path("out/fracture")
DATA_DIR = Path("data")
PDF_DIR = Path("pdf")
GRAPH_DIR = Path("graphs")
RESULTS_FILE = Path("output.csv")
GRAPH_BEGIN = Path("graph-begin.dot")
GRAPH_END = Path("graph-end.dot")
OPENER = "open" if sys.platform == "darwin" else "xdg-open"


def open_file(path: Path) -> None:
    subprocess.run([OPENER, str(path)])


def render_graph(dot_file: Path, pdf_path: Path) -> None:
    subprocess.run(["dot", str(dot_file), "-Tpdf", f"-o{pdf_path}"])
    dot_file.unlink()
    open_file(pdf_path)


def parse_args() -> argparse.Namespace:
    opt = argparse.ArgumentParser(
        description="Execute fracture simulations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    opt.add_argument(
        "-g", "--geometry", nargs="+", choices=["t", "h", "s"], default=["t"]
    )
    opt.add_argument("-l", "--length", nargs="+", type=int, default=[14])
    opt.add_argument(
        "-d", "--disorder", nargs="+", type=float, default=[0, 0.2, 0.4, 0.6, 1.0]
    )
    opt.add_argument("--show", action="store_true")
    opt.add_argument("--graph", action="store_true")
    return opt.parse_args()


def main() -> None:
    args = parse_args()

    DATA_DIR.mkdir(exist_ok=True)
    PDF_DIR.mkdir(exist_ok=True)
    GRAPH_DIR.mkdir(exist_ok=True)

    plt.style.use("ggplot")

    for G in args.geometry:
        for L in args.length:
            print()
            plt.figure()
            plt.suptitle(f"L = {L}, G = {G}")

            for D in args.disorder:
                subprocess.run([str(EXE), str(L), str(D), G])

                if args.graph:
                    render_graph(GRAPH_BEGIN, PDF_DIR / f"{D}output.pdf")
                    render_graph(GRAPH_END, PDF_DIR / f"{D}result.pdf")

                result_path = DATA_DIR / f"{L}.{G}.{D}.csv"
                RESULTS_FILE.replace(result_path)

                out = np.genfromtxt(
                    result_path, delimiter=",", skip_header=1, names=["V", "I"]
                )
                plt.plot(out["V"], out["I"], label=f"D={int(D * 100)}%")

            plt.legend(loc="upper left")
            if args.show:
                plt.show()
            plt.savefig(GRAPH_DIR / f"{G}.{L}.png", bbox_inches="tight")


if __name__ == "__main__":
    main()
