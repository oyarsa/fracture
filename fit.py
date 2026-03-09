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

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import subprocess
import sys

opt = argparse.ArgumentParser(
    description='Execute fracture simulations.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

opt.add_argument(
    '-g', '--geometry',
    choices=['t', 'h', 's'],
    default='t'
)

opt.add_argument(
    '-l', '--length',
    type=int,
    default=14
)

opt.add_argument(
    '-c', '--count',
    type=int,
    default=10
)

opt.add_argument(
    '-d', '--disorder',
    type=float,
    default=0
)

opt.add_argument(
    '--noshow',
    action='store_true'
)

opt.add_argument(
    '--graph',
    action='store_true'
)

datafolder = 'data'
pdffolder = 'pdf'
graphfolder = 'graphs'
results = 'output.csv'
pdf = 'output.pdf'
pdf2 = 'result.pdf'
graph = 'graph-begin.dot'
graphend = 'graph-end.dot'

os.makedirs(datafolder, exist_ok=True)
os.makedirs(pdffolder, exist_ok=True)
os.makedirs(graphfolder, exist_ok=True)

exe = os.path.join('out', 'fracture')
opener = 'open' if sys.platform == 'darwin' else 'xdg-open'

plt.style.use('ggplot')

args = opt.parse_args();
G = args.geometry
L = args.length
D = args.disorder

print('')
plt.figure()
plt.suptitle(f'L = {L}, G = {G}, D = {D}')

color_map = plt.get_cmap('gist_rainbow')

for r in range(args.count):
    subprocess.run(f'{exe} {L} {D} {G}')

    if args.graph:
        begin_path = os.path.join(pdffolder, str(D) + pdf)
        subprocess.run(f'dot {graph} -Tpdf -o{begin_path}')
        os.remove(graph)
        subprocess.run([opener, begin_path])

        end_path = os.path.join(pdffolder, str(D) + pdf2)
        subprocess.run(f'dot {graphend} -Tpdf -o{end_path}')
        os.remove(graphend)
        subprocess.run([opener, end_path])

    result_path = os.path.join(datafolder, f'{L}.{G}.{D}-{r}.csv')
    os.replace(results, result_path)

    out = np.genfromtxt(result_path, delimiter=',',
                        skip_header=1, names=['V', 'I'])
    lines = plt.plot(out['V'], out['I'], label=f'{r}')
    lines[0].set_color(color_map(r/args.count))

plt.legend(loc='upper left')

fig_path = os.path.join(graphfolder, f'{G}.{L}.png')
plt.savefig(fig_path, bbox_inches='tight')

fig_path2 = os.path.join(datafolder, f'{G}.{L}.{D}.png')
plt.savefig(fig_path2, bbox_inches='tight')

if not args.noshow:
    plt.show()
