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

import os
import matplotlib.pyplot as plt
import numpy as np
import sys
from subprocess import run, DEVNULL

if len(sys.argv) < 2:
    L = 14
else:
    L = sys.argv[1]

if len(sys.argv) < 3:
    tipo = "t"
else:
    tipo = sys.argv[2]

outfolder = 'results'
results = 'output.csv'
pdf = 'output.pdf'
pdf2 = 'result.pdf'
graph = 'graph-begin.dot'
graphend = 'graph-end.dot'

exename = 'fracture'
if sys.platform == 'win32':
    exename += '.exe'

exe = os.path.join('out', exename)

Ds = [0]
Ds = [0, 0.2, 0.4, 0.6, 0.8, 1]
data = []

if not os.path.isfile(exe):
    run(os.path.join(sys.path[0], 'build'), shell=True)

for D in Ds:
    run(f'{exe} {L} {D} {tipo}')

    os.makedirs(outfolder, exist_ok=True)

    # begin_path = os.path.join(outfolder, pdf)
    # run(f'dot {graph} -Tpdf -o{begin_path}')
    # os.remove(graph)
    # run(f'start {begin_path}', shell=True)

    # end_path = os.path.join(outfolder, pdf2)
    # run(f'dot {graphend} -Tpdf -o{end_path}')
    # os.remove(graphend)
    # run(f'start {end_path}', shell=True)

    result_path = os.path.join(outfolder, f'{D}.csv')
    os.replace(results, result_path)

    out = np.genfromtxt(result_path, delimiter=',', skip_header=1, names=['V', 'I'])
    data.append((D, out['V'], out['I']))
    # plt.plot(data['V'], data['I'], linestyle='-', marker='.', linewidth=1)

for D, V, I in data:
    plt.plot(V, I, label=f'D={int(D*100)}%')

plt.legend(loc='upper left')
plt.show()
