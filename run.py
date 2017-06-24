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

outfolder = '.\\result\\'
results = 'output.csv'
pdf = 'output.pdf'
pdf2 = 'result.pdf'
graph = 'graph-begin.dot'
graphend = 'graph-end.dot'
exe = '.\\out\\fracture.exe'

Ds = [0, 0.2, 0.4, 0.6, 0.8, 1]
data = []

for D in Ds:
    run(f'{exe} {L} {D} {tipo}')

    run(f'if not exist "{outfolder}" mkdir {outfolder}', shell=True)

    # run(f'dot {graph} -Tpdf -o{outfolder}{pdf}')
    # run(f'del {graph}', shell=True)
    # run(f'start {outfolder}{pdf}', shell=True)

    # run(f'dot {graphend} -Tpdf -o{outfolder}{pdf2}')
    # run(f'del {graphend}', shell=True)
    # run(f'start {outfolder}{pdf2}', shell=True)

    run(f'move {results} {outfolder}{D}{results}', shell=True, stdout=DEVNULL)

    out = np.genfromtxt(f'{outfolder}{D}{results}', delimiter=',', skip_header=1, names=['V', 'I'])
    data.append((D, out['V'], out['I']))
    # plt.plot(data['V'], data['I'], linestyle='-', marker='.', linewidth=1)

for D, V, I in data:
    plt.plot(V, I, label=f'D={D*100}%')

plt.legend(loc='upper left')
plt.show()
