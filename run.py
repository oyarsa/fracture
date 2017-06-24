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

if len(sys.argv) < 3:
    L = 14
    D = 1.0
else:
    L = sys.argv[1]
    D = sys.argv[2]

if len(sys.argv) < 4:
    tipo = "t"
else:
    tipo = sys.argv[3]

outfolder = '.\\result\\'
results = 'output.csv'
pdf = 'output.pdf'
pdf2 = 'result.pdf'
graph = 'graph-begin.dot'
graphend = 'graph-end.dot'
exe = '.\\out\\fracture.exe'

run(f'{exe} {L} {D} {tipo}')

run(f'if not exist "{outfolder}" mkdir {outfolder}', shell=True)

run(f'dot {graph} -Tpdf -o{outfolder}{pdf}')
run(f'del {graph}', shell=True)
run(f'start {outfolder}{pdf}', shell=True)

run(f'dot {graphend} -Tpdf -o{outfolder}{pdf2}')
run(f'del {graphend}', shell=True)
run(f'start {outfolder}{pdf2}', shell=True)

run(f'move {results} {outfolder}{results}', shell=True, stdout=DEVNULL)

data = np.genfromtxt(f'{outfolder}{results}', delimiter=',', skip_header=1, names=['V', 'I'])
plt.plot(data['V'], data['I'], linestyle='-', marker='.', linewidth=1)
plt.show()
