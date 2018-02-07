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
import sys
from pathlib import Path

opt = argparse.ArgumentParser(
    description='Plot folder.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

opt.add_argument(
    '-f', '--folder',
    required=True
)

opt.add_argument(
    '--noshow',
    action='store_true'
)

opt.add_argument(
    '-g', '--geometry',
    choices=['t', 'h', 's'],
    required=True
)

opt.add_argument(
    '-l', '--length',
    type=int,
    required=True
)

plt.style.use('ggplot')

args = opt.parse_args();

print('')
plt.figure()
plt.suptitle(f'L = {args.length}, G = {args.geometry}')

pathlist = Path(args.folder).glob('*.csv')

for path in pathlist:
    out = np.genfromtxt(str(path), delimiter=',',
                        skip_header=1, names=['V', 'I'])
    parts = path.stem.split('.')
    D = '.'.join(parts[2:])
    lines = plt.plot(out['V'], out['I'], label=f'D = {D}')

plt.legend(loc='upper left')

plt.savefig(args.folder, bbox_inches='tight')

if not args.noshow:
    plt.show()
