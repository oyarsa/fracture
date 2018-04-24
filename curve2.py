import argparse
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

opt = argparse.ArgumentParser(
    description='Evaluate fracture power law.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

opt.add_argument(
    '-s', '--source',
    type=str
)

opt.add_argument(
  '-l', '--length',
  type=int
)

opt.add_argument(
    '-g', '--geometry',
    type=str,
    choices=['t', 'h', 's']
)

def convert_geometry(initial):
  return {
    't': '45 graus',
    'h': 'hexagonal',
    's': 'quadrada'
  }[initial]


def sort_files(path):
    filename = os.path.basename(path)
    disorder = filename.split('.')[2]
    disorder = disorder.replace(',', '.')
    return float(disorder)


args = opt.parse_args()
files = sorted(glob.glob(args.source + "/*.csv"), key=sort_files)
disorders = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
data = [np.genfromtxt(path, delimiter=',', skip_header=1, names=['V', 'I'])
        for path in files]

for D, d in zip(disorders, data):
    plt.plot(d['V'], d['I'], label=f'D = {D}')

plt.legend(loc='upper left')
plt.grid(True)
plt.axis('equal')

plt.xlabel(r'$V$')
plt.ylabel(r'$I$')

#plt.gca().set_adjustable("box")
#plt.gca().set_ylim(bottom=0.0)

length = args.length
geometry = convert_geometry(args.geometry)
plt.title(f'$L = {length}$, G = {geometry}')
plt.show()
