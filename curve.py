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
	'-d', '--disorder',
	type=float
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
    length = filename.split('.')[0]
    return int(length)


args = opt.parse_args()
files = sorted(glob.glob(args.source + "/*.csv"), key=sort_files)
sizes = [7, 14, 20, 28]
data = [np.genfromtxt(path, delimiter=',', skip_header=1, names=['V', 'I'])
        for path in files]

for L, data in zip(sizes, data):
    plt.plot(data['V'], data['I'], label=f'L = {L}')

plt.legend(loc='upper left')
plt.grid(True)
plt.axis('equal')
plt.xlabel(r'$V$')
plt.ylabel(r'$I$')
disorder = args.disorder
geometry = convert_geometry(args.geometry)
plt.title(f'$D = {disorder}, G = ${geometry}')
plt.show()
