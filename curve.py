import argparse
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

opt = argparse.ArgumentParser(
    description='Plot fracture V-I curves.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

opt.add_argument(
    '-s', '--source',
    type=str,
    required=True
)

opt.add_argument(
    '-m', '--mode',
    type=str,
    choices=['size', 'disorder'],
    required=True,
    help='Vary over lattice size or disorder'
)

opt.add_argument(
    '-g', '--geometry',
    type=str,
    choices=['t', 'h', 's']
)

opt.add_argument(
    '-d', '--disorder',
    type=float,
    help='Fixed disorder value (required for mode=size)'
)

opt.add_argument(
    '-l', '--length',
    type=int,
    help='Fixed length value (required for mode=disorder)'
)


def convert_geometry(initial):
    return {
        't': '45 graus',
        'h': 'hexagonal',
        's': 'quadrada'
    }[initial]


def sort_by_size(path):
    filename = os.path.basename(path)
    length = filename.split('.')[0]
    return int(length)


def sort_by_disorder(path):
    filename = os.path.basename(path)
    disorder = filename.split('.')[2]
    disorder = disorder.replace(',', '.')
    return float(disorder)


args = opt.parse_args()

if args.mode == 'size':
    files = sorted(glob.glob(args.source + "/*.csv"), key=sort_by_size)
    sizes = [7, 14, 20, 28]
    data = [np.genfromtxt(path, delimiter=',', skip_header=1, names=['V', 'I'])
            for path in files]

    for L, d in zip(sizes, data):
        plt.plot(d['V'], d['I'], label=f'L = {L}')

    disorder = args.disorder
    geometry = convert_geometry(args.geometry)
    plt.title(f'$D = {disorder}, G = ${geometry}')

else:
    files = sorted(glob.glob(args.source + "/*.csv"), key=sort_by_disorder)
    disorders = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    data = [np.genfromtxt(path, delimiter=',', skip_header=1, names=['V', 'I'])
            for path in files]

    for D, d in zip(disorders, data):
        plt.plot(d['V'], d['I'], label=f'D = {D}')

    length = args.length
    geometry = convert_geometry(args.geometry)
    plt.title(f'$L = {length}$, G = {geometry}')

plt.legend(loc='upper left')
plt.grid(True)
plt.axis('equal')
plt.xlabel(r'$V$')
plt.ylabel(r'$I$')
plt.show()
