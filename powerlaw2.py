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


def fit(data, eta, nu, D):
    V = data['V']
    I = data['I']

    vlb = V * (1 + D)**nu
    ilb = I * (1 + D)**eta

    return ilb, vlb


def sort_files(path):
    filename = os.path.basename(path)
    disorder = filename.split('.')[2]
    return float(disorder)


args = opt.parse_args()
files = sorted(glob.glob(args.source + "/*.csv"), key=sort_files)
disorders = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
data = [np.genfromtxt(path, delimiter=',', skip_header=1, names=['V', 'I'])
        for path in files]

while True:
    try:
        eta = float(input('\neta: '))
        nu = float(input('nu: '))
    except ValueError:
        break
    fitresult = [(D, fit(d, eta, nu, D)) for D, d in zip(disorders, data)]

    for D, (ilb, vlb) in fitresult:
        plt.plot(vlb, ilb, label=f'D = {D}')

    plt.legend(loc='upper left')
    plt.grid(True)
    plt.axis('equal')

    plt.xlabel(r'$V(1+D)^{\nu}$')
    plt.ylabel(r'$I(1+D)^{\eta}$')

    length = args.length
    geometry = convert_geometry(args.geometry)
    plt.title(f'$L = {length}$, G = {geometry}, $\\eta = {eta}$ e $\\nu = {nu}$')
    plt.show()
