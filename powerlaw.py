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
    choices=['t', 'h', 's'],
    help='Geometry type (required for mode=disorder)'
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


def fit_size(data, alfa, beta, L):
    V = data['V']
    I = data['I']
    vlb = V / (L**beta)
    ilb = I / (L**alfa)
    return ilb, vlb


def fit_disorder(data, eta, nu, D):
    V = data['V']
    I = data['I']
    vlb = V * (1 + D)**nu
    ilb = I * (1 + D)**eta
    return ilb, vlb


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
    print('\n'.join(files))
    sizes = [7, 14, 20, 28]
    data = [np.genfromtxt(path, delimiter=',', skip_header=1, names=['V', 'I'])
            for path in files]

    while True:
        try:
            alfa = float(input('\nalfa: '))
            beta = float(input('beta: '))
        except ValueError:
            break
        fitresult = [(L, fit_size(d, alfa, beta, L)) for L, d in zip(sizes, data)]

        for L, (ilb, vlb) in fitresult:
            plt.plot(vlb, ilb, label=f'L = {L}')

        plt.legend(loc='upper left')
        plt.grid(True)
        plt.axis('equal')
        plt.xlabel(r'$V/L^\beta$')
        plt.ylabel(r'$I/L^\alpha$')
        plt.title(f'$D = 0$, $\\alpha = {alfa}$ e $\\beta = {beta}$')
        plt.show()

else:
    files = sorted(glob.glob(args.source + "/*.csv"), key=sort_by_disorder)
    print('\n'.join(files))
    disorders = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    data = [np.genfromtxt(path, delimiter=',', skip_header=1, names=['V', 'I'])
            for path in files]

    while True:
        try:
            eta = float(input('\neta: '))
            nu = float(input('nu: '))
        except ValueError:
            break
        fitresult = [(D, fit_disorder(d, eta, nu, D)) for D, d in zip(disorders, data)]

        for D, (ilb, vlb) in fitresult:
            plt.plot(vlb, ilb, label=f'D = {D}')

        plt.legend(loc='upper left')
        plt.grid(True)
        plt.xlabel(r'$V(1+D)^{\nu}$')
        plt.ylabel(r'$I(1+D)^{\eta}$')

        length = args.length
        geometry = convert_geometry(args.geometry)
        plt.title(f'$L = {length}$, G = {geometry}, $\\eta = {eta}$ e $\\nu = {nu}$')
        plt.show()
