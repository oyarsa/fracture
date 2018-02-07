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

def fit(data, alfa, beta, L):
    V = data['V']
    I = data['I']

    vlb = V / (L**beta)
    ilb = I / (L**alfa)

    return ilb, vlb


def sort_files(path):
    filename = os.path.basename(path)
    length = filename.split('.')[0]
    return int(length)


args = opt.parse_args()
files = sorted(glob.glob(args.source + "/*.csv"), key=sort_files)
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
    fitresult = [(L, fit(d, alfa, beta, L)) for L, d in zip(sizes, data)]

    for L, (ilb, vlb) in fitresult:
        plt.plot(vlb, ilb, label=f'L = {L}')

    plt.legend(loc='upper left')
    plt.grid(True)
    plt.axis('equal')
    plt.xlabel(r'$V/L^\beta$')
    plt.ylabel(r'$I/L^\alpha$')
    plt.title(f'$D = 0$, $\\alpha = {alfa}$ e $\\beta = {beta}$')
    plt.show()
