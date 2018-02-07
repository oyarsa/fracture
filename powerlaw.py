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

def fit(data, alfa, beta, L):
    V = data['V']
    I = data['I']

    vlb = V * (L**-beta)
    ilb = I * (L**-alfa)

    return ilb, vlb


args = opt.parse_args()
data = [np.genfromtxt(path, delimiter=',', skip_header=1, names=['V', 'I'])
        for path in glob.glob(args.source + "/*.csv")]

while True:
    try:
        alfa = float(input('\nalfa: '))
        beta = float(input('beta: '))
    except ValueError:
        break
    fitresult = [(args.length, fit(d, alfa, beta, args.length)) for d in data]

    for l, (ilb, vlb) in fitresult:
        plt.plot(vlb, ilb, label=f'L = {l}')

    plt.legend(loc='upper left')
    plt.show()
