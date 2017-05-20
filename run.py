import matplotlib.pyplot as plt
import numpy as np
from subprocess import run

results = 'saida.csv'
pdf = 'saida.pdf'
graph = 'graph.dot'
exe = 'opt.exe'

run(exe)

run(f'dot {graph} -Tpdf -o{pdf}')
run(f'start {pdf}', shell=True)

data = np.genfromtxt(results, delimiter=',', skip_header=1, names=['V', 'I'])
plt.plot(data['V'], data['I'])
plt.show()
