import matplotlib.pyplot as plt
import numpy as np
import sys
from subprocess import run

if len(sys.argv) != 3:
    L = 14
    D = 1.0
else:
    L = sys.argv[1]
    D = sys.argv[2]

results = 'saida.csv'
pdf = 'saida.pdf'
pdf2 = 'resultado.pdf'
graph = 'graph-begin.dot'
graphend = 'graph-end.dot'
exe = 'opt.exe'

run(f'{exe} {L} {D}')

run(f'dot {graph} -Tpdf -o{pdf}')
run(f'start {pdf}', shell=True)
run(f'dot {graphend} -Tpdf -o{pdf2}')
run(f'start {pdf2}', shell=True)

data = np.genfromtxt(results, delimiter=',', skip_header=1, names=['V', 'I'])
plt.plot(data['V'], data['I'], '-')
plt.show()
