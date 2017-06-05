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

import matplotlib.pyplot as plt
import itertools
from graphviz import Digraph
#
import timeit
import cProfile
import pstats

inf = 1e9

def gerar_grid_serial(L, D):
    g = Circuito()

    init = Node(0, 0)
    g.add_node(init)

    prev = init
    for i in range(1, L + 1):
        node = Node(i, i)
        g.add_node(node)
        R = 0 if prev is init else 20
        Imax = inf if prev is init else 20
        g.add_edge(Edge(prev, node, Fusivel(R, Imax)))
        prev = node

    finish = Node(L + 1, L + 1)
    g.add_node(finish)
    g.add_edge(Edge(prev, finish, Fusivel(0, inf)))

    return g


def gerar_grid_paralelo(L, D):
    g = Circuito()

    init = Node(0, 0)
    g.add_node(init)

    finish = Node(L + 1, 3)

    for i in range(1, L, 2):
        node_a = Node(i, 1)
        g.add_node(node_a)
        g.add_edge(Edge(init, node_a, Fusivel(0, inf)))

        node_b = Node(i + 1, 2)
        g.add_node(node_b)
        g.add_edge(Edge(node_a, node_b, Fusivel(20, 20 * i)))

        g.add_edge(Edge(node_b, finish, Fusivel(0, inf)))

    g.add_node(finish)

    return g


def desenhar_grafo(g):
    dot = Digraph()
    dot.attr('graph', rankdir='LR')
    dot.attr('node', shape='point', label='', width='0.05', height='0.05')
    dot.attr('edge', arrowhead='empty', arrowsize='0.5', fontsize='8')

    for node in g.nodes:
        dot.node(str(node.num))

        for edge in g.adjacencias[node.num]:
            f = edge.fusivel
            if f.R == 0:
                label = ''
            else:
                label = f'{f.R}Ω, {f.Imax}A'
            dot.edge(str(node.num), str(edge.dst.num), label)

    print(dot.source)
    dot.render('test', view=True)


def gerar_grid(L, D):
    # return gerar_grid_serial(L, D)
    return gerar_grid_paralelo(L,D)


def calcula_corrente(g, V):
    # return calcula_corrente_serial(g, V)
    return calcula_corrente_paralelo(g, V)


def flatten(listOfLists):
    "Flatten one level of nesting"
    return itertools.chain.from_iterable(listOfLists)


def calcula_corrente_serial(g, V):
    edges = flatten(w for _, w in g.adjacencias.items())
    weights = (e.fusivel.R for e in edges)
    resistance = sum(weights)
    return V / resistance


def calcula_corrente_paralelo(g, V):
    edges = flatten(w for _, w in g.adjacencias.items())
    weights = (1 / e.fusivel.R for e in edges if e.fusivel.R != 0)
    resistance = sum(weights)
    return V * resistance


class Fusivel:
    def __init__(self, R, Imax):
        self.R = R
        self.Imax = Imax

    def queimou(self, I):
        return I > self.Imax


class Edge:
    def __init__(self, src, dst, fusivel):
        self.src = src
        self.dst = dst
        self.fusivel = fusivel


class Node:
    def __init__(self, num, nivel):
        self.num = num
        self.nivel = nivel


class Circuito:
    def __init__(self):
        self.nodes = []
        self.niveis = []
        self.adjacencias = {}

    def add_node(self, node):
        self.nodes.append(node)
        self.adjacencias[node.num] = []
        self.add_node_to_nivel(node)

    def add_node_to_nivel(self, node):
        if len(self.niveis) <= node.nivel:
            self.niveis.append([])
        assert (len(self.niveis) >= node.nivel + 1)
        self.niveis[node.nivel].append(node)

    def add_edge(self, edge):
        self.adjacencias[edge.src.num].append(edge)

    def remove(self, p):
        self.nodes.remove(p)
        self.niveis[p.nivel].remove(p)
        del self.adjacencias[p.num]

        for adj in self.adjacencias.values():
            for i, e in enumerate(adj):
                if e.dst == p:
                    del adj[i]
                    break

    def remove_edge(self, e):
        self.adjacencias[e.src.num].remove(e)

    def conectado(self):
        inicio = self.nodes[0]
        fim = self.nodes[-1]
        return self.dfs(inicio, fim, set())


    def dfs(self, cur, target, percorridos):
        if cur == target:
            return True

        percorridos.add(cur)
        for nxt in self.adjacencias[cur.num]:
            if nxt.dst not in percorridos and self.dfs(nxt.dst, target, percorridos):
                return True

        return False


def iteracao(g, V, deltaV, registro):
    """Tenta fazer uma iteração. Se alguém queimar, retorna a mesma
    ddp para que a simulação dessa rodada seja reiniciada com os
    fusíveis queimados. Se ninguém queimar, registra o resultado e
    retorna a próxima voltagem, para que comece a próxima etapa
    (uma etapa só termina quando a ddp muda, várias iterações podem
    acontecer na mesma etapa).
    """
    I = calcula_corrente(g, V)

    for pontos in g.niveis:
        queimado = False

        # A verificação é feita por níveis, de forma que fusíveis
        # nos mesmos níveis queimam juntos.
        for p in pontos:
            for e in g.adjacencias[p.num][:]:
                if e.fusivel.queimou(I):
                    # print(f'Queimou: {p.num}')
                    g.remove_edge(e)
                    queimado = True
        if queimado:
            return V

    registro.registrar(V, I)
    return V + deltaV


class Registro:
    def __init__(self):
        self.V = []
        self.I = []

    def registrar(self, V, I):
        self.V.append(V)
        self.I.append(I)
        # print(f'V: {V}, I: {I}')

    def plot(self):
        plt.plot(self.V, self.I)
        plt.show()


def simulacao(L, D, V0, deltaV):
    g = gerar_grid(L, D)
    desenhar_grafo(g)
    V = V0
    registro = Registro()

    while g.conectado():
        V = iteracao(g, V, deltaV, registro)

    registro.registrar(V, 0)
    return registro


def main():
    r = simulacao(20, 0, 0, 0.5)
    r.plot()

def profile():
    cProfile.run('simulacao(20, 0, 0, 0.1)', 'restats')
    p = pstats.Stats('restats')
    p.strip_dirs().sort_stats('time').print_stats()


def time():
    n = 1
    t = timeit.timeit('simulacao(20, 0, 0, 0.1)', globals=globals(), number=n)
    print(f'Total: {t}s, média: {t/n}s')


if __name__ == '__main__':
    # main()
    profile()
    # time()
