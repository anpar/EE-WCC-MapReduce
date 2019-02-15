"""
MIT License

Copyright (c) 2019 Université Catholique de Louvain (UCLouvain)

The software provided allows to reproduce the results presented in the
research paper "Energy-Efficient Edge-Facilitated Wireless Collaborative
Computing using Map-Reduce" by Antoine Paris, Hamed Mirghasemi, Ivan Stupia
and Luc Vandendorpe from ICTEAM/ELEN/CoSy (UCLouvain).

Contact: antoine.paris@uclouvain.be

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import matplotlib.pyplot as plt

#from matplotlib2tikz import save as tikz_save
from core import *
from utils import *

# Produces Fig. 3a and 3b

# Number of nodes K
K = [10 * i for i in range(2, 11)]

# Task parameters [bits]
L = 4e6
D = 1e2
T = 5e3

# Storing the results
E_map_opt = np.zeros((len(K),))
E_red_opt = np.zeros((len(K),))
E_shu_opt = np.zeros((len(K),))

E_map_blind = np.zeros((len(K),))
E_red_blind = np.zeros((len(K),))
E_shu_blind = np.zeros((len(K),))

# Number of iterations
n_iter = 10000

for (i, k) in enumerate(K):
    n = 0
    while n < n_iter:
        # Note: we're using a quite large allowed latency (i.e. 0.8s)
        # because we want to compare the energy consumption of both schemes
        # on *feasible* instances of the problem. Hence,
        # we are limited by the scenario where the nodes collaborate
        # naively in setting the allowed latency.
        task = TaskParam(D=D, L=L, T=T, tau=1)
        comp = CompParam(k, homogeneous=False)
        comm = CommParam(k, B=15e3, N0=1e-9, gap=1, PL=1e-3,
                         homogeneous=False)
        problem = Problem(task, comp, comm)

        # Only compare energy consumptions when the problem is actually
        # feasible for both scheme.
        if problem.feasible_opt() and \
           problem.feasible_blind(1.02):

            # Get energy consumptions for the optimal scheme
            solver = Solver(problem)
            (l, lamb) = solver.solve()
            opt = Solution(problem, l, lamb)

            E_map_opt[i] += np.sum(opt.E_map)/n_iter
            E_red_opt[i] += np.sum(opt.E_red)/n_iter
            E_shu_opt[i] += np.sum(opt.E_shu)/n_iter

            # Get energy consumptions for the blind scheme
            # Uniformly distribute the file w
            l = np.repeat(problem.task.L/k, k)
            opt = Solution(problem, l, lamb)

            E_map_blind[i] += np.sum(opt.E_map)/n_iter
            E_red_blind[i] += np.sum(opt.E_red)/n_iter
            E_shu_blind[i] += np.sum(opt.E_shu)/n_iter

            n += 1
            pprogress(n, n_iter, prefix='K = {:d}\t'.format(k),
                           bar_length=40)

# Plotting and saving the results (Fig. 3)
fix, ax = plt.subplots()

ax.semilogy(K, E_map_opt + E_red_opt + E_shu_opt, 'g',
            label='\\tiny{{Optimal scheme}}',
           marker='o', markersize=4)
ax.semilogy(K, E_map_blind + E_red_blind + E_shu_blind, 'b',
            label='\\tiny{{Blind scheme}}',
            marker='s', markersize=4, linestyle='--')

plt.xlabel('\small{Number of nodes $K$}')
plt.ylabel('\small{Total energy consumption [J]}')
ax.legend()

#tikz_save("../../dc-comm/paper/figures/tot-energy-vs-nodes.tikz")
plt.show()

# Plotting and saving the results (Fig. 4)
fix, ax = plt.subplots()

# Optimal scheme
ax.semilogy(K, E_map_opt, 'blue',
           label='\\tiny{{$E^{{MAP}}$ (opt)}}',
           marker='o', markersize=4)
ax.semilogy(K, E_red_opt, 'green',
           label='\\tiny{{$E^{{RED}}$ (opt)}}',
           marker='o', markersize=3)
ax.semilogy(K, E_shu_opt, 'orange',
           label='\\tiny{{$E^{{SHU}}$ (opt)}}',
           marker='o', markersize=4)

# Blind scheme
ax.semilogy(K, E_map_blind, 'blue',
            label='\\tiny{{$E^{{MAP}}$ (blind)}}',
            marker='s', markersize=4, linestyle='--')
ax.semilogy(K, E_red_blind, 'green',
            label='\\tiny{{$E^{{RED}}$ (blind)}}',
            marker='s', markersize=4, linestyle='--')
ax.semilogy(K, E_shu_blind, 'orange',
            label='\\tiny{{$E^{{SHU}}$ (blind)}}',
            marker='s', markersize=4, linestyle='--')

plt.xlabel('\small{Number of nodes $K$}')
plt.ylabel('\small{Energy consumption [J]}')
ax.legend()

#tikz_save("../../dc-comm/paper/figures/energy-breakdown.tikz")
plt.show()
