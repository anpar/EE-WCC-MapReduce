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

# Produces Fig. 2.

# Allowed latency \tau [s]
lat = [0.1*i for i in range(2, 26)]
# Number of nodes K
K = [10*i for i in range(1, 5)]

# Task parameters [bits]
L = 4e6
D = 1e2
T = 5e3

# Storing the results
outage_prob_opt = np.zeros((len(lat), len(K)))
outage_prob_blind = np.zeros((len(lat), len(K)))
outage_prob_solo = np.zeros((len(lat),))

# Number of iterations
n_iter = int(1e6)

for (i, tau) in enumerate(lat):
    for (j, k) in enumerate(K):
        for n in range(n_iter):
            task = TaskParam(D=D, L=L, T=T, tau=tau)
            comp = CompParam(k, homogeneous=False)
            comm = CommParam(k, B=15e3, N0=1e-9, gap=1, PL=1e-3,
                             homogeneous=False)
            problem = Problem(task, comp, comm)

            outage_prob_opt[i, j] += problem.feasible_opt()/n_iter
            outage_prob_blind[i, j] += problem.feasible_blind()/n_iter

            if k == K[0]:
                outage_prob_solo[i] += problem.feasible_solo()/n_iter

            pprogress(n+1, n_iter,
                      prefix='tau = {:.2e}, K = {:.1e}\t'.format(tau, k),
                      bar_length=40)

# Plotting and saving the results
fix, ax = plt.subplots()
for (j, k) in enumerate(K):
    ax.semilogy(lat, 1-outage_prob_opt[:, j],
                label='\\tiny{{$K = {:d}$ (opt)}}'.format(k),
                marker='o', markersize=4)

for (j, k) in enumerate(K):
    ax.semilogy(lat, 1-outage_prob_blind[:, j],
                label='\\tiny{{$K = {:d}$ (blind)}}'.format(k),
                marker='s', linestyle='--', markersize=4)

ax.semilogy(lat, 1-outage_prob_solo[:],
            label='\\tiny{{$K = 10$ (solo)}}',
            marker='v', markersize=3, linestyle=':')

plt.legend(prop={'size': 6})
plt.ylim([1e-4, 2e0])
plt.xlabel('\small{Allowed latency $\\tau$ [s]}')
plt.ylabel('\small{Outage probability $P^*_{out}$}')

#tikz_save("../../dc-comm/paper/figures/outage-prob.tikz")
plt.show()
