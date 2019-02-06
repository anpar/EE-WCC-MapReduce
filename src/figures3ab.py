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

E_solo = np.zeros((len(K),))

# Number of iterations
n_iter = 1000

for (i, k) in enumerate(K):
    n = 0
    while n < n_iter:
        # Note: we're using a quite large allowed latency (i.e. 3.5s)
        # because we want to compare the energy consumption of the three
        # different schemes on *feasible* instances of the problem. Hence
        # we are limited by the scenario where the nodes do not collaborate
        # which needs a high enough allowed latency to be feasible.
        task = TaskParam(D=D, L=L, T=T, tau=3.5)
        comp = CompParam(k, homogeneous=False)
        comm = CommParam(k, B=15e3, N0=1e-9, gap=1, PL=1e-3,
                         homogeneous=False)
        problem = Problem(task, comp, comm)

        # Only compare energy consumptions when the problem is actually
        # feasible for all the three schemes.
        if problem.feasible_opt() and \
           problem.feasible_blind() and \
           problem.feasible_solo():

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

            # Get energy consumption for the scenario where the nodes do not
            # collaborate
            l = np.repeat(problem.task.L, k)
            opt = Solution(problem, l, lamb)

            E_solo[i] += np.sum(opt.E_solo)/n_iter

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
ax.semilogy(K, E_solo, 'r',
            label='\\tiny{{No collaboration}}',
            marker='v', markersize=4, linestyle=':')

plt.xlabel('\small{Number of nodes $K$}')
plt.ylabel('\small{Total energy consumption [J]}')
ax.legend()

#tikz_save("../../dc-comm/paper/figures/energy_vs_K.tex")
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

#tikz_save("../../dc-comm/paper/figures/energy_breakdown.tex")
plt.show()
