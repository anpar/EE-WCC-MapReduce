import matplotlib.pyplot as plt

#from matplotlib2tikz import save as tikz_save
from core import *
from utils import *

# Produces Fig. 5

# Allowed latency \tau [s]
lat = [0.25 * i for i in range(1, 13)]

# Task parameters [bits]
L = 4e6
D = 1e2
T = 5e3

# Number of nodes K
K = 60

# Storing the results
E_map_opt = np.zeros((len(lat),))
E_shu_opt = np.zeros((len(lat),))
E_red_opt = np.zeros((len(lat),))

# Number of iterations
n_iter = 1000

for (i, tau) in enumerate(lat):
    n = 0
    while n < n_iter:
        task = TaskParam(D=D, L=L, T=T, tau=tau)
        comp = CompParam(K, homogeneous=False)
        comm = CommParam(K, B=15e3, N0=1e-9, gap=1, PL=1e-3,
                         homogeneous=False)
        problem = Problem(task, comp, comm)

        # Check if the problem is feasible by the optimal scheme first
        if problem.feasible_opt():
            # Get the energy consumptions of the optimal scheme
            solver = Solver(problem)
            (l, lamb) = solver.solve()
            opt = Solution(problem, l, lamb)

            E_map_opt[i] += np.sum(opt.E_map)/n_iter
            E_red_opt[i] += np.sum(opt.E_red)/n_iter
            E_shu_opt[i] += np.sum(opt.E_shu)/n_iter

            n += 1
            pprogress(n, n_iter, prefix='tau = {:.2f}\t'.format(tau),
                      bar_length=40)

# Plotting and saving the results
fix, ax = plt.subplots()
ax.semilogy(lat, E_map_opt, color='blue',
        marker='s', markersize=2, label='\scriptsize{{$E^{{MAP}}$}}')
ax.semilogy(lat, E_shu_opt, color='orange',
        marker='s', markersize=2, label='\scriptsize{{$E^{{SHU}}$}}')
ax.semilogy(lat, E_red_opt, color='green',
        marker='s', markersize=2, label='\scriptsize{{$E^{{RED}}$}}')

plt.xlabel('\small{Allowed latency \\tau [s]}')
plt.ylabel('\small{Energy consumption [J]}')
ax.legend()

#tikz_save("../../dc-comm/paper/figures/energy_vs_lat.tex")
plt.show()