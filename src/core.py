import numpy as np

from math import sqrt
from mpmath import lambertw
from termcolor import colored
from numpy.random import uniform, choice, normal

class TaskParam:
    """
    Contains all the parameters related to the tasks to be executed:
        - D: node input size [bits]
        - L: large file size [bits]
        - T: intermediate value size [bits]
        - tau = allowed latency [s]
    """
    def __init__(self, D=1e3, L=1e9, T=1e4, tau=1):
        self.D = D
        self.L = L
        self.T = T
        self.tau = tau

class CompParam:
    """
    Contains all the parameters related to the computation capabilities
    of the nodes:
        - K: number of nodes
        - homogeneous: indicates whether the nodes are homogeneous in term
        of computation capabilities
        - C: K-length vector, C_k is the number of CPU cycles required for
        computing 1-bit of input at node k [CPU cycles/bit]
        - P: K-length vector, P_k is the power required for each CPU cycle
        at node k [W/CPU cycle]
        - F: K-length vector, F_k is the clock frequency of node k [CPU
        cycles/s]
    """
    def __init__(self, K, homogeneous=False):
        self.K = K

        self.C = uniform(low=500, high=1500, size=(K,))
        self.P = uniform(low=1e-11, high=20e-11, size=(K,))
        self.F = choice([i * 1e8 for i in range(1, 11)], size=(K,))

        if homogeneous:
            self.C = np.repeat(self.C[0], K)
            self.P = np.repeat(self.P[0], K)
            self.F = np.repeat(self.F[0], K)

class CommParam:
    """
    Contains all the parameters related to the communication between the
    the nodes:
        - K: number of nodes
        - homogeneous: indicates whether the nodes are homogeneous in term
        of channel strength to the AP
        - B: bandwidth [Hz]
        - N0: variance of complex white Gaussian noise
        - gap: SNR gap
        - PL: average power loss
        - h: K-length vector, h_k is the Rayleigh fading channel from node k
        to the AP
    """
    def __init__(self, K, B=1e6, N0=1e-9, gap=1, PL=1e-3, homogeneous=False):
        self.K = K

        self.B = B
        self.N0 = N0
        self.gap = gap

        self.h = sqrt(PL/2) * (normal(size=(K,)) + 1j*normal(size=(K,)))

        if homogeneous:
            self.h = np.repeat(self.h[0], K)

class Problem:
    """
    Bundles TaskParam, CompParam and CommParam together to create a instance of
    the problem.
    """
    def __init__(self, task, comp, comm):
        assert comp.K == comm.K, "Parameters inconsistency"

        self.K = comp.K

        self.task = task
        self.comp = comp
        self.comm = comm

        # tauk is a K-length vector containing the effective latency constraint
        # for each node, removing from tau the time needed to process the
        # inputs from all the nodes and the time needed by the slowest node to
        # compute its Reduce function
        self.tauk = task.tau - self.K * task.D * comp.C/comp.F - \
                task.T * np.max(comp.C/comp.F)
        assert np.all(self.tauk > 0), "Effective latency < 0"

    def max_comp_load_opt(self):
        """
        Returns the maximum achievable computation load of the optimal
        collaborative-computing scheme.
        """
        return np.sum(self.comp.F/self.comp.C * self.tauk)

    def max_comp_load_blind(self):
        """
        Returns the maximum achievable computation load of the blind
        collaborative-computing scheme.
        """
        return np.min(self.comp.F/self.comp.C * self.tauk) * self.K

    def max_comp_load_solo(self):
        """
        Returns the maximum achievable computation load in the case
        where the nodes are not collaborating.
        """
        return (self.task.tau - self.task.D * self.comp.C[0]/self.comp.F[0]) * \
                self.comp.F[0]/self.comp.C[0]

    def feasible_opt(self):
        """
        Check if the problem is feasible by the optimal scheme by comparing
        the actual load with the maximum achievable load.
        """
        return self.max_comp_load_opt() > self.task.L

    def feasible_blind(self):
        """
        Check if the problem is feasible by the blind scheme by comparing
        the actual load with the maximum achievable load.
        """
        return self.max_comp_load_blind() > self.task.L

    def feasible_solo(self):
        """
        Check if the problem is feasible without collaboration by comparing
        the actual load with the maximum achievable load.
        """
        return self.max_comp_load_solo() > self.task.L

class Solver:
    def __init__(self, problem):
        self.problem = problem

    def optimum(self, lamb):
        """
        Given the optimal value of the Lagrange multiplier \lambda^*, computes
        the optimal values of t^\text{SHU}_k and l_k using the threshold-based
        policy given by Eq. (10) and Eq. (11)
        """
        # Easing notations by unpacking all parameters
        (K, tauk) = (self.problem.K, self.problem.tauk)
        (L, T) = (self.problem.task.L,
                  self.problem.task.T)
        (C, P, F) = (self.problem.comp.C,
                     self.problem.comp.P,
                     self.problem.comp.F)
        (B, N0, gap, h) = (self.problem.comm.B,
                           self.problem.comm.N0,
                           self.problem.comm.gap,
                           self.problem.comm.h)

        a = (K-1) * (T/L)   # \alpha in the paper

        # t is a vector containing the K t^\text{SHU}_k for k \in [K]
        # l is a vector containing the K l_k for k in [K]
        (t, l) = (np.zeros((K,)), np.zeros((K,)))

        # For each node
        for k in range(K):
            # Computing the left term of Eq. (10)
            th = C[k]*P[k] + a*gap*N0/np.abs(h[k])**2 * np.log(2)/B
            # and comparing it to the threshold given by \lambda^*
            if th >= lamb:
                # if higher, node k does not participate to the Map and Reduces
                # phases
                t[k] = tauk[k]
                l[k] = 0
            else:
                # else, compute optimal t^\text{SHU}_k with Eq. (11)
                tmp = (np.abs(h[k])**2/(gap*N0) * F[k]/C[k] * \
                       (lamb - C[k]*P[k]) - 1) * \
                        np.exp(a * np.log(2)/B * F[k]/C[k] - 1)
                t[k] = (a * F[k]/C[k] * np.log(2)/B * tauk[k]) \
                        / (np.real(lambertw(tmp, k=0)) + 1)
                l[k] = F[k]/C[k] * (tauk[k] - t[k])

        return l

    def solve(self):
        """
        Uses Algorithm 1 to solve an instance of the problem.
        """
        assert self.problem.feasible_opt(), "Computation load too large, infeasible"

        # Easing notations by unpacking all parameters
        (K, tauk) = (self.problem.K, self.problem.tauk)
        (L, T) = (self.problem.task.L,
                  self.problem.task.T)
        (C, P, F) = (self.problem.comp.C,
                     self.problem.comp.P,
                     self.problem.comp.F)
        (B, N0, gap, h) = (self.problem.comm.B,
                           self.problem.comm.N0,
                           self.problem.comm.gap,
                           self.problem.comm.h)

        a = (K-1) * (T/L)   # \alpha in the paper

        # Bissection search
        mult_l = 0
        mult_h = 100*np.max(C*P + a*gap*N0/np.abs(h)**2 * np.log(2)/B)

        # Maximum number of iteration
        max_iter = 100
        for i in range(max_iter):
            L_l = np.sum(self.optimum(mult_l))
            L_h = np.sum(self.optimum(mult_h))

            mult_m = (mult_l + mult_h)/2
            L_m = np.sum(self.optimum(mult_m))

            if __debug__:
                print("[DEBUG] iteration #{:d}".format(i))
                print("[DEBUG] mult_l = {:.5e}".format(mult_l))
                print("[DEBUG] mult_m = {:.5e}".format(mult_m))
                print("[DEBUG] mult_h = {:.5e}".format(mult_h))

            # Declare convergence if the following condition is satisfied
            if abs(L_m - L)/L <= 1e-7:
                self.msg = "Converged."
                if __debug__:
                    print("[DEBUG] converged.")

                return (self.optimum(mult_m), mult_m)
            else:
                if L_m > L:
                    mult_h = mult_m
                else:
                    mult_l = mult_m

                if __debug__:
                    print("[DEBUG] L_l = {:.5e}".format(L_l))
                    print("[DEBUG] L_m = {:.5e}".format(L_m))
                    print("[DEBUG] L_h = {:.5e}".format(L_h))

            if i == max_iter - 1:
                self.msg = "Stopped."
                if __debug__:
                    print("[DEBUG] didn't converged.")
                    print("[DEBUG] L = {:.3e} while L* = {:.3e}".format(L, L_m))

                return (self.optimum(mult_m), mult_m)

class Solution:
    """
    Helps to pretty-print the solution (i.e., E_MAP, E_RED, E_SHU,
    RF TX power, t_MAP, t_RED, t_SHU, etc) given the set of optimal
    l_k and the value of the Lagrange multiplier \lambda^*.
    """

    def __init__(self, problem, l, lamb):
        self.task = problem.task
        self.comp = problem.comp
        self.comm = problem.comm
        self.lamb = lamb

        # Easing notations by unpacking all parameters
        tauk = problem.tauk
        (D, L, T) = (self.task.D, self.task.L, self.task.T)
        (K, C, P, F) = (self.comp.K, self.comp.C,
                        self.comp.P, self.comp.F)
        (B, N0, gap, h) = (self.comm.B, self.comm.N0,
                           self.comm.gap, self.comm.h)

        a = (K-1) * (T/L)   # \alpha in the paper

        # Optimal l_k for k \in [K]
        self.l = l

        # Shuffle phase
        self.t_shu = np.zeros((K,))
        self.E_shu = np.zeros((K,))
        self.p_shu = np.zeros((K,))

        # Map phase
        self.t_map = np.zeros((K,))
        self.E_map = np.zeros((K,))

        for k in range(K):
            if l[k] == 0:
                # Node k does not participate to the Map and Shuffle phases
                self.t_map[k] = 0
                self.E_map[k] = 0

                self.t_shu[k] = 0
                self.p_shu[k] = 0
                self.E_shu[k] = 0
            else:
                self.t_shu[k] = tauk[k] - C[k]/F[k] * l[k]
                self.p_shu[k] = 1/np.abs(h[k])**2 * \
                gap*N0*(2**(a*l[k]/B/self.t_shu[k]) - 1)
                self.E_shu[k] = self.p_shu[k] * self.t_shu[k]

                self.t_map[k] = (l[k] + K*D)*C[k]/F[k]
                self.E_map[k] = (l[k] + K*D)*C[k]*P[k]

        # Reduce phase
        self.t_red = T*C/F
        self.E_red = T*C*P

        # Energy that would have been used if all the nodes were working
        # alone (i.e., no collaboration)
        self.E_solo = (l + D)*C*P

        # The two terms of the left-hand-side of Eq. (10)
        self.th_comp = C*P
        self.th_comm = a * gap*N0/np.abs(h)**2 * np.log(2)/B

    def pretty_print(self):
        (D, L, T) = (self.task.D, self.task.L, self.task.T)
        (K, C, P, F) = (self.comp.K, self.comp.C,
                        self.comp.P, self.comp.F)
        (B, N0, gap, h) = (self.comm.B, self.comm.N0,
                           self.comm.gap, self.comm.h)

        a = (K-1) * (T/L)   # \alpha in the paper

        print("k \t | F_k/C_k \t | C_kP_k \t | |h_k|^2 \t | Th. \t\t || l_k \t | t_MAP \t | E_MAP \t | t_SHU \t | E_SHU \t | p_k")
        for k in range(K):
            if self.l[k] == 0:
                print("{:d} \t | {:.2e} \t | {:.2e} \t | {:.2e} \t | {:.2e} \t || {:.2e} \t | {:.2e} \t | {:.2e} \t | {:.2e} \t | {:.2e} \t | {:.2e}".format(
                    k, F[k]/C[k], C[k]*P[k], np.abs(h[k])**2,
                    self.th_comp[k] + self.th_comm[k],
                    self.l[k], self.t_map[k],
                    self.E_map[k], self.t_shu[k], self.E_shu[k], self.p_shu[k]))
            else:
                # Color the nodes participating to the Map and Reduce phase in
                # green to help visualize the result
                print(colored("{:d} \t | {:.2e} \t | {:.2e} \t | {:.2e} \t | {:.2e} \t || {:.2e} \t | {:.2e} \t | {:.2e} \t | {:.2e} \t | {:.2e} \t | {:.2e}".format(
                    k, F[k]/C[k], C[k]*P[k], np.abs(h[k])**2,
                    self.th_comp[k] + self.th_comm[k],
                    self.l[k], self.t_map[k],
                    self.E_map[k], self.t_shu[k], self.E_shu[k], self.p_shu[k]), 'green'))

