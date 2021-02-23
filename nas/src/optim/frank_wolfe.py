import numpy as np
import torch
from cvxopt import matrix, solvers
from scipy.linalg import block_diag
from scipy.optimize import linprog
import logging
from torch.optim.optimizer import Optimizer

from external.distributed_manager import DistributedManager

import sys

from nas.nas_utils.general_purpose import EPSILON, expected_latency

np.set_printoptions(threshold=sys.maxsize, suppress=True, precision=6)


class FrankWolfe(Optimizer):
    def __init__(self, params, list_alphas, delta_lp, inference_time_limit, no_expert, lp_method, tol_lp,
                 max_gamma=1, writer=None):
        super(FrankWolfe, self).__init__(params, {})




    def _gamma_over_alpha_is_beta(self, alpha_blocks, beta_blocks, latency_vec, n):
        if self.A_gamma_over_alpha_is_beta is not None and self.b_gamma_over_alpha_is_beta is not None:
            return self.A_gamma_over_alpha_is_beta, self.b_gamma_over_alpha_is_beta

        alphas = np.sum(alpha_blocks)
        assert alphas == len(latency_vec)

        betas = np.sum(beta_blocks)
        assert betas == len(alpha_blocks)

        self.A_gamma_over_alpha_is_beta = None
        self.b_gamma_over_alpha_is_beta = []
        beta_offset = 0
        gamma_offset = 0
        for beta_block_size in beta_blocks:
            for b in range(beta_block_size):
                beta = beta_offset + b
                eq = np.zeros(n)
                eq[alphas + betas + gamma_offset: alphas + betas + gamma_offset + alpha_blocks[beta]] = 1
                eq[alphas + beta] = -1
                self.b_gamma_over_alpha_is_beta.append(0)

                self.A_gamma_over_alpha_is_beta = np.vstack([self.A_gamma_over_alpha_is_beta, eq]) \
                    if self.A_gamma_over_alpha_is_beta is not None else eq

                gamma_offset += alpha_blocks[beta]

            beta_offset += beta_block_size

        self.b_gamma_over_alpha_is_beta = np.array(self.b_gamma_over_alpha_is_beta)

        return self.A_gamma_over_alpha_is_beta, self.b_gamma_over_alpha_is_beta


    def _gamma_binary_alpha_beta(self, alpha_blocks, beta_blocks, latency_vec, n):
        if self.A_gamma_binary_alpha_beta is not None and self.b_gamma_binary_alpha_beta is not None:
            return self.A_gamma_binary_alpha_beta, self.b_gamma_binary_alpha_beta

        alphas = np.sum(alpha_blocks)
        assert alphas == len(latency_vec)

        betas = np.sum(beta_blocks)
        assert betas == len(alpha_blocks)

        A_ub = None
        b_ub = []
        gamma = 0
        alpha_offset = 0
        beta_offset = 0
        for beta_block_size in beta_blocks:
            start_alpha_offset = alpha_offset
            for b in range(beta_block_size):
                beta = beta_offset + b
                alpha_offset += alpha_blocks[beta]
                for a in range(start_alpha_offset, alpha_offset):
                    # alpha + beta - 1 <= gamma
                    lb = np.zeros(n)
                    lb[a] = 1
                    lb[alphas + beta] = 1
                    lb[alphas + betas + gamma] = -1
                    b_ub.append(1)

                    # gamma <= alpha
                    ub_a = np.zeros(n)
                    ub_a[a] = -1
                    ub_a[alphas + betas + gamma] = 1
                    b_ub.append(0)

                    # gamma <= beta
                    ub_b = np.zeros(n)
                    ub_b[alphas + beta] = -1
                    ub_b[alphas + betas + gamma] = 1
                    b_ub.append(0)

                    A_ub = np.vstack([A_ub, lb, ub_a, ub_b]) if A_ub is not None else np.vstack([lb, ub_a, ub_b])
                    gamma += 1

            beta_offset += beta_block_size

        b_ub = np.array(b_ub)

        self.A_gamma_binary_alpha_beta = A_ub
        self.b_gamma_binary_alpha_beta = b_ub

        return A_ub, b_ub


    def _bounds_0_1_matrices(self, n):
        if self.A_bounds is not None and self.b_bounds is not None:
            return self.A_bounds, self.b_bounds
        # Bounds as inequalities
        A_lb = -np.eye(n)
        b_lb = np.zeros(n)

        A_ub = np.eye(n)
        b_ub = np.ones(n)

        self.A_bounds = np.concatenate((A_lb, A_ub), axis=0)
        self.b_bounds = np.concatenate((b_lb, b_ub), axis=0)

        return self.A_bounds, self.b_bounds


    def _relaxed_lp_constraint_matrices(self, alpha_blocks, beta_blocks, latency_vec):
        # Inequality constraints
        A_latency, b_latency = self._gamma_latency_matrix(alpha_blocks, beta_blocks, latency_vec)
        n = A_latency.shape[1]
        A_gab, b_gab = self._gamma_binary_alpha_beta(alpha_blocks, beta_blocks, latency_vec, n)
        A_ab_bonuds, b_ab_bounds = self._bounds_0_1_matrices(n)

        # Equality constraints
        A_ab_simplex, b_ab_simplex = self._simplex_eq_constraint(alpha_blocks, beta_blocks, n)
        A_gamma_over_alpha_is_beta, b_gamma_over_alpha_is_beta = \
            self._gamma_over_alpha_is_beta(alpha_blocks, beta_blocks, latency_vec, n)

        G = matrix(np.concatenate((A_latency, A_gab, A_ab_bonuds)), tc='d')
        h = matrix(np.concatenate((b_latency, b_gab, b_ab_bounds)), tc='d')
        A = matrix(np.concatenate((A_ab_simplex, A_gamma_over_alpha_is_beta)), tc='d')
        b = matrix(np.concatenate((b_ab_simplex, b_gamma_over_alpha_is_beta)), tc='d')

        return G, h, A, b, n


    def _gamma_latency_matrix(self, alpha_blocks, beta_blocks, latency_vec):
        if self.A_gamma_latency is not None and self.b_gamma_latency is not None:
            return self.A_gamma_latency, self.b_gamma_latency

        alphas = np.sum(alpha_blocks)
        assert alphas == len(latency_vec)

        betas = np.sum(beta_blocks)
        assert betas == len(alpha_blocks)

        self.A_gamma_latency = np.zeros(alphas + betas)

        alpha_offset = 0
        beta_offset = 0
        for beta_block_size in beta_blocks:
            start_alpha_offset = alpha_offset
            for b in range(beta_block_size):
                beta = beta_offset + b
                alpha_offset += alpha_blocks[beta]
                self.A_gamma_latency = np.concatenate((self.A_gamma_latency,
                                                       latency_vec[start_alpha_offset: alpha_offset]))

            beta_offset += beta_block_size

        self.A_gamma_latency = np.asmatrix(self.A_gamma_latency)
        self.b_gamma_latency = np.array([self.T])

        return self.A_gamma_latency, self.b_gamma_latency


    def _simplex_eq_constraint(self, alpha_blocks, beta_blocks, n=None):
        if self.A_eq is not None and self.b_eq is not None:
            return self.A_eq, self.b_eq
        alpha_beta_blocks = alpha_blocks + beta_blocks
        rows = len(alpha_beta_blocks)
        cols = n if n is not None else np.sum(alpha_beta_blocks)
        self.A_eq = np.zeros((rows, cols))

        self.q = np.zeros(cols)
        offset = 0
        for r, block_size in enumerate(alpha_beta_blocks):
            # print(r)
            # print(range(offset, (offset + block_size)))
            # print(A_eq[r, offset: (offset + block_size)].shape)
            self.A_eq[r, offset: (offset + block_size)] = 1
            self.q[offset: (offset + block_size)] = 1./block_size
            offset += block_size

        self.b_eq = np.ones(rows)

        return self.A_eq, self.b_eq


    def _block_uniform_vector(self, alpha_blocks, beta_blocks, n=None):
        if self.q is not None:
            return self.q

        alpha_beta_blocks = alpha_blocks + beta_blocks

        n = np.sum(alpha_beta_blocks) if n is None else n
        self.q = np.zeros(n)

        offset = 0
        for block_size in alpha_beta_blocks:
            self.q[offset: (offset + block_size)] = 1./block_size
            offset += block_size

        return self.q


    def _qp_init(self):
        # Flatten all the layers attentions, measured latencies and gradients as corresponding column stack vectors
        # attention_vec, latency_vec, _ = self._flatten_attention_latency_grad()
        alpha_attention_vec, latency_vec, _, alpha_blocks, beta_attention_vec, _, beta_blocks = \
            self._flatten_attention_latency_grad_alpha_beta_blocks()

        print('Before')
        print('alpha_attention_vec')
        print(np.reshape(alpha_attention_vec, (-1, 20)))
        print('beta_attention_vec')
        print(np.reshape(beta_attention_vec, (-1, 4)))

        alphas, betas = len(alpha_attention_vec), len(beta_attention_vec)

        G, h, A, b, n = self._relaxed_lp_constraint_matrices(alpha_blocks, beta_blocks, latency_vec)

        q = self._block_uniform_vector(alpha_blocks, beta_blocks, n)

        p = matrix(2 * block_diag(np.eye(alphas + betas), np.eye(n - alphas - betas)), tc='d')
        q = matrix(-2 * q, tc='d')

        sol = solvers.qp(P, q, G, h, A, b)
        attention_vec = np.array(sol['x'])

        alpha_attention_vec = attention_vec[:alphas]
        beta_attention_vec = attention_vec[alphas:alphas + betas]

        # TODO: Solve the negative gammas
        # gamma_attention_vec = attention_vec[alphas + betas:]

        print('Intermediate')
        print('alpha_attention_vec')
        print(np.reshape(alpha_attention_vec, (-1, 20)))
        print('beta_attention_vec')
        print(np.reshape(beta_attention_vec, (-1, 4)))
        # print('gamma_attention_vec')
        # print(gamma_attention_vec.T)

        # assert np.all(attention_vec >= 0)
        # assert np.all(attention_vec <= 1)
        # assert G.dot(attention_vec) <= h
        # assert np.allclose(A_eq.dot(attention_vec), b_eq)

        # Update
        self._update_attentions_inplace(alpha_attention_vec, beta_attention_vec)

        alpha_attention_vec, latency_vec, _, alpha_blocks, beta_attention_vec, _, beta_blocks = \
            self._flatten_attention_latency_grad_alpha_beta_blocks()

        print('After')
        print('alpha_attention_vec')
        print(np.reshape(alpha_attention_vec, (-1, 20)))
        print('beta_attention_vec')
        print(np.reshape(beta_attention_vec, (-1, 4)))

        # Latency test
        updated_abg = self._init_abg_vec(alpha_blocks, beta_blocks, alpha_attention_vec, beta_attention_vec, n)
        A_latency, b_latency = self._gamma_latency_matrix(alpha_blocks, beta_blocks, latency_vec)
        constrained_latency = (A_latency @ attention_vec).item()
        actual_expected_latency = expected_latency(self.list_alphas)
        print('------------------------------- Check solution ------------------------------- ')
        self._compare_abg(alpha_blocks, beta_blocks, attention_vec)
        print('------------------------------- Check Derived ------------------------------- ')
        self._compare_abg(alpha_blocks, beta_blocks, updated_abg)
        print('============================================================================= ')

        print('Updated actual expected latency: {}'.format(actual_expected_latency))
        print('Derived gamma from updated alpha and beta: {}'.format((A_latency @ updated_abg).item()))
        print('Latency constraint: {} <= {}'.format(constrained_latency, b_latency.item()))
        assert actual_expected_latency <= b_latency.item()
        assert np.allclose(constrained_latency, actual_expected_latency)


    def _gamma_binary_bounds(self, alpha_blocks, beta_blocks, alpha_attention_vec, beta_attention_vec):
        alphas = np.sum(alpha_blocks)
        assert alphas == len(alpha_attention_vec)

        betas = len(beta_attention_vec)
        assert betas == len(alpha_blocks) and betas == np.sum(beta_blocks)

        gamma_bounds = []
        alpha_offset = 0
        beta_offset = 0
        for beta_block_size in beta_blocks:
            start_alpha_offset = alpha_offset
            for b in range(beta_block_size):
                beta = beta_offset + b
                alpha_offset += alpha_blocks[beta]
                for a in range(start_alpha_offset, alpha_offset):
                    lb = alpha_attention_vec[a] + beta_attention_vec[beta] - 1
                    ub = np.min([alpha_attention_vec[a], beta_attention_vec[beta]])
                    gamma_bounds.append((lb, ub))

            beta_offset += beta_block_size

        return gamma_bounds


    def _init_abg_vec(self, alpha_blocks, beta_blocks, alpha_attention_vec, beta_attention_vec, n):
        alphas = np.sum(alpha_blocks)
        assert alphas == len(alpha_attention_vec)

        betas = len(beta_attention_vec)
        assert betas == len(alpha_blocks) and betas == np.sum(beta_blocks)

        x0 = np.zeros(n)
        x0[:alphas] = alpha_attention_vec
        x0[alphas:alphas + betas] = beta_attention_vec
        gamma = 0
        alpha_offset = 0
        beta_offset = 0
        for beta_block_size in beta_blocks:
            start_alpha_offset = alpha_offset
            for b in range(beta_block_size):
                beta = beta_offset + b
                alpha_offset += alpha_blocks[beta]
                for a in range(start_alpha_offset, alpha_offset):
                    x0[alphas + betas + gamma] = alpha_attention_vec[a] * beta_attention_vec[beta]
                    gamma += 1

            beta_offset += beta_block_size

        return x0

    def _compare_abg(self, alpha_blocks, beta_blocks, attention_vec):
        alphas = np.sum(alpha_blocks)
        betas = np.sum(beta_blocks)
        assert betas == len(alpha_blocks)

        gamma = 0
        alpha_offset = 0
        beta_offset = 0
        for stage, beta_block_size in enumerate(beta_blocks):
            start_alpha_offset = alpha_offset
            for b in range(beta_block_size):
                beta = beta_offset + b
                alpha_offset += alpha_blocks[beta]
                for a in range(start_alpha_offset, alpha_offset):
                    alpha_value = attention_vec[a].item()
                    beta_value = attention_vec[alphas + beta].item()
                    gamma_value = attention_vec[alphas + betas + gamma].item()

                    print('s={}, k={}, i={}, j={} | alpha={:.2f}, beta={:.2f}, alpha * beta={:.5f}, gamma={:.5f}'
                          .format(stage, (a - start_alpha_offset) % alpha_blocks[beta],
                                  int((a - start_alpha_offset) / alpha_blocks[beta]), b,
                                  alpha_value, beta_value, alpha_value * beta_value, gamma_value))

                    gamma += 1

            beta_offset += beta_block_size


    def step(self):
        # Flatten all the layers attentions, measured latencies and gradients as corresponding column stack vectors
        alpha_attention_vec, latency_vec, alpha_grad_vec, alpha_blocks, beta_attention_vec, beta_grad_vec, beta_blocks = \
            self._flatten_attention_latency_grad_alpha_beta_blocks()

        alphas, betas = len(alpha_attention_vec), len(beta_attention_vec)

        A_ub, b_ub, A_eq, b_eq, n = self._relaxed_lp_constraint_matrices(alpha_blocks, beta_blocks, latency_vec)

        c = np.zeros(n)
        c[:alphas] = alpha_grad_vec
        c[alphas:alphas + betas] = beta_grad_vec
        print('alpha_grads')
        print(np.reshape(c[:alphas], (-1, 20)))
        print('beta_grads')
        print(np.reshape(c[alphas:alphas + betas], (-1, 4)))

        # c[:alphas] = np.random.uniform(0, 1, size=alphas)
        # c[alphas:alphas + betas] = np.random.uniform(0, 1, size=betas)
        # c[:alphas] = -np.arange(alphas)
        # c[alphas:alphas + betas] = -np.arange(betas)

        # LP Solver
        print('Started solving')
        c = matrix(c, tc='d')
        sol = solvers.lp(c, A_ub, b_ub, A_eq, b_eq)
        print(sol['status'])
        x = np.array(sol['x'])

        # Init point
        # x0 = self._init_abg_vec(alpha_blocks, beta_blocks, alpha_attention_vec, beta_attention_vec, n)
        # assert np.all(A_ub @ x0 <= b_ub)
        # assert np.allclose(A_eq @ x0, b_eq)

        # res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
        #               #bounds=bounds,
        #               x0=x0,
        #               method=self.lp_method, options={'tol': self.tol_lp, 'maxiter': 1000})
        #             # method=self.lp_method)
        # res = linprog(c=grad_vec, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
        # if not res.success:
        #     raise Exception(f"Process on GPU: {DistributedManager.get_rank_()} raised the following error: "
        #                     + res.message)
        # x = res.x
        print('Finished solving')

        assert np.all(A_ub @ x <= b_ub + EPSILON)
        assert np.allclose(A_eq @ x, b_eq)

        alpha = x[:alphas].squeeze()
        beta = x[alphas: alphas + betas].squeeze()

        print('solution:')
        print('alpha_attention_vec')
        print(np.reshape(alpha, (-1, 20)))
        print('beta_attention_vec')
        print(np.reshape(beta, (-1, 4)))
        # print('gamma_attention_vec')
        # print(x[alphas + betas + 1:])

        # Update
        gamma_step = self._calculate_step_size()
        alpha_attention_vec += gamma_step * alpha
        beta_attention_vec += gamma_step * beta

        # Update
        self._update_attentions_inplace(alpha_attention_vec, beta_attention_vec)

        alpha_attention_vec, latency_vec, _, alpha_blocks, beta_attention_vec, _, beta_blocks = \
            self._flatten_attention_latency_grad_alpha_beta_blocks()

        # print('stepped:')
        # print('alpha_attention_vec')
        # print(np.reshape(alpha_attention_vec, (-1, 20)))
        # print('beta_attention_vec')
        # print(np.reshape(beta_attention_vec, (-1, 4)))

        # Latency test
        updated_abg = self._init_abg_vec(alpha_blocks, beta_blocks, alpha_attention_vec, beta_attention_vec, n)
        A_latency, b_latency = self._gamma_latency_matrix(alpha_blocks, beta_blocks, latency_vec)
        constrained_latency = (A_latency @ x).item()
        actual_expected_latency = expected_latency(self.list_alphas)
        print('------------------------------- Check solution ------------------------------- ')
        self._compare_abg(alpha_blocks, beta_blocks, x)
        # print('------------------------------- Check Derived ------------------------------- ')
        # self._compare_abg(alpha_blocks, beta_blocks, updated_abg)
        print('============================================================================= ')

        print('Updated actual expected latency: {}'.format(actual_expected_latency))
        print('Derived gamma from updated alpha and beta: {}'.format((A_latency @ updated_abg).item()))
        print('Latency constraint: {} <= {}'.format(constrained_latency, b_latency.item()))
        assert actual_expected_latency <= b_latency.item()
        assert np.allclose(constrained_latency, actual_expected_latency)


