import logging
import sys

from cvxopt import matrix, solvers
from torch.optim.optimizer import Optimizer
from tqdm import tqdm

from nas.nas_utils.general_purpose import update_alpha_beta_tensorboard
from nas.src.optim.utils import *

np.set_printoptions(threshold=sys.maxsize, suppress=True, precision=11)


class BlockFrankWolfe(Optimizer):
    def __init__(self, params, list_alphas, inference_time_limit, max_gamma,
                 min_depth=1, alternate=False, fixed_grads=False, one_gamma=False, momentum=1, start_with_alpha=False):
        super(BlockFrankWolfe, self).__init__(params, {})

        self.list_alphas = list_alphas
        self.T = inference_time_limit
        self.k = 0

        self.q = None
        self.A_bounds = None
        self.b_bounds = None
        self.A_eq = None
        self.b_eq = None

        self.writer = None
        self._max_gamma = min(max_gamma, 1)

        self.Q = None
        self.p = None
        self.min_depth = min_depth
        self._epoch = None
        self._same_alpha = False
        self._same_beta = False
        self._done = False
        self._alternate = alternate
        self._fixed_grads = fixed_grads
        self._one_gamma = one_gamma
        self.only_alpha = False
        self.only_beta = False
        self.qp_init_steps = 100
        self.sparsify_steps = 100
        self._fixed_latency = 0
        self._momentum = momentum
        self._alpha_grad_buf = None
        self._beta_grad_buf = None
        self.start_with_alpha = start_with_alpha


    @property
    def fixed_latency(self):
        return self._fixed_latency


    @fixed_latency.setter
    def fixed_latency(self, fixed_latency):
        self._fixed_latency = fixed_latency


    def set_epoch(self, epoch=None):
        self._epoch = epoch


    def is_done(self):
        return self._done


    def _bounds_0_1_matrices(self, n):
        # Bounds as inequalities
        A_lb = -np.eye(n)
        b_lb = np.zeros(n)

        A_ub = np.eye(n)
        b_ub = np.ones(n)

        self.A_bounds = np.concatenate((A_lb, A_ub), axis=0)
        self.b_bounds = np.concatenate((b_lb, b_ub), axis=0)

        return self.A_bounds, self.b_bounds


    def _alpha_beta_latency_matrix(self, alpha_blocks, beta_blocks, latency_vec):
        if self.Q is not None and self.p is not None:
            return self.Q, self.p

        # for i in range(1, len(latency_vec), 2):
        #     print('without se {} <- with se {}'.format(latency_vec[i], latency_vec[i-1]))
        #     latency_vec[i] = latency_vec[i-1]

        alphas = np.sum(alpha_blocks)
        assert alphas == len(latency_vec)

        betas = np.sum(beta_blocks)
        assert betas == len(alpha_blocks) - len(beta_blocks) * self.min_depth

        self.Q = np.zeros((alphas, betas))
        self.p = np.zeros(alphas)

        alpha_offset = 0
        beta_offset = 0
        for beta_block, beta_block_size in enumerate(beta_blocks):
            start_alpha_offset = alpha_offset
            alpha_offset += alpha_blocks[beta_offset + self.min_depth * beta_block]
            self.p[start_alpha_offset: alpha_offset] = latency_vec[start_alpha_offset: alpha_offset]

            start_alpha_offset = alpha_offset
            for b in range(beta_block_size):
                beta = beta_offset + b
                alpha_offset += alpha_blocks[beta + self.min_depth * (beta_block + 1)]
                self.Q[start_alpha_offset: alpha_offset, beta] = latency_vec[start_alpha_offset: alpha_offset]

            beta_offset += beta_block_size

        # print('------------------ p --------------')
        # print(np.reshape(self.p, (-1, alpha_blocks[0] * (1 + beta_blocks[0]))))
        # print('-----------------------------------')
        #
        # print('------------------ Q --------------')
        # print(np.reshape(self.Q.T, (-1, alpha_blocks[0] * (1 + beta_blocks[0]))))
        # print('-----------------------------------')

        return self.Q, self.p


    def _latency_constraint(self, alpha_blocks, beta_blocks, latency_vec, alpha_vec=None, beta_vec=None):
        assert alpha_vec is not None or beta_vec is not None

        Q, p = self._alpha_beta_latency_matrix(alpha_blocks, beta_blocks, latency_vec)

        if alpha_vec is not None:
            A_latency = alpha_vec @ Q
            b_latency = np.array([self.T - self._fixed_latency - alpha_vec @ p])
        else:
            A_latency = beta_vec @ Q.T + p
            b_latency = np.array([self.T - self._fixed_latency])

        return np.asmatrix(A_latency), b_latency


    def _simplex_eq_constraint(self, blocks, n=None):
        assert n is None or n == np.sum(blocks)

        rows = len(blocks)
        cols = n if n is not None else np.sum(blocks)
        self.A_eq = np.zeros((rows, cols))

        self.q = np.zeros(cols)
        offset = 0
        for r, block_size in enumerate(blocks):
            self.A_eq[r, offset: (offset + block_size)] = 1
            self.q[offset: (offset + block_size)] = 1. / block_size
            offset += block_size

        self.b_eq = np.ones(rows)

        return self.A_eq, self.b_eq


    def _close_consecutive(self, blocks, n=None):
        assert n is None or n == np.sum(blocks)

        rows = np.sum([block - 1 for block in blocks])
        cols = n if n is not None else np.sum(blocks)
        U = np.zeros((rows, cols))

        r = 0
        offset = 0
        for block_size in blocks:
            for c in range(block_size - 1):
                U[r, offset + c] = 1
                U[r, offset + c + 1] = -1
                r += 1

            offset += block_size

        P = U.T @ U
        return P


    def alpha_qp_step(self, alpha_attention_vec, alpha_blocks, latency_vec, beta_attention_vec, beta_blocks):
        alphas = len(alpha_attention_vec)

        A_latency, b_latency = self._latency_constraint(alpha_blocks, beta_blocks, latency_vec,
                                                        alpha_vec=None, beta_vec=beta_attention_vec)
        A_bounds, b_bounds = self._bounds_0_1_matrices(alphas)
        A_simplex, b_simplex = self._simplex_eq_constraint(alpha_blocks, alphas)

        P = 2 * self._close_consecutive(alpha_blocks, alphas)
        q = np.zeros(alphas)
        # P = 2 * block_diag(np.eye(alphas))
        # q = -2 * self._block_uniform_vector(alpha_blocks, beta_blocks)[:alphas]
        # def F(x=None, z=None):
        #     if x is None: return 0, matrix(1.0, (alphas, 1))
        #     if min(x) <= 0: return None
        #     f = x.T * log(x)
        #     grad = 1.0 + log(x)
        #     if z is None: return f, grad.T
        #     H = spdiag(z[0] * x ** -1)
        #     return f, grad.T, H

        A_ub = matrix(np.concatenate((A_latency, A_bounds)), tc='d')
        b_ub = matrix(np.concatenate((b_latency, b_bounds)), tc='d')
        A_eq = matrix(A_simplex, tc='d')
        b_eq = matrix(b_simplex, tc='d')
        P = matrix(P, tc='d')
        q = matrix(q, tc='d')

        # sol = solvers.cp(F, G=A_ub, h=b_ub, A=A_eq, b=b_eq, options=dict(show_progress=False))
        sol = solvers.qp(P, q, A_ub, b_ub, A_eq, b_eq, options=dict(show_progress=False))
        if sol['status'] == 'primal infeasible':
            print('qp init primal infeasible in alpha')
            return alpha_attention_vec

        x = np.array(sol['x'])
        alpha = x.squeeze().clip(0, 1)

        # print('solution:')
        # print('lp solution: alpha_attention_vec')
        # print(np.reshape(alpha, (len(alpha_blocks), -1)))

        # Early stop in stagnation mechanism
        if np.allclose(alpha_attention_vec, alpha, atol=1e-3):
            self.k += 1
            # print('alpha same')
            self._same_alpha = True
            if self._same_beta:
                # print('alpha done')
                self._done = True

            return alpha_attention_vec

        self._same_alpha = False
        self._same_beta = False

        # Update
        gamma_step = self._calculate_step_size(n=2, one=self._one_gamma)
        alpha_attention_vec += gamma_step * (alpha - alpha_attention_vec)

        return alpha_attention_vec


    def beta_qp_step(self, alpha_attention_vec, alpha_blocks, latency_vec, beta_attention_vec, beta_blocks):
        betas = len(beta_attention_vec)

        A_latency, b_latency = self._latency_constraint(alpha_blocks, beta_blocks, latency_vec, alpha_attention_vec)
        A_bounds, b_bounds = self._bounds_0_1_matrices(betas)
        A_simplex, b_simplex = self._simplex_eq_constraint(beta_blocks, betas)
        # print('beta simplex constraints')
        # print(A_simplex)
        # print(b_simplex)
        # print('---------')

        P = 2 * self._close_consecutive(beta_blocks, betas)
        q = np.zeros(betas)
        # P = 2 * block_diag(np.eye(betas))
        # q = -2 * self._block_uniform_vector(alpha_blocks, beta_blocks)[-betas:]
        # def F(x=None, z=None):
        #     if x is None: return 0, matrix(1.0, (betas, 1))
        #     if min(x) <= 0: return None
        #     f = x.T * log(x)
        #     grad = 1.0 + log(x)
        #     if z is None: return f, grad.T
        #     H = spdiag(z[0] * x ** -1)
        #     return f, grad.T, H

        A_ub = matrix(np.concatenate((A_latency, A_bounds)), tc='d')
        b_ub = matrix(np.concatenate((b_latency, b_bounds)), tc='d')
        A_eq = matrix(A_simplex, tc='d')
        b_eq = matrix(b_simplex, tc='d')

        P = matrix(P, tc='d')
        q = matrix(q, tc='d')

        # sol = solvers.cp(F, G=A_ub, h=b_ub, A=A_eq, b=b_eq, options=dict(show_progress=False))
        sol = solvers.qp(P, q, A_ub, b_ub, A_eq, b_eq, options=dict(show_progress=False))
        if sol['status'] == 'primal infeasible':
            print('qp init primal infeasible in beta')
            return alpha_attention_vec

        x = np.array(sol['x'])
        # assert np.all(A_ub @ x <= b_ub + self.EPSILON)
        # assert np.allclose(A_eq @ x, b_eq)
        beta = x.squeeze().clip(0, 1)
        # print('Current: beta_attention_vec')
        # print(np.reshape(beta_attention_vec, (-1, 3)))

        # print('lp solution: beta_attention_vec')
        # print(np.reshape(beta, (len(beta_blocks), -1)))
        # print('A_eq @ beta_attention')
        # print(A_eq @ beta)

        # Early stop in stagnation mechanism
        if np.allclose(beta_attention_vec, beta, atol=1e-3):
            self.k += 1
            # print('beta same')
            self._same_beta = True
            if self._same_alpha:
                # print('beta done')
                self._done = True

            return beta_attention_vec

        self._same_alpha = False
        self._same_beta = False

        # Update
        gamma_step = self._calculate_step_size(n=2, one=self._one_gamma)
        beta_attention_vec += gamma_step * (beta - beta_attention_vec)

        return beta_attention_vec


    def reset_state(self):
        self._same_alpha = False
        self._same_beta = False
        self._done = False


    def bc_qp_init(self):
        self.reset_state()
        alpha_attention_vec, latency_vec, _, alpha_blocks, beta_attention_vec, _, beta_blocks = \
            flatten_attention_latency_grad_alpha_beta_blocks(self.list_alphas)

        bar = tqdm(range(self.qp_init_steps)) if DistributedManager.is_master() else range(self.qp_init_steps)

        max_gamma = self._max_gamma
        self.set_max_gamma_step(1)
        one_gamma = self._one_gamma
        self._one_gamma = True

        for k in bar:
            if k % 2 == 1 - int(self.start_with_alpha):
                alpha_attention_vec = self.alpha_qp_step(alpha_attention_vec, alpha_blocks, latency_vec,
                                                         beta_attention_vec, beta_blocks)
            else:
                beta_attention_vec = self.beta_qp_step(alpha_attention_vec, alpha_blocks, latency_vec,
                                                       beta_attention_vec, beta_blocks)

            if self._done:
                break

        update_attentions_inplace(self.list_alphas,
                                  alpha_attention_vec=alpha_attention_vec, beta_attention_vec=beta_attention_vec)
        print('Balanced Init Alpha Attention')
        print(np.reshape(alpha_attention_vec, (len(alpha_blocks), -1)))
        print('Balanced Init Beta Attention')
        print(np.reshape(beta_attention_vec, (len(beta_blocks), -1)))

        self.set_max_gamma_step(max_gamma)
        self._one_gamma = one_gamma


    def sparsify(self, alpha_negative_importance=None, beta_negative_importance=None):
        self.reset_state()

        alpha_attention_vec, latency_vec, _, alpha_blocks, beta_attention_vec, _, beta_blocks = \
            flatten_attention_latency_grad_alpha_beta_blocks(self.list_alphas)

        if alpha_negative_importance is None:
            alpha_negative_importance = -alpha_attention_vec.copy()

        if beta_negative_importance is None:
            beta_negative_importance = -beta_attention_vec.copy()

        max_gamma = self._max_gamma
        self.set_max_gamma_step(1)
        one_gamma = self._one_gamma
        self._one_gamma = True
        self.reset_gamma_step()
        bar = tqdm(range(self.sparsify_steps)) if DistributedManager.is_master() else range(self.sparsify_steps)

        for k in bar:
            if k % 2 == 1 - int(self.start_with_alpha):
                alpha_attention_vec = self.alpha_lp(alpha_attention_vec, alpha_blocks, latency_vec,
                                                    alpha_negative_importance, beta_attention_vec, beta_blocks)
            else:
                beta_attention_vec = self.beta_lp(alpha_attention_vec, latency_vec, alpha_blocks,
                                                  beta_attention_vec, beta_negative_importance, beta_blocks)

            if self._done:
                break

        update_attentions_inplace(self.list_alphas,
                                  alpha_attention_vec=alpha_attention_vec, beta_attention_vec=beta_attention_vec)
        print('Sparsified Alpha Attention')
        print(np.reshape(alpha_attention_vec, (len(alpha_blocks), -1)))
        print('Sparsified Beta Attention')
        print(np.reshape(beta_attention_vec, (len(beta_blocks), -1)))

        self.set_max_gamma_step(max_gamma)
        self._one_gamma = one_gamma


    def alpha_lp(self, alpha_attention_vec, alpha_blocks, latency_vec, alpha_grad_vec, beta_attention_vec, beta_blocks):
        alphas = len(alpha_attention_vec)

        A_latency, b_latency = self._latency_constraint(alpha_blocks, beta_blocks, latency_vec,
                                                        alpha_vec=None, beta_vec=beta_attention_vec)
        A_bounds, b_bounds = self._bounds_0_1_matrices(alphas)
        A_simplex, b_simplex = self._simplex_eq_constraint(alpha_blocks, alphas)

        A_ub = matrix(np.concatenate((A_latency, A_bounds)), tc='d')
        b_ub = matrix(np.concatenate((b_latency, b_bounds)), tc='d')
        A_eq = matrix(A_simplex, tc='d')
        b_eq = matrix(b_simplex, tc='d')

        if self._momentum < 1:
            self._alpha_grad_buf = alpha_grad_vec if self._alpha_grad_buf is None \
                else self._momentum * self._alpha_grad_buf + (1 - self._momentum) * alpha_grad_vec

        c = alpha_grad_vec if self._alpha_grad_buf is None else self._alpha_grad_buf

        # print('alpha_grads')
        # print(np.reshape(c, (-1, 12)))
        # c = np.random.uniform(0, 1, size=alphas)
        # c = -np.arange(alphas)
        c = matrix(c, tc='d')

        # LP Solver
        try:
            sol = solvers.lp(c, A_ub, b_ub, A_eq, b_eq, options=dict(show_progress=False))
        except:
            logging.info(f"{DistributedManager.get_rank_()}:")
            logging.info(f'c: {c}')
            logging.info(f'A_ub: {A_ub}')
            logging.info(f'b_ub: {b_ub}')
            logging.info(f'A_eq: {A_eq}')
            logging.info(f'b_eq: {b_eq}')
            return False

        # print(sol['status'])
        if sol['status'] == 'primal infeasible':
            return False

        x = np.array(sol['x'])
        # assert np.all(A_ub @ x <= b_ub + self.EPSILON)
        # assert np.allclose(A_eq @ x, b_eq)

        alpha = x.squeeze().clip(0, 1)

        # print('solution:')
        # print('lp solution: alpha_attention_vec')
        # print(np.reshape(alpha, (len(alpha_blocks), -1)))

        # Early stop in stagnation mechanism
        if self._fixed_grads and np.allclose(alpha_attention_vec, alpha, atol=1e-3):
            self.k += 1
            # print('alpha same')
            self._same_alpha = True
            if self._same_beta:
                # print('alpha done')
                self._done = True

            return True

        self._same_alpha = False
        self._same_beta = False

        # Update
        gamma_step = self._calculate_step_size(n=2, one=self._one_gamma)
        alpha_attention_vec += gamma_step * (alpha - alpha_attention_vec)

        return alpha_attention_vec


    def beta_lp(self, alpha_attention_vec, latency_vec, alpha_blocks, beta_attention_vec, beta_grad_vec, beta_blocks):
        betas = len(beta_attention_vec)

        A_latency, b_latency = self._latency_constraint(alpha_blocks, beta_blocks, latency_vec, alpha_attention_vec)
        A_bounds, b_bounds = self._bounds_0_1_matrices(betas)
        A_simplex, b_simplex = self._simplex_eq_constraint(beta_blocks, betas)
        # print('beta simplex constraints')
        # print(A_simplex)
        # print(b_simplex)
        # print('---------')

        A_ub = matrix(np.concatenate((A_latency, A_bounds)), tc='d')
        b_ub = matrix(np.concatenate((b_latency, b_bounds)), tc='d')
        A_eq = matrix(A_simplex, tc='d')
        b_eq = matrix(b_simplex, tc='d')

        if self._momentum < 1:
            self._beta_grad_buf = beta_grad_vec if self._beta_grad_buf is None \
                else self._momentum * self._beta_grad_buf + (1 - self._momentum) * beta_grad_vec

        c = beta_grad_vec if self._beta_grad_buf is None else self._beta_grad_buf

        # c = np.random.uniform(0, 1, size=betas)
        # c = -np.arange(betas)
        # print('beta_grads')
        # print(np.reshape(c, (-1, 3)))
        c = matrix(c, tc='d')

        # LP Solver
        try:
            sol = solvers.lp(c, A_ub, b_ub, A_eq, b_eq, options=dict(show_progress=False))
        except:
            logging.info(f"{DistributedManager.get_rank_()}:")
            logging.info(f'c: {c}')
            logging.info(f'A_ub: {A_ub}')
            logging.info(f'b_ub: {b_ub}')
            logging.info(f'A_eq: {A_eq}')
            logging.info(f'b_eq: {b_eq}')
            return False

        # print(sol['status'])
        if sol['status'] == 'primal infeasible':
            return False

        x = np.array(sol['x'])
        # assert np.all(A_ub @ x <= b_ub + self.EPSILON)
        # assert np.allclose(A_eq @ x, b_eq)
        beta = x.squeeze().clip(0, 1)
        # print('Current: beta_attention_vec')
        # print(np.reshape(beta_attention_vec, (-1, 3)))

        # print('lp solution: beta_attention_vec')
        # print(np.reshape(beta, (len(beta_blocks), -1)))
        # print('A_eq @ beta_attention')
        # print(A_eq @ beta)

        # Early stop in stagnation mechanism
        if self._fixed_grads and np.allclose(beta_attention_vec, beta, atol=1e-3):
            self.k += 1
            # print('beta same')
            self._same_beta = True
            if self._same_alpha:
                # print('beta done')
                self._done = True

            return True

        # Update
        gamma_step = self._calculate_step_size(n=2, one=self._one_gamma)
        beta_attention_vec += gamma_step * (beta - beta_attention_vec)

        # print('stepped:')
        # print('beta_attention_vec')
        # print(np.reshape(beta_attention_vec, (-1, 3)))
        return beta_attention_vec


    def alpha_step(self):
        # Flatten all the layers attentions, measured latencies and gradients as corresponding column stack vectors
        alpha_attention_vec, latency_vec, alpha_grad_vec, alpha_blocks, beta_attention_vec, _, beta_blocks = \
            flatten_attention_latency_grad_alpha_beta_blocks(self.list_alphas)

        alpha_attention_vec = self.alpha_lp(alpha_attention_vec, alpha_blocks, latency_vec, alpha_grad_vec,
                                            beta_attention_vec, beta_blocks)

        update_attentions_inplace(self.list_alphas, alpha_attention_vec)

        # alpha_attention_vec, latency_vec, _, alpha_blocks, beta_attention_vec, _, beta_blocks = \
        #     flatten_attention_latency_grad_alpha_beta_blocks(self.list_alphas)
        # print('stepped:')
        # print('alpha_attention_vec')
        # print(np.reshape(alpha_attention_vec, (-1, 12)))

        # Latency test
        # constrained_latency = (A_latency @ x).item()
        # print('Latency constraint: {} <= {}'.format(constrained_latency, b_latency.item()))
        # stepped_latency = (A_latency @ alpha_attention_vec).item()
        # print('Stepped latency: {}'.format(stepped_latency))
        # actual_expected_latency = expected_latency(self.list_alphas).item()
        # print('Updated actual expected latency: {}'.format(actual_expected_latency))
        # assert actual_expected_latency <= b_latency.item() + self.EPSILON
        # assert np.allclose(stepped_latency, actual_expected_latency)

        if self._epoch is not None:
            latency = self.latency_formula(alpha_attention_vec, beta_attention_vec, fixed_latency=self._fixed_latency)
            latency = {'latency_formula': latency, 'constraint': self.T}
            update_alpha_beta_tensorboard(self._epoch + 1, self.list_alphas, self.writer, latency)

        return True


    def beta_step(self):
        # Flatten all the layers attentions, measured latencies and gradients as corresponding column stack vectors
        alpha_attention_vec, latency_vec, _, alpha_blocks, beta_attention_vec, beta_grad_vec, beta_blocks = \
            flatten_attention_latency_grad_alpha_beta_blocks(self.list_alphas)

        beta_attention_vec = self.beta_lp(alpha_attention_vec, latency_vec, alpha_blocks,
                                          beta_attention_vec, beta_grad_vec, beta_blocks)

        update_attentions_inplace(self.list_alphas, alpha_attention_vec=None, beta_attention_vec=beta_attention_vec)

        # alpha_attention_vec, latency_vec, _, alpha_blocks, beta_attention_vec, _, beta_blocks = \
        #     flatten_attention_latency_grad_alpha_beta_blocks(self.list_alphas)
        # print('updated:')
        # print('beta_attention_vec')
        # print(np.reshape(beta_attention_vec, (-1, 3)))

        # Latency test
        # constrained_latency = (A_latency @ x).item()
        # print('Latency constraint: {} <= {}'.format(constrained_latency, b_latency.item()))
        # stepped_latency = (A_latency @ beta_attention_vec).item()
        # print('Stepped latency: {}'.format(stepped_latency))
        # actual_expected_latency = expected_latency(self.list_alphas).item()
        # print('Updated actual expected latency: {}'.format(actual_expected_latency))
        # assert actual_expected_latency <= b_latency.item() + self.EPSILON
        # assert np.allclose(stepped_latency, actual_expected_latency)

        if self._epoch is not None:
            latency = self.latency_formula(alpha_attention_vec, beta_attention_vec, fixed_latency=self._fixed_latency)
            latency = {'latency_formula': latency, 'constraint': self.T}
            update_alpha_beta_tensorboard(self._epoch + 1, self.list_alphas, self.writer, latency)

        return True


    def latency_formula(self, alpha, beta, fixed_latency=0):
        return fixed_latency + (beta @ self.Q.T + self.p) @ alpha


    def reset_gamma_step(self):
        self.k = 0


    def set_max_gamma_step(self, val):
        self._max_gamma = min(val, 1)


    def _calculate_step_size(self, n=1, one=False):
        gamma = self._max_gamma * 2 * n / (self.k + 2 * n)
        self.k += 1
        if one:
            return self._max_gamma

        if self.writer is not None:
            self.writer.add_scalar('gamma_step', gamma, self.k)

        return gamma


    def set_writer(self, writer):
        self.writer = writer


    def requires_grad_(self, requires_grad=True):
        for group in self.param_groups:
            for p in group['params']:
                p.requires_grad = requires_grad


    def none_grad(self):
        for group in self.param_groups:
            for p in group['params']:
                p.grad = None


    def alternate_block_step(self):
        if self.k % 2 > 0:
            success = self.alpha_step()
        else:
            success = self.beta_step()

        if not success:
            self.alpha_step()


    def random_block_step(self):
        prob = torch.tensor(np.random.random()).cuda()
        if DistributedManager.distributed:
            grp = DistributedManager.grp
            ws = torch.distributed.get_world_size()
            torch.distributed.all_reduce(prob, op=torch.distributed.ReduceOp.SUM, group=grp)
            prob /= ws

        if prob < 0.5:
            success = self.alpha_step()
        else:
            success = self.beta_step()

        if not success:
            self.alpha_step()


    def step(self):
        self.reset_state()

        if self._epoch is not None and self._epoch == 0:
            alpha_attention_vec, _, _, _, beta_attention_vec, _, _ = \
                flatten_attention_latency_grad_alpha_beta_blocks(self.list_alphas)
            latency = self.latency_formula(alpha_attention_vec, beta_attention_vec, fixed_latency=self._fixed_latency)
            latency = {'latency_formula': latency, 'constraint': self.T}
            update_alpha_beta_tensorboard(0, self.list_alphas, self.writer, latency)

        if self.only_alpha:
            self.alpha_step()
        elif self.only_beta:
            self.beta_step()
        elif self._alternate:
            self.alternate_block_step()
        else:
            self.random_block_step()

        return self._done
