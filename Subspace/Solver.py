import os
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from Subspace.func import generate_rand_points_on_sphere
from Subspace.func import generate_sample_points
from Subspace.func import linear_regression
from Subspace.func import big_quadratic_regression
from Subspace.func import quadratic_regression
from Subspace.func import solve_for_tau

# NOTE: 1 means we need record of information, 0 means not
INFO_OPT = 1

# NOTE: 1 means return gradient norm, 0 means not
GRAD_NORM_OPT = 1

# TODO: it is hard-coded now to avoid numerical issues
epsilon = 1e-12

# NOTE: it is used to compute sample radius and gd sample size
machine_eps = 1e-16

# TODO: it is hard-coded now
neta = 0.15

# TODO: it is hard-coded now
# NOTE: it is used only when using fixed sampling step size
sample_radius_0 = 1e-6
gd_sample_size_0 = 1e-8

# NOTE: 0 means fixed sampling step size, 1 means self-adaptive sampling step size
sample_radius_opt = 1
gd_sample_size_opt = 1

# NOTE: 0 means Guassian sphere sampling, 1 means LHS sampling
sample_opt = 0


# record all the needed information for output
class his_manager:
    def __init__(self):
        self.message = None  # str
        self.success = None  # bool
        self.y_star = None
        self.x_star = None

        self.nfev = 0
        self.niter = 0
        self.y_iter_his = []
        self.y_total_his = []

        self.tr_radius_his = []
        self.sample_radius_his = []
        self.gd_sample_size_his = []

        self.rho_his = []
        self.grad_norm_his = []
        self.step_size_his = []


# subspace solver
class solver:
    def __init__(self, obj_func, x0, target=None):
        # the input of obj_func should be 1D array
        # x0 should be 1D array
        self.his = his_manager()

        self.obj_func = obj_func
        self.x0 = x0

        self.target = target

    def init_option(
        self,
        max_iter,
        max_nfev,
        mom_num=1,
        gd_num=1,
        ds_num=0,
        extra_sample_num=0,
        gd_estimator_opt="FFD",
        gd_sample_num_opt=1,
        lbfgs_opt=0,
        history_size=100,
    ):
        self.max_iter = max_iter
        self.max_nfev = max_nfev

        self.mom_num = mom_num
        self.gd_num = gd_num
        self.ds_num = ds_num

        self.extra_sample_num = extra_sample_num

        self.gd_estimator_opt = gd_estimator_opt
        if self.gd_estimator_opt in ["BSG"]:
            self.gd_sample_num_opt = gd_sample_num_opt

        self.lbfgs_opt = lbfgs_opt
        if self.lbfgs_opt == 1:
            self.history_size = history_size

    def init_hyperparams(
        self,
        tr_radius_0=1e-2,
        tr_radius_tol=1e-12,
        tr_radius_max=np.inf,
        rou_1_bar=0.25,
        rou_2_bar=0.75,
        gamma_1=0.8,
        gamma_2=2,
    ):
        self.tr_radius_0 = tr_radius_0
        self.tr_radius_tol = tr_radius_tol
        self.tr_radius_max = tr_radius_max
        self.rou_1_bar = rou_1_bar
        self.rou_2_bar = rou_2_bar
        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2

    def get_func_val(self, x):
        y = self.obj_func(x)
        self.his.nfev += 1
        self.his.y_total_his.append(y)
        return y

    # NOTE: it is used in draw_landscape
    def get_val(self, x):
        y = self.obj_func(x)
        return y

    def LBFGS(self):
        pass

    def start(self):
        self.dim = len(self.x0)
        self.x_current = self.x0
        self.y_current = self.get_func_val(self.x_current)

        self.tr_radius = self.tr_radius_0
        if gd_sample_size_opt == 1:
            self.gd_sample_size = (machine_eps * max(1, np.abs(self.y_current))) ** (
                1 / 2
            )
        elif gd_sample_size_opt == 0:
            self.gd_sample_size = gd_sample_size_0
        if sample_radius_opt == 1:
            self.sample_radius = (machine_eps * max(1, np.abs(self.y_current))) ** (
                1 / 3
            )
        elif sample_radius_opt == 0:
            self.sample_radius = sample_radius_0

        if self.gd_estimator_opt in ["BSG"]:
            if self.gd_sample_num_opt <= 1:
                self.gd_sample_num = int(self.gd_sample_num_opt * self.dim)
            else:
                self.gd_sample_num = int(self.gd_sample_num_opt)

        if self.mom_num > 0:
            self.mom_buffer = deque(maxlen=self.mom_num)

        if self.gd_num > 0:
            self.gd_buffer = deque(maxlen=self.gd_num)

        if self.lbfgs_opt == 1:
            self.lbfgs_buffer = deque(maxlen=1)
            self.state = {}

        self.init_gd_estimator()

        self.slow_move = 0

    def get_directions(self):
        directions = []
        if self.gd_num > 0:
            directions += list(self.gd_buffer)
        if self.mom_num > 0 and self.his.niter > 0:
            directions += list(self.mom_buffer)
        if self.ds_num > 0:
            for _ in range(self.ds_num):
                directions.append(np.random.normal(0, 1, self.dim))
        if self.lbfgs_opt == 1:
            directions += list(self.lbfgs_buffer)
        self.directions = np.vstack(directions)  # each row is a direction

    def preprocess_directions(self):
        Q, _ = np.linalg.qr(self.directions.T, mode="reduced")
        self.basis = Q.T  # each row is a direction
        self.sub_dim = self.basis.shape[0]

    def transform_from_vec_to_d(self, vec):
        # vec is a vector in the subspace, d is a vector in the original space
        vec = vec.flatten()
        d_new = vec @ self.basis
        return d_new

    def construct_model_1(self):
        self.subspace_const = self.y_current
        subspace_grad = self.grad @ self.basis.T

        num = max(1, int(0.5 * self.sub_dim * (self.sub_dim + 1)))
        num += self.extra_sample_num
        num = int(num)

        self.r = self.sample_radius
        if sample_opt == 0:
            self.X = generate_rand_points_on_sphere(self.sub_dim, num, radius=self.r)
        elif sample_opt == 1:
            self.X = generate_sample_points(self.sub_dim, num, radius=self.r)

        y_ls = []
        for n in range(num):
            d_new = self.transform_from_vec_to_d(self.X[n])
            exact_val = self.get_func_val(self.x_current + d_new)
            y_ls.append(
                exact_val - self.subspace_const - np.dot(subspace_grad, self.X[n])
            )
        y = np.array(y_ls)

        self.subspace_hessian = quadratic_regression(self.X, y)
        self.subspace_grad = subspace_grad.reshape(-1, 1)

    def construct_model_2(self):
        self.subspace_const = self.y_current

        num = max(1, int(0.5 * self.sub_dim * (self.sub_dim + 1) + self.sub_dim))
        num += self.extra_sample_num
        num = int(num)

        self.r = self.sample_radius
        if sample_opt == 0:
            self.X = generate_rand_points_on_sphere(self.sub_dim, num, radius=self.r)
        elif sample_opt == 1:
            self.X = generate_sample_points(self.sub_dim, num, radius=self.r)

        y_ls = []
        for n in range(num):
            d_new = self.transform_from_vec_to_d(self.X[n])
            exact_val = self.get_func_val(self.x_current + d_new)
            y_ls.append(exact_val - self.subspace_const)
        y = np.array(y_ls)

        self.subspace_grad, self.subspace_hessian = big_quadratic_regression(self.X, y)

    def construct_model_3(self):
        self.subspace_const = self.y_current

        num = max(1, int(0.5 * self.sub_dim * (self.sub_dim + 1) + self.sub_dim))
        num += self.extra_sample_num
        num = int(num)

        X = generate_rand_points_on_sphere(self.sub_dim, num)

        shuffled_indices = np.random.permutation(num)
        split_point = len(shuffled_indices) // 2
        indices_part1 = shuffled_indices[:split_point]
        indices_part2 = shuffled_indices[split_point:]

        self.r = self.sample_radius
        r_1 = self.r
        r_2 = 0.5 * self.r
        self.X = np.vstack((r_1 * X[indices_part1, :], r_2 * X[indices_part2, :]))

        y_ls = []
        for n in range(num):
            d_new = self.transform_from_vec_to_d(self.X[n])
            exact_val = self.get_func_val(self.x_current + d_new)
            y_ls.append(exact_val - self.subspace_const)
        y = np.array(y_ls)

        self.subspace_grad, self.subspace_hessian = big_quadratic_regression(self.X, y)

    def get_model_val(self, p):
        val = (
            self.subspace_const
            + p.T @ self.subspace_grad
            + 0.5 * p.T @ self.subspace_hessian @ p
        )
        return val[0][0]

    def truncated_CG(self):
        max_iter = self.sub_dim * 100

        s = np.zeros((self.sub_dim, 1))
        r = self.subspace_grad
        r_norm_0 = np.linalg.norm(r)
        p = -self.subspace_grad
        k = 0
        while k < max_iter:
            if p.T @ self.subspace_hessian @ p <= 0:
                t = solve_for_tau(s, p, self.tr_radius)
                return s + t * p
            alpha = (r.T @ r) / (p.T @ self.subspace_hessian @ p)
            s_new = s + alpha * p
            if np.linalg.norm(s_new) >= self.tr_radius:
                t = solve_for_tau(s, p, self.tr_radius)
                return s + t * p
            r_new = r + alpha * (self.subspace_hessian @ p)
            if np.linalg.norm(r_new) < min(epsilon, epsilon * r_norm_0):
                return s_new
            beta = (r_new.T @ r_new) / (r.T @ r)
            p = -1 * r_new + beta * p
            k += 1
            s = s_new
            r = r_new
        return s

    def step(self):
        if self.his.niter == 0 or self.rho > neta:
            self.grad = self.gd_estimator()

            # upgrade directions in the buffer
            if self.gd_num > 0:
                self.gd_buffer.append(np.copy(self.grad))

            if self.lbfgs_opt == 1:
                self.lbfgs_buffer.append(np.copy(self.LBFGS()))

            # construct basis
            self.get_directions()
            self.preprocess_directions()

            # construct model
            self.construct_model()
        elif self.ds_num > 0:
            # construct basis
            self.get_directions()
            self.preprocess_directions()

            # construct model
            self.construct_model()
        else:
            # construct model
            self.construct_model()

        # record information: current function value and current trust region radius
        if INFO_OPT == 1:
            self.his.y_iter_his.append(self.y_current)
            self.his.tr_radius_his.append(self.tr_radius)
            self.his.sample_radius_his.append(self.sample_radius)
            self.his.gd_sample_size_his.append(self.gd_sample_size)

        # get p_star
        self.p_star = self.truncated_CG()
        self.p_star_norm = np.linalg.norm(self.p_star)

        # get d_star
        self.d_star = self.transform_from_vec_to_d(self.p_star)
        self.tcg_x = self.x_current + self.d_star
        self.tcg_y = self.get_func_val(self.tcg_x)

        tcg_y_model = self.get_model_val(self.p_star)
        y_current_model = self.get_model_val(np.zeros((self.sub_dim, 1)))

        if y_current_model - tcg_y_model > 0:
            self.rho = (self.y_current - self.tcg_y) / (y_current_model - tcg_y_model)
        else:
            print("TCG generate a wrong point")
            self.rho = -1

        # update new point
        if self.rho > neta:
            if self.y_current - self.tcg_y < epsilon:
                self.slow_move += 1
            else:
                self.slow_move = 0
            self.x_pre = self.x_current  # it is used in draw_landscape
            self.x_current = self.tcg_x
            self.y_current = self.tcg_y
        else:
            self.slow_move += 1

        self.step_end()

        # return the norm of gradient
        if GRAD_NORM_OPT == 1:
            return self.grad_norm

    def para_upd(self):
        if self.rho < self.rou_1_bar:
            self.tr_radius *= self.gamma_1
        elif self.rho > self.rou_2_bar and self.p_star_norm >= 0.9 * self.tr_radius:
            self.tr_radius = min(self.tr_radius * self.gamma_2, self.tr_radius_max)

        if gd_sample_size_opt == 1:
            self.gd_sample_size = (machine_eps * max(1, np.abs(self.y_current))) ** (
                1 / 2
            )
        if sample_radius_opt == 1:
            self.sample_radius = (machine_eps * max(1, np.abs(self.y_current))) ** (
                1 / 3
            )

    def mom_upd(self):
        self.mom_buffer.append(self.d_star)

    def step_end(self):
        # NOTE: the norm of subspace gradient equals to the norm of gradient if gd_num = 1 ( gd becomes the first base)
        if GRAD_NORM_OPT == 1 or INFO_OPT == 1:
            self.grad_norm = np.linalg.norm(self.grad)
        if INFO_OPT == 1:
            if self.rho > neta:
                self.step_size = np.linalg.norm(self.d_star)
            else:
                self.step_size = 0

        # output some relative information for observation
        if INFO_OPT == 1:
            print(
                f"Iteration: {self.his.niter} | Nfev: {self.his.nfev} | Function value: {self.y_current:.6f} | Rho: {self.rho:.4f} | Trust region radius: {self.tr_radius:.6f} | Step size: {self.step_size:.6f} | Gradient norm: {self.grad_norm:.4f}"
            )

        # record information
        self.his.niter += 1
        if INFO_OPT == 1:
            self.his.rho_his.append(self.rho)
            self.his.grad_norm_his.append(self.grad_norm)
            self.his.step_size_his.append(self.step_size)

        # update momentum
        if self.rho > neta:
            if self.mom_num > 0:
                self.mom_upd()
        else:
            pass

        # update hyperparameters
        self.para_upd()

    def BSG(self):
        B = generate_rand_points_on_sphere(self.dim, num_points=self.gd_sample_num)
        x_sample = self.x_current + self.gd_sample_size * B
        y_sample = np.apply_along_axis(
            self.get_func_val, axis=1, arr=x_sample
        )  # 1D array
        grad = (
            np.mean(
                ((y_sample - self.y_current) / self.gd_sample_size).reshape(-1, 1) * B,
                axis=0,
            )
            * self.dim
        )
        return grad

    def FFD(self):
        B = np.eye(self.dim)
        x_sample = self.x_current + self.gd_sample_size * B
        y_sample = np.apply_along_axis(
            self.get_func_val, axis=1, arr=x_sample
        )  # 1D array
        grad = (y_sample - self.y_current) / self.gd_sample_size

        return grad

    def CFD(self):
        B1 = np.eye(self.dim)
        B2 = -np.eye(self.dim)
        x_sample_1 = self.x_current + self.gd_sample_size * B1
        y_sample_1 = np.apply_along_axis(
            self.get_func_val, axis=1, arr=x_sample_1
        )  # 1D array
        x_sample_2 = self.x_current + self.gd_sample_size * B2
        y_sample_2 = np.apply_along_axis(
            self.get_func_val, axis=1, arr=x_sample_2
        )  # 1D array
        grad = (y_sample_1 - y_sample_2) / (2 * self.gd_sample_size)
        return grad

    def LI(self):
        B = generate_rand_points_on_sphere(self.dim, num_points=self.dim)
        x_sample = self.x_current + self.gd_sample_size * B
        y_sample = np.apply_along_axis(
            self.get_func_val, axis=1, arr=x_sample
        )  # 1D array
        delta_y = y_sample - self.y_current
        grad = linear_regression((self.gd_sample_size * B), delta_y)
        return grad

    def init_gd_estimator(self):
        if self.gd_estimator_opt == "BSG":
            self.gd_estimator = self.BSG
        elif self.gd_estimator_opt == "FFD":
            self.gd_estimator = self.FFD
        elif self.gd_estimator_opt == "CFD":
            self.gd_estimator = self.CFD
        elif self.gd_estimator_opt == "LI":
            self.gd_estimator = self.LI

        # TODO: now it is hard-coded
        self.construct_model = self.construct_model_3

    def check_stop(self, gd_tol=1e-6):
        if self.his.niter >= self.max_iter:
            self.his.x_star = self.x_current
            self.his.y_star = self.y_current
            self.his.message = (
                "Return from solver because max iteration number has achieved"
            )
            self.his.success = False
            return 1

        if self.his.nfev >= self.max_nfev:
            self.his.x_star = self.x_current
            self.his.y_star = self.y_current
            self.his.message = "Return from solver because max function evaluations number has achieved"
            self.his.success = False
            return 1

        if self.tr_radius < self.tr_radius_tol:
            self.his.x_star = self.x_current
            self.his.y_star = self.y_current
            self.his.message = (
                "Return from solver because trust region radius tolerance has achieved"
            )
            self.his.success = True
            return 1

        # if (
        #     self.gd_estimator_opt == "FFD"
        #     and GRAD_NORM_OPT == 1
        #     and self.his.niter >= 1
        #     and self.grad_norm < gd_tol
        # ):
        #     self.his.x_star = self.x_current
        #     self.his.y_star = self.y_current
        #     self.his.message = "Return from solver because stopping criterion for gradient norm has achieved"
        #     self.his.success = True
        #     return 1

        if self.slow_move >= 50:
            self.his.x_star = self.x_current
            self.his.y_star = self.y_current
            self.his.message = "Return from solver because stopping criterion for slow moving has achieved"
            self.his.success = True
            return 1

        if self.target != None and self.y_current < self.target:
            self.his.x_star = self.x_current
            self.his.y_star = self.y_current
            self.his.message = "Return from solver because great accuracy has achieved"
            self.his.success = True
            return 1

        return 0

    def solve(self):
        self.start()
        while True:
            stop = self.check_stop()
            if stop == 0:
                self.step()
            else:
                return 0

    def observe(self, landscape_freq=1):
        self.start()
        index = 0
        while True:
            stop = self.check_stop()
            if stop == 0:
                self.step()
                if index % landscape_freq == 1:
                    self.draw_landscape(id=index)
                index += 1
            else:
                self.draw_landscape(id=index)
                return 0

    def display_result(self):
        res = self.his
        print("----------------")
        print("Message: ", res.message)
        print("Success: ", res.success)
        print("Number of function evaluations: ", res.nfev)
        print("Number of iterations: ", res.niter)
        print("Y_star: ", res.y_star)
        print("----------------")
        return res

    # NOTE: this function should be used in the end when INFO_OPT == 1
    def draw_info(self):
        fig, axes = plt.subplots(2, 2, figsize=(8, 6))
        (ax1, ax2), (ax3, ax4) = axes

        ax1.plot(self.his.y_iter_his)
        ax2.plot(self.his.tr_radius_his, label=rf"$\Delta_k$")
        ax2.plot(self.his.step_size_his, label="Step size")
        ax2.plot(self.his.sample_radius_his, label=rf"$r_k$")
        ax2.plot(self.his.gd_sample_size_his, label=rf"$\delta_k$")
        ax3.plot(self.his.grad_norm_his)
        ax4.plot(self.his.rho_his)

        ax1.set_xlabel("Number of iterations")
        ax1.set_ylabel("Function value")
        ax2.set_xlabel("Number of iterations")
        ax2.set_ylabel("Tr radius")
        ax3.set_xlabel("Number of iterations")
        ax3.set_ylabel("Gradient norm")
        ax4.set_xlabel("Number of iterations")
        ax4.set_ylabel("Rho")

        ax1.set_yscale("log")
        ax2.set_yscale("log")
        ax3.set_yscale("log")
        ax4.set_yscale("log")

        ax2.legend()

        fig.suptitle(f"DRSOM training information")
        fig.subplots_adjust(wspace=0.4, hspace=0.4)

        folder = f"Subspace/fig"
        if not os.path.exists(folder):
            os.makedirs(folder)

        fig.savefig(f"{folder}/temp.png", dpi=1000)
        plt.close()
        return 0

    # NOTE: this function should be used after one step
    def draw_landscape(self, id=0, resolution=50, padding=0.1):
        if self.sub_dim != 2:
            print(f"Subspace dimension != 2: {self.sub_dim}")
            return 0

        # get two basis
        d1 = self.basis[0]
        d2 = self.basis[1]

        # get three points
        star_x = [np.dot(self.d_star, d1), np.dot(self.d_star, d2)]
        init_x = [0, 0]
        if self.rho > neta:
            final_x = star_x
        else:
            final_x = init_x

        # get dynamic range
        r = self.r

        def get_dynamic_range(opt, axes):
            if opt == 0:
                fig_r = 0.5 * np.linalg.norm(self.d_star)
                all_x1_vals = [0, star_x[0], fig_r, -fig_r]
                all_x2_vals = [0, star_x[1], fig_r, -fig_r]
                min_x1, max_x1 = min(all_x1_vals), max(all_x1_vals)
                min_x2, max_x2 = min(all_x2_vals), max(all_x2_vals)
                range_x1 = (
                    min_x1 - padding * abs(max_x1 - min_x1),
                    max_x1 + padding * abs(max_x1 - min_x1),
                )
                range_x2 = (
                    min_x2 - padding * abs(max_x2 - min_x2),
                    max_x2 + padding * abs(max_x2 - min_x2),
                )
                x1_vals = np.linspace(range_x1[0], range_x1[1], resolution)
                x2_vals = np.linspace(range_x2[0], range_x2[1], resolution)
            else:
                all_x1_vals = [0, r, -r]
                all_x2_vals = [0, r, -r]
                min_x1, max_x1 = min(all_x1_vals), max(all_x1_vals)
                min_x2, max_x2 = min(all_x2_vals), max(all_x2_vals)
                range_x1 = (
                    min_x1 - padding * abs(max_x1 - min_x1),
                    max_x1 + padding * abs(max_x1 - min_x1),
                )
                range_x2 = (
                    min_x2 - padding * abs(max_x2 - min_x2),
                    max_x2 + padding * abs(max_x2 - min_x2),
                )
                x1_vals = np.linspace(range_x1[0], range_x1[1], resolution)
                x2_vals = np.linspace(range_x2[0], range_x2[1], resolution)
            for ax in axes:
                ax.set_xlim(range_x1[0], range_x1[1])
                ax.set_ylim(range_x2[0], range_x2[1])
            return x1_vals, x2_vals

        # get landscape values
        def get_landscape_values(x1_vals, x2_vals):
            z_exact_vals = np.zeros((resolution, resolution))
            z_model_vals = np.zeros((resolution, resolution))

            for i, x1 in enumerate(x1_vals):
                for j, x2 in enumerate(x2_vals):
                    coeff = np.array([x1, x2])
                    d_new = x1 * d1 + x2 * d2
                    z_exact_vals[j, i] = self.get_val(self.x_pre + d_new)
                    z_model_vals[j, i] = self.get_model_val(coeff.reshape(-1, 1))

            vmin = min(z_exact_vals.min(), z_model_vals.min())
            vmax = max(z_exact_vals.max(), z_model_vals.max())

            return z_exact_vals, z_model_vals, vmin, vmax

        # draw landscape and trajectories
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        (ax1, ax2), (ax3, ax4) = axes

        # part 1
        x1_vals, x2_vals = get_dynamic_range(opt=0, axes=(ax1, ax2))
        z_exact_vals, z_model_vals, vmin, vmax = get_landscape_values(x1_vals, x2_vals)

        contour1 = ax1.contourf(
            x1_vals,
            x2_vals,
            z_exact_vals,
            levels=100,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        if sample_opt == 0:
            circle = patches.Circle((0, 0), r, fill=False, color="red")
            circle.set_clip_box(ax1.bbox)
            ax1.add_patch(circle)
        elif sample_opt == 1:
            square = patches.Rectangle((-r, -r), 2 * r, 2 * r, fill=False, color="red")
            square.set_clip_box(ax1.bbox)
            ax1.add_patch(square)
        ax1.scatter(
            0,
            0,
            color="tab:green",
            marker="o",
            s=30,
            label=rf"$x_{{{self.his.niter-1}}}$",
        )
        ax1.scatter(
            final_x[0],
            final_x[1],
            color="tab:orange",
            marker="o",
            s=30,
            label=rf"$x_{{{self.his.niter}}}$",
        )
        ax1.scatter(
            star_x[0],
            star_x[1],
            color="tab:red",
            marker="*",
            s=15,
            label="Model Solution",
        )

        _ = ax2.contourf(
            x1_vals,
            x2_vals,
            z_model_vals,
            levels=100,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        if sample_opt == 0:
            circle = patches.Circle((0, 0), r, fill=False, color="red")
            circle.set_clip_box(ax2.bbox)
            ax2.add_patch(circle)
        elif sample_opt == 1:
            square = patches.Rectangle((-r, -r), 2 * r, 2 * r, fill=False, color="red")
            square.set_clip_box(ax2.bbox)
            ax2.add_patch(square)
        ax2.scatter(
            0,
            0,
            color="tab:green",
            marker="o",
            s=30,
            label=rf"$x_{{{self.his.niter-1}}}$",
        )
        ax2.scatter(
            final_x[0],
            final_x[1],
            color="tab:orange",
            marker="o",
            s=30,
            label=rf"$x_{{{self.his.niter}}}$",
        )
        ax2.scatter(
            star_x[0],
            star_x[1],
            color="tab:red",
            marker="*",
            s=15,
            label="Model Solution",
        )

        # part 2
        x1_vals, x2_vals = get_dynamic_range(opt=1, axes=(ax3, ax4))
        z_exact_vals, z_model_vals, vmin, vmax = get_landscape_values(x1_vals, x2_vals)

        contour3 = ax3.contourf(
            x1_vals,
            x2_vals,
            z_exact_vals,
            levels=20,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        if sample_opt == 0:
            circle = patches.Circle((0, 0), r, fill=False, color="red")
            ax3.add_patch(circle)
        elif sample_opt == 1:
            square = patches.Rectangle((-r, -r), 2 * r, 2 * r, fill=False, color="red")
            ax3.add_patch(square)
        ax3.scatter(
            0,
            0,
            color="tab:green",
            marker="o",
            s=30,
            label=rf"$x_{{{self.his.niter-1}}}$",
        )
        for x in self.X:
            ax3.scatter(x[0], x[1], color="tab:pink", marker="o", s=30)

        _ = ax4.contourf(
            x1_vals,
            x2_vals,
            z_model_vals,
            levels=20,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        if sample_opt == 0:
            circle = patches.Circle((0, 0), r, fill=False, color="red")
            ax4.add_patch(circle)
        elif sample_opt == 1:
            square = patches.Rectangle((-r, -r), 2 * r, 2 * r, fill=False, color="red")
            ax4.add_patch(square)
        ax4.scatter(
            0,
            0,
            color="tab:green",
            marker="o",
            s=30,
            label=rf"$x_{{{self.his.niter-1}}}$",
        )
        for x in self.X:
            ax4.scatter(x[0], x[1], color="tab:pink", marker="o", s=30)

        ax1.legend()
        ax1.set_xlabel("First direction")
        ax1.set_ylabel("Second direction")
        ax1.set_title("Exact value")

        ax2.legend()
        ax2.set_xlabel("First direction")
        ax2.set_ylabel("Second direction")
        ax2.set_title("Model value")

        ax3.legend()
        ax3.set_xlabel("First direction")
        ax3.set_ylabel("Second direction")
        ax3.set_title("Exact value")

        ax4.legend()
        ax4.set_xlabel("First direction")
        ax4.set_ylabel("Second direction")
        ax4.set_title("Model value")

        fig.subplots_adjust(wspace=0.4, hspace=0.4)
        cbar1 = fig.colorbar(
            contour1, ax=(ax1, ax2), orientation="vertical", label="Function value"
        )
        cbar2 = fig.colorbar(
            contour3, ax=(ax3, ax4), orientation="vertical", label="Function value"
        )
        fig.suptitle(f"DRSOM Landscape and Optimization Trajectories")

        for ax in axes.flat:
            ax.ticklabel_format(style="sci", axis="both", scilimits=(-2, 2))
        cbar1.ax.ticklabel_format(style="sci", scilimits=(-2, 2))
        cbar2.ax.ticklabel_format(style="sci", scilimits=(-2, 2))

        folder = f"Subspace/fig/landscape"
        if not os.path.exists(folder):
            os.makedirs(folder)
        fig.savefig(f"{folder}/{id}.png", dpi=500)
        plt.close()
