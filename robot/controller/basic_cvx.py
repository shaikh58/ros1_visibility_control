import numpy as np
from env.config import (epsilon_s, alpha_fov, radius, v_lb, v_ub, omega_lb, omega_ub, use_penalty_smoothing, omega_penalty_l2,
                        pen_smooth_v, pen_smooth_w, alpha_obs, use_ema_omega, ema_frac)
import cvxpy as cp
from utils import SDF_RT, polygon_SDF


class BasicController:
    def __init__(self, mapinfo=None):
        self.map_info = mapinfo
        self.raytracing_radius = radius
        self.raytracing_res = 50
        self.raytracing_fov_range_angle = np.pi / 6  # angle to sweep through for fov

    @staticmethod
    def G(x):
        theta = x[2]
        return np.array([[np.cos(theta), 0], [np.sin(theta), 0], [0, 1]])

    def R(self, theta):
        return np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])

    def solvecvx(self, rbt, tgt, ydot, ref, scan, mapinfo, u_prev):
        """CVXPY solver"""
        self.map_info = mapinfo
        '''Visibility CBF constraint'''
        rdrx, rdry, sdf = self.visibility_gradient(rbt, tgt)
        P_visibility = np.array([[1, 0], [0, 1]])  # Quadratic term (must be positive semi-definite)
        P_smoothing = np.diag([pen_smooth_v, pen_smooth_w])
        u = cp.Variable((2, 1))
        ref = ref.reshape((2, 1))
        u_prev = u_prev.reshape((2, 1))
        rbt = rbt.reshape((3, 1))
        grad = rdrx.reshape((1, 3))
        grady = rdry.reshape((1, 3))
        ydot = np.array([ydot[0], ydot[1], 0]).reshape((3, 1))
        alf = alpha_fov
        LGh = -grad @ self.G(rbt.flatten())
        dhdt = -grady @ ydot
        h = -sdf
        print("Target SDF: ", sdf)
        safe_observation_margin = epsilon_s
        visibility_const = LGh @ u + dhdt + alf * (h - safe_observation_margin) >= 0

        '''DR CBF obstacle avoidance constraint'''
        top_1_obstacle = self.sample_cbf(scan, rbt)
        obs_const = True
        if top_1_obstacle is not None:
            l = 0.1
            r = 0.6
            obs_vec = top_1_obstacle - rbt[:2].T
            obs_dist = np.linalg.norm(obs_vec)
            print('dir vec of sample cbf: ', obs_vec)
            obs_grad = (obs_vec / np.linalg.norm(obs_vec)).T
            print('gradient of sample cbf: ', obs_grad)
            obs_const = obs_grad.T @ self.R(rbt[2, 0]) @ np.array([[1, 0], [0, l]]) @ u + alpha_obs * (obs_dist - r) >= 0

        '''Motion constraints'''
        vel_const_min_w = omega_lb <= u[1]
        vel_const_max_w = u[1] <= omega_ub

        objective = cp.Minimize(cp.quad_form((u - ref), P_visibility))
        if use_penalty_smoothing:
            objective = cp.Minimize(cp.quad_form((u - ref), P_visibility) + cp.quad_form((u-u_prev), P_smoothing))
        if omega_penalty_l2 > 0:
            print("Using L2 penalty on angular velocity")
            objective = cp.Minimize(cp.quad_form((u - ref), P_visibility) + omega_penalty_l2 * u[1]**2)

        constraints = [visibility_const, obs_const]
        prob = cp.Problem(objective, constraints)
        prob.solve()

        # check if problem was solved successfully
        if u.value is None or prob.status != 'optimal':
            print('-------------------------- SOLVER NOT OPTIMAL -------------------------')
            print("[In solver] solver status: ", prob.status)
            return ref, self.get_fov(rbt), sdf, obs_dist

        if use_ema_omega:
            u_final = u_prev + ema_frac * (u.value)
            print("EMA smoothed, original control: ", u.value-u_prev)
        else:
            u_final = u.value

        return self.crop(u_final), self.get_fov(rbt), sdf, obs_dist

    def sample_cbf(self, scan_w, rbt):
        if len(scan_w) == 0: return None
        scan_w = self.polar_to_cartesian(scan_w, rbt)
        distances = np.linalg.norm(scan_w - rbt[:2].T, axis=1)
        inner_radius = 0.1
        outer_radius = 3.0
        mask = (distances >= inner_radius) & (distances <= outer_radius)
        valid_indices = np.where(mask)[0]  # This gives the indices of valid distances
        if valid_indices.size > 0:
            min_index = valid_indices[np.argmin(distances[mask])]
            min_value = distances[min_index]
            return scan_w[min_index]
        else:
            print('Is this a free space?')
            return None

    def visibility_gradient(self, rbt, tgt):
        grad_est, SDF_center = self.new_fd_grad(rbt, tgt)
        return grad_est.reshape(3, ), np.hstack([-grad_est[:2].reshape(2, ), [0]]), SDF_center

    def new_fd_grad(self, rbt, tgt):
        delta = 0.05
        delta_t = 0.1
        X = np.array(
            ([[delta, 0, 0], [-delta, 0, 0], [0, delta, 0], [0, -delta, 0], [0, 0, delta_t], [0, 0, -delta_t]]))
        df_dx = (self.sdf(rbt + X[0], tgt) - self.sdf(rbt + X[1], tgt)) / (2 * delta)
        df_dy = (self.sdf(rbt + X[2], tgt) - self.sdf(rbt + X[3], tgt)) / (2 * delta)
        df_dt = (self.sdf(rbt + X[4], tgt) - self.sdf(rbt + X[5], tgt)) / (2 * delta_t)
        return np.array([df_dx, df_dy, df_dt]), self.sdf(rbt, tgt)

    def get_fov(self, rbt):
        rbt_d = self.w2m(self.map_info, rbt.reshape((3, 1))).squeeze()
        rt_visible = SDF_RT(rbt_d, self.raytracing_fov_range_angle, radius, self.raytracing_res, self.map_info['map'])
        visible_region = (self.m2w(self.map_info, rt_visible.T)[0:2, :]).T
        return visible_region

    def sdf(self, rbt, tgt):
        rbt_d = self.w2m(self.map_info, rbt.reshape((3, 1))).squeeze()
        rt_visible = SDF_RT(rbt_d, self.raytracing_fov_range_angle, radius, self.raytracing_res, self.map_info['map'])
        visible_region = (self.m2w(self.map_info, rt_visible.T)[0:2, :]).T
        return polygon_SDF(visible_region, tgt[0:2])

    def w2m(self, map_info, x_w):
        # formerly display2map
        res = map_info['resolution']
        h = map_info['height']
        x_w = x_w.reshape((3, -1))
        m_w = np.array([[map_info['origin'][0]], [map_info['origin'][1]], [0]])
        scale = np.array([[1 / res, 0, 0], [0, 1 / res, 0], [0, 0, 1]])
        dRm = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        dtm = np.array([[h], [0], [np.pi / 2]])
        return dRm @ scale @ (x_w - m_w) + dtm

    def m2w(self, map_info, x_m):
        # formerly map2display
        res = map_info['resolution']
        h = map_info['height']
        if len(x_m) == 2: x_m = np.array([[1, 0], [0, 1], [0, 0]]) @ x_m
        m_w = np.array([[map_info['origin'][0]], [map_info['origin'][1]], [0]])
        scale = np.array([[res, 0, 0], [0, res, 0], [0, 0, 1]])
        mRd = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        mtd = np.array([[-h], [0], [-np.pi / 2]])
        return scale @ mRd @ (x_m + mtd) + m_w

    @staticmethod
    def crop(u):
        u[0] = min(max(u[0], v_lb), v_ub)
        u[1] = min(max(u[1], omega_lb), omega_ub)
        return u

    def polar_to_cartesian(self, ranges, lidar_pose):
        # Lidar pose
        x_lidar, y_lidar, theta_lidar = lidar_pose.reshape(3,)

        # Number of range measurements
        n = len(ranges)

        # Angles for each range measurement assuming 360-degree field of view
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + theta_lidar

        # Convert polar to Cartesian
        x_points = ranges * np.cos(angles) + x_lidar
        y_points = ranges * np.sin(angles) + y_lidar

        # Stack into an (n, 2) array
        cartesian_points = np.vstack((x_points, y_points)).T

        return cartesian_points
