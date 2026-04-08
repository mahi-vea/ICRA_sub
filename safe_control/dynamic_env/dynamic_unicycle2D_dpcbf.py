import numpy as np
import casadi as ca

from safe_control.robots.dynamic_unicycle2D import DynamicUnicycle2D

"""
Dynamic Parabolic CBF (DPCBF) for the Dynamic Unicycle 2D model.

Inherits from DynamicUnicycle2D and overrides only the continuous-time and
discrete-time barrier functions with their DPCBF counterparts, following the
same pattern as KinematicBicycle2D_DPCBF.

State  : X = [x, y, theta, v]
Control: U = [a, omega]

The DPCBF is built on the collision-cone idea in relative coordinates:
    p_rel = [obs_x - x,   obs_y - y]
    v_rel = [obs_vx - v*cos(theta),  obs_vy - v*sin(theta)]

A local frame aligned with p_rel is constructed so that:
    v_rel_new_x  — component along the line-of-sight (closing speed)
    v_rel_new_y  — component perpendicular to the line-of-sight (lateral speed)

Barrier function (relative degree 1 after the frame rotation):
    h = v_rel_new_x + lambda(x) * v_rel_new_y^2 + mu(x)

with adaptive parameters:
    lambda = k_lambda * sqrt(d_safe) / v_rel_mag * sqrt(s^2 - 1) / ego_dim
    mu     = k_mu    * sqrt(d_safe)               * sqrt(s^2 - 1) / ego_dim

Reference: https://arxiv.org/abs/2403.07043
"""


class DynamicUnicycle2D_DPCBF(DynamicUnicycle2D):
    def __init__(self, dt, robot_spec, k_lambda=0.1, k_mu=0.5):
        super().__init__(dt, robot_spec)
        self.k_lambda = k_lambda
        self.k_mu = k_mu

    # ------------------------------------------------------------------
    # Continuous-time DPCBF  (relative degree 1)
    # ------------------------------------------------------------------
    def agent_barrier(self, X, obs, robot_radius, s=1.05):
        """
        Continuous-Time Dynamic Parabolic CBF for circular obstacles.

        For superellipsoid obstacles (obs[-1] == 1) the method falls back to
        the parent class implementation, which provides the standard HOCBFs
        needed by CBF-QP / MPC-CBF for those shapes.

        Returns
        -------
        h       : scalar  — barrier value
        dh_dx   : (1, 4)  — gradient ∂h/∂x   (Lf h and Lg h can be formed
                            by the caller as  h_dot = dh_dx @ (f + g @ u))
        """
        # Superellipsoid fallback — detected by obs[6] flag (not obs[-1], which
        # may be y_max from dynamic-env obs format [x,y,r,vx,vy,y_min,y_max])
        if obs.shape[0] >= 8 and obs[7] == 1:   # was obs[6]
            h, h_dot, dh_dot_dx = super().agent_barrier(X, obs, robot_radius)
            # Re-pack to match the (h, dh_dx) interface expected by DPCBF callers
            return h, dh_dot_dx  # caller must be aware of the fallback signature

        theta = X[2, 0]
        v     = X[3, 0]

        # Obstacle velocity — indices 3,4 when available (format: [x,y,r,vx,vy,...])
        if obs.shape[0] >= 5:
            obs_vel_x = obs[3]
            obs_vel_y = obs[4]
        else:
            obs_vel_x = 0.0
            obs_vel_y = 0.0

        # Combined safety radius with margin s
        ego_dim = (obs[2] + robot_radius) * s

        # Relative position and velocity
        p_rel = np.array([[obs[0] - X[0, 0]],
                          [obs[1] - X[1, 0]]])
        v_rel = np.array([[obs_vel_x - v * np.cos(theta)],
                          [obs_vel_y - v * np.sin(theta)]])

        p_rel_mag = np.linalg.norm(p_rel)
        eps       = 1e-6
        v_rel_mag = max(np.linalg.norm(v_rel), eps)  # clamp to avoid /0

        p_rel_x = p_rel[0, 0]
        p_rel_y = p_rel[1, 0]

        # Rotation into the line-of-sight frame
        rot_angle = np.arctan2(p_rel_y, p_rel_x)
        R = np.array([[ np.cos(rot_angle),  np.sin(rot_angle)],
                      [-np.sin(rot_angle),  np.cos(rot_angle)]])

        v_rel_new     = R @ v_rel
        v_rel_new_x   = v_rel_new[0, 0]
        v_rel_new_y   = v_rel_new[1, 0]

        d_safe = np.maximum(p_rel_mag**2 - ego_dim**2, eps)

        # Adaptive parameters
        scale      = np.sqrt(s**2 - 1) / ego_dim
        func_lambda = self.k_lambda * np.sqrt(d_safe) / v_rel_mag * scale
        func_mu     = self.k_mu    * np.sqrt(d_safe)               * scale

        # Barrier value
        h = v_rel_new_x + func_lambda * v_rel_new_y**2 + func_mu

        # ------------------------------------------------------------------
        # Gradient ∂h/∂x  (1 × 4)
        # ------------------------------------------------------------------
        # The unicycle control matrix g(X) = [[0,0],[0,0],[0,1],[1,0]].
        # The Lf h / Lg h terms are formed by the CBF-QP layer as dh_dx @ f
        # and dh_dx @ g, so we only need the gradient here.
        #
        # Derivatives follow the same chain-rule expansion as the bicycle
        # DPCBF, with the substitution that the unicycle's ego velocity is
        #   ego_vx = v*cos(theta),  ego_vy = v*sin(theta)
        # (identical kinematic structure for the position states).

        dh_dx = np.zeros((1, 4))

        # ∂h/∂x  (index 0)
        dh_dx[0, 0] = (
            p_rel_y * v_rel_new_y / p_rel_mag**2
            - self.k_lambda * p_rel_x * v_rel_new_y**2 / v_rel_mag / np.sqrt(d_safe) * scale
            - 2 * self.k_lambda * np.sqrt(d_safe) / v_rel_mag * v_rel_new_y
              * p_rel_y / p_rel_mag**2 * v_rel_new_x * scale
            - self.k_mu * p_rel_x / np.sqrt(d_safe) * scale
        )

        # ∂h/∂y  (index 1)
        dh_dx[0, 1] = (
            - p_rel_x * v_rel_new_y / p_rel_mag**2
            - self.k_lambda * p_rel_y * v_rel_new_y**2 / v_rel_mag / np.sqrt(d_safe) * scale
            + 2 * self.k_lambda * np.sqrt(d_safe) / v_rel_mag * v_rel_new_y
              * p_rel_x / p_rel_mag**2 * v_rel_new_x * scale
            - self.k_mu * p_rel_y / np.sqrt(d_safe) * scale
        )

        # ∂h/∂theta  (index 2)
        # v_rel rotates with theta; ∂v_rel/∂theta = [v*sin(theta), -v*cos(theta)]
        # After the frame rotation R, the component along the perpendicular axis
        # is  ∂v_rel_new_y/∂theta = v * cos(rot_angle - theta)
        # and along the l-o-s axis:  ∂v_rel_new_x/∂theta = -v * sin(rot_angle - theta)
        dh_dx[0, 2] = (
            - v * np.sin(rot_angle - theta)   # ∂v_rel_new_x/∂theta contribution
            - self.k_lambda * np.sqrt(d_safe) * scale
              * v * (obs_vel_x * np.sin(theta) - obs_vel_y * np.cos(theta))
              * v_rel_new_y**2 / v_rel_mag**3
            - 2 * self.k_lambda * np.sqrt(d_safe) * scale
              * v_rel_new_y * v * np.cos(rot_angle - theta) / v_rel_mag
        )

        # ∂h/∂v  (index 3)
        # ∂v_rel_new_x/∂v = -cos(rot_angle - theta)
        # ∂v_rel_new_y/∂v = -sin(rot_angle - theta) ... wait, let's be precise:
        #   v_rel_new = R @ [obs_vx - v*cos(theta), obs_vy - v*sin(theta)]
        #   ∂v_rel_new_x/∂v = -cos(rot_angle)*cos(theta) - sin(rot_angle)*sin(theta)
        #                    = -cos(rot_angle - theta)
        #   ∂v_rel_new_y/∂v = sin(rot_angle)*cos(theta) - cos(rot_angle)*sin(theta)
        #                    = sin(rot_angle - theta)   ... but sign flips:
        #                    = -(-sin(rot_angle)*cos(theta) + cos(rot_angle)*sin(theta))
        #   Let's just compute directly: same formula as bicycle
        dh_dx[0, 3] = (
            - np.cos(rot_angle - theta)   # ∂v_rel_new_x/∂v
            - self.k_lambda * np.sqrt(d_safe) * scale / v_rel_mag**3
              * (v - obs_vel_x * np.cos(theta) - obs_vel_y * np.sin(theta))
              * v_rel_new_y**2
            - 2 * self.k_lambda * np.sqrt(d_safe) * scale
              * v_rel_new_y * np.sin(rot_angle - theta) / v_rel_mag
        )

        return h, dh_dx

    # ------------------------------------------------------------------
    # Discrete-time DPCBF  (one-step, relative degree 1)
    # ------------------------------------------------------------------
    def agent_barrier_dt(self, x_k, u_k, obs, robot_radius, s=1.05):
        """
        Discrete-Time Dynamic Parabolic CBF for circular obstacles.

        Returns
        -------
        h_k  : CBF value at current step  (CasADi scalar)
        d_h  : h(k+1) - h(k)             (CasADi scalar)

        Note: only one forward step is needed because the DPCBF has
        relative degree 1.  For superellipsoid obstacles the parent's
        second-order discrete CBF is used instead.
        """
        # Superellipsoid fallback — check index 6 explicitly (not obs[-1],
        # which may be y_max in dynamic-env format [x,y,r,vx,vy,y_min,y_max])
        if obs.shape[0] >= 8 and obs[7] == 1:   # was obs[6]
            return super().agent_barrier_dt(x_k, u_k, obs, robot_radius)

        # One forward step using CasADi dynamics
        x_k1 = self.step(x_k, u_k, casadi=True)

        def h_dpcbf(x, obs, robot_radius, s=1.05):
            """CasADi DPCBF barrier evaluated at state x."""
            theta = x[2, 0]
            v     = x[3, 0]

            # Velocity available at indices 3,4 (format: [x,y,r,vx,vy,...])
            if obs.shape[0] >= 5:
                obs_vel_x = obs[3]
                obs_vel_y = obs[4]
            else:
                obs_vel_x = 0.0
                obs_vel_y = 0.0

            ego_dim = (obs[2] + robot_radius) * s

            # Relative vectors (CasADi)
            p_rel = ca.vertcat(obs[0] - x[0, 0],
                               obs[1] - x[1, 0])
            v_rel = ca.vertcat(obs_vel_x - v * ca.cos(theta),
                               obs_vel_y - v * ca.sin(theta))

            rot_angle = ca.atan2(p_rel[1], p_rel[0])

            # Rotation matrix
            R = ca.vertcat(
                ca.horzcat( ca.cos(rot_angle), ca.sin(rot_angle)),
                ca.horzcat(-ca.sin(rot_angle), ca.cos(rot_angle))
            )

            v_rel_new = ca.mtimes(R, v_rel)

            p_rel_mag = ca.norm_2(p_rel)
            eps       = 1e-6
            # Clamp v_rel_mag to avoid division by zero when velocities match
            v_rel_mag = ca.fmax(ca.norm_2(v_rel), eps)
            d_safe    = ca.fmax(p_rel_mag**2 - ego_dim**2, eps)

            scale      = ca.sqrt(s**2 - 1) / ego_dim
            k_lam      = self.k_lambda * scale
            k_mu_s     = self.k_mu    * scale

            lam = k_lam * ca.sqrt(d_safe) / v_rel_mag
            mu  = k_mu_s * ca.sqrt(d_safe)

            h = v_rel_new[0] + lam * v_rel_new[1]**2 + mu
            return h

        h_k  = h_dpcbf(x_k,  obs, robot_radius, s)
        h_k1 = h_dpcbf(x_k1, obs, robot_radius, s)

        d_h = h_k1 - h_k

        return h_k, d_h