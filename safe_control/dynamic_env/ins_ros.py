#!/usr/bin/env python3
"""
cbf_qp_ros_node.py
==================
ROS 1 node that ports the LocalTrackingControllerDyn obstacle-avoidance
logic to a real robot / Gazebo simulation.

Now integrates VFH* for local heading selection: the LIDAR scan is processed
through VFH* to pick a valley heading, which is converted into a short-range
waypoint fed to the CBF-QP controller.  The final goal position logic is
untouched — VFH* only steers the *local* target the QP tracks.

Subscribes
----------
/front/scan           (sensor_msgs/LaserScan)
    Used both for VFH* heading selection AND for CBF-QP obstacle array.

/odometry/filtered    (nav_msgs/Odometry)
    Robot state — x, y, yaw, v (linear), omega (angular).

Publishes
---------
/cmd_vel              (geometry_msgs/Twist)

Parameters (rosparam)
---------------------
~init_position   : [x, y, theta]   default [-2, 3, 1.57]
~goal_position   : [x, y]          default [0, 10]
~controller_type : str             default 'cbf_qp'
                                   options: cbf_qp | optimal_decay_cbf_qp | mpc_cbf
~dt              : float            default 0.01
~obs_radius      : float            default 0.15
~robot_radius    : float            default 0.25
~v_max           : float            default 1.0
~w_max           : float            default 0.5
~a_max           : float            default 0.5
~num_constraints : int              default 15
~goal_tolerance  : float            default 0.5
~waypoint_dist   : float            default 2.0   (VFH* waypoint lookahead)
"""

import math
import rospy
import numpy as np
import transforms3d.euler as t3d_euler

from std_msgs.msg      import Float32MultiArray
from nav_msgs.msg      import Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg   import LaserScan

# VFH* utilities — expected to sit next to this file
from navigation_utils import (
    calcDanger, calc_h, calc_hp, calc_Hb,
    calc_Target, vfh_star_full
)


# ---------------------------------------------------------------------------
# Main node
# ---------------------------------------------------------------------------

class CBFQPRosNode:

    def __init__(self):
        rospy.init_node('cbf_qp_avoidance_node', anonymous=False)

        # ------------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------------
        self.init_pos = rospy.get_param('~init_position', [-2.0, 3.0, 1.57])
        goal_rel      = rospy.get_param('~goal_position', [10.0, 0.0])
        self.target_absolute_position = (
            self.init_pos[0] + goal_rel[0],
            self.init_pos[1] + goal_rel[1],
        )

        self.dt    = rospy.get_param('~dt',             0.01)
        self.obs_r = rospy.get_param('~obs_radius',     0.01)
        num_con    = rospy.get_param('~num_constraints', 15)
        goal_tol   = rospy.get_param('~goal_tolerance',  0.5)

        self.pos_controller_type = rospy.get_param('~controller_type', 'cbf_qp')

        robot_spec = {
            'model':  'DynamicUnicycle2D',
            'w_max':  1.0,
            'a_max':  0.5,
            'v_max':  1.5,
            'radius': 0.2,
        }

        self.robot_spec      = robot_spec
        self.num_constraints = num_con
        self.goal_tol        = goal_tol

        # Final goal — only used for goal-reached check
        self.goal = np.array(rospy.get_param('~goal_position', [10.0, 0.0]),
                             dtype=float)

        # ------------------------------------------------------------------
        # VFH* parameters
        # ------------------------------------------------------------------
        self.max_range      = 4.0
        self.sector_size    = 8
        self.filter_width   = 3
        self.vfh_threshold  = 0.9
        self.robotDim       = 0.6
        self.WidevalleyMin  = 22
        self.vfh_ds         = self.robotDim   # projection step for VFH* search
        self.vfh_ng         = 2               # search depth

        # How far ahead (meters) to place the VFH* waypoint
        self.waypoint_dist = rospy.get_param('~waypoint_dist', 2.0)

        # Temporal smoothing for VFH* heading
        self.prev_heading = None
        self.alpha        = 0.5

        # Raw LIDAR storage (set by _scan_cb)
        self.processed_lidar_ranges = None

        # ------------------------------------------------------------------
        # Robot — offscreen matplotlib to satisfy BaseRobotDyn.__init__
        # ------------------------------------------------------------------
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        _fig, _ax = plt.subplots()

        from safe_control.dynamic_env.robot import BaseRobotDyn
        x0, y0, theta0 = self.init_pos
        X0 = np.array([[x0], [y0], [theta0], [0.0]])
        self.robot = BaseRobotDyn(X0, robot_spec, self.dt, ax=_ax)
        plt.close(_fig)
        self.odom_received    = False
        self.current_position = (x0, y0)
        self.current_heading  = 0.0        # degrees, for VFH*

        # ------------------------------------------------------------------
        # Obstacles — (n, 5): [x, y, r, vx, vy] in world frame
        # ------------------------------------------------------------------
        self.obs = np.empty((0, 5), dtype=float)

        # ------------------------------------------------------------------
        # Position controller
        # ------------------------------------------------------------------
        if self.pos_controller_type == 'cbf_qp':
            from position_control.cbf_qp import CBFQP
            self.pos_controller = CBFQP(self.robot, robot_spec, num_obs=num_con)

        elif self.pos_controller_type == 'optimal_decay_cbf_qp':
            from position_control.optimal_decay_cbf_qp import OptimalDecayCBFQP
            self.pos_controller = OptimalDecayCBFQP(
                self.robot, robot_spec, num_obs=num_con)

        elif self.pos_controller_type == 'mpc_cbf':
            from position_control.mpc_cbf import MPCCBF
            self.pos_controller = MPCCBF(self.robot, robot_spec, num_obs=num_con)

        else:
            raise ValueError(
                f"Unknown controller_type: '{self.pos_controller_type}'. "
                f"Choose from: cbf_qp | optimal_decay_cbf_qp | mpc_cbf")

        # ------------------------------------------------------------------
        # ROS I/O
        # ------------------------------------------------------------------
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

        rospy.Subscriber('/odometry/filtered', Odometry,
                         self._odom_cb, queue_size=1)
        rospy.Subscriber('front/scan', LaserScan,
                         self._scan_cb, queue_size=1)

        # ------------------------------------------------------------------
        # Control loop timer
        # ------------------------------------------------------------------
        self.timer = rospy.Timer(rospy.Duration(self.dt), self._control_loop)

        rospy.loginfo(
            f"[cbf_qp_node] Ready.  controller={self.pos_controller_type}  "
            f"goal={self.goal}  obs_r={self.obs_r}  "
            f"waypoint_dist={self.waypoint_dist}"
        )

    # -----------------------------------------------------------------------
    # Callbacks
    # -----------------------------------------------------------------------

    def _odom_cb(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        ori = msg.pose.pose.orientation
        q_ros = [ori.w, ori.x, ori.y, ori.z]
        _, _, yaw_rad = t3d_euler.quat2euler(q_ros, axes='sxyz')

        yaw_deg = math.degrees(yaw_rad)
        if yaw_deg < -180:
            yaw_deg += 360
        elif yaw_deg > 180:
            yaw_deg -= 360
        yaw = math.radians(yaw_deg)

        lin = msg.twist.twist.linear
        v   = np.hypot(lin.x, lin.y)

        self.robot.X[0, 0] = x
        self.robot.X[1, 0] = y
        self.robot.X[2, 0] = yaw
        self.robot.X[3, 0] = v
        self.current_position = (x, y)
        self.current_heading  = yaw_deg          # degrees for VFH*
        self.odom_received    = True

        rospy.loginfo_throttle(1.0,
            f"[ODOM] x={x:.2f} y={y:.2f} yaw={yaw:.3f} v={v:.3f}")

    def _scan_cb(self, msg: LaserScan):
        """
        1) Build obstacle array for CBF-QP  (world-frame points).
        2) Store processed LIDAR ranges for VFH*.
        """
        # ---- CBF-QP obstacle array (same logic as original) ----
        obs_list = []
        cos_y = np.cos(self.robot.X[2, 0])
        sin_y = np.sin(self.robot.X[2, 0])
        rob_x = self.robot.X[0, 0]
        rob_y = self.robot.X[1, 0]

        DOWNSAMPLE = 5
        MAX_RANGE  = 4.0
        MIN_RANGE  = 0.05

        for i, r in enumerate(msg.ranges):
            angle_i = msg.angle_min + i * msg.angle_increment
            if i % DOWNSAMPLE != 0:
                continue
            if not np.isfinite(r) or r < MIN_RANGE or r > MAX_RANGE:
                continue
            bx = r * math.cos(angle_i)
            by = r * math.sin(angle_i)
            wx = rob_x + cos_y * bx - sin_y * by
            wy = rob_y + sin_y * bx + cos_y * by
            obs_list.append([wx, wy, self.obs_r, 0.0, 0.0])

        self.obs = np.array(obs_list, dtype=float) if obs_list \
                   else np.empty((0, 5), dtype=float)

        # ---- VFH* LIDAR processing (matches NavigationNode) ----
        lidar_range = np.array(msg.ranges)
        lidar_ranges = np.flip(lidar_range)
        lidar_ranges[lidar_ranges > self.max_range] = self.max_range
        self.processed_lidar_ranges = lidar_ranges

    # -----------------------------------------------------------------------
    # VFH* local waypoint computation
    # -----------------------------------------------------------------------

    def _compute_vfh_waypoint(self):
        """
        Run VFH* on the current LIDAR data and return a local waypoint
        (x, y) in the world frame, placed `waypoint_dist` metres ahead
        along the chosen valley heading.

        Returns None if LIDAR data is not yet available.
        """
        if self.processed_lidar_ranges is None:
            return None

        # --- VFH* pipeline (identical to NavigationNode) ---
        m  = calcDanger(self.processed_lidar_ranges, self.max_range)
        h  = calc_h(m, self.sector_size)
        hp = calc_hp(h, self.filter_width)
        hb = calc_Hb(h, self.vfh_threshold)

        heading_sector = calc_Target(
            self.target_absolute_position,
            self.current_position,
            self.current_heading,          # degrees
        )

        prev_h = self.prev_heading if self.prev_heading is not None \
                 else heading_sector

        candidate_heading = vfh_star_full(
            self.current_position,
            self.current_heading,
            heading_sector,
            self.vfh_ds,
            self.vfh_ng,
            hb,
            self.vfh_threshold,
            self.robotDim,
            self.WidevalleyMin,
            prev_h,
        )

        # Temporal smoothing
        if self.prev_heading is None:
            smoothed = candidate_heading
        else:
            smoothed = (self.alpha * candidate_heading
                        + (1 - self.alpha) * self.prev_heading)
        self.prev_heading = smoothed

        # --- Convert VFH* sector heading to world-frame waypoint ---
        #
        # VFH* works in a sector index space 1..90 that maps to the
        # angular range [-135°, +135°] relative to the robot heading.
        # calc_Target returns:  Th_s = ((Th + 135) * 89 / 270) + 1
        # Invert:               Th   = (Th_s - 1) * 270 / 89  - 135
        #
        # Th is the *signed offset* (degrees) from current_heading where
        # positive = left of heading.  The robot's world-frame yaw is
        # current_heading (degrees, CCW from east / X-axis), so the
        # world-frame angle of the chosen direction is:
        #   world_angle = current_heading - Th
        #
        # (Minus because calc_Target defines positive Th as "target is
        # to the left", which in a standard frame means *adding* angle,
        # but calc_Target already flips the sign convention internally.)

        th_offset_deg = (smoothed - 1.0) * 270.0 / 89.0 - 135.0
        world_angle_deg = self.current_heading - th_offset_deg
        world_angle_rad = math.radians(world_angle_deg)

        rx, ry = self.current_position
        wp_x = rx + self.waypoint_dist * math.cos(world_angle_rad)
        wp_y = ry + self.waypoint_dist * math.sin(world_angle_rad)

        rospy.loginfo_throttle(1.0,
            f"[VFH*] sector={smoothed:.1f}  offset={th_offset_deg:.1f}°  "
            f"wp=({wp_x:.2f}, {wp_y:.2f})")

        return np.array([wp_x, wp_y], dtype=float)

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _get_nearest_obs(self):
        """Return the N closest obstacles as (n, 5) array, or None."""
        if self.obs.shape[0] == 0:
            return None
        robot_xy = self.robot.X[:2, 0]
        dists    = np.linalg.norm(self.obs[:, :2] - robot_xy, axis=1)
        n        = min(self.num_constraints, len(dists))
        idx      = np.argsort(dists)[:n]
        return self.obs[idx]

    def _goal_reached(self):
        distance_to_goal = math.hypot(
            self.target_absolute_position[0] - self.current_position[0],
            self.target_absolute_position[1] - self.current_position[1],
        )
        return distance_to_goal < self.goal_tol

    def _stop(self):
        self.cmd_pub.publish(Twist())

    # -----------------------------------------------------------------------
    # Control loop
    # -----------------------------------------------------------------------

    def _control_loop(self, event):
        if not self.odom_received:
            rospy.logwarn_throttle(2.0, "[cbf_qp_node] Waiting for odometry...")
            return

        if self._goal_reached():
            rospy.loginfo_once("[cbf_qp_node] Goal reached — stopping.")
            self._stop()
            return

        # ------------------------------------------------------------------
        # 0. Compute VFH* local waypoint (or fall back to final goal)
        # ------------------------------------------------------------------
        vfh_wp = self._compute_vfh_waypoint()
        local_goal = vfh_wp if vfh_wp is not None else self.goal

        # ------------------------------------------------------------------
        # 1. Nominal input toward the VFH* waypoint
        # ------------------------------------------------------------------
        if self.pos_controller_type == 'optimal_decay_cbf_qp':
            u_ref = self.robot.nominal_input(
                local_goal, k_omega=3.0, k_a=0.5, k_v=0.5)
        else:
            u_ref = self.robot.nominal_input(local_goal)

        # ------------------------------------------------------------------
        # 2. CBF-QP safety filter
        # ------------------------------------------------------------------
        nearest_obs = self._get_nearest_obs()

        bad_X   = not np.all(np.isfinite(self.robot.X))
        bad_u   = not np.all(np.isfinite(u_ref))
        bad_obs = (nearest_obs is not None and
                   not np.all(np.isfinite(nearest_obs)))
        if bad_X or bad_u or bad_obs:
            rospy.logwarn(
                f"[cbf_qp_node] NaN/Inf in QP inputs -- skipping tick.\n"
                f"  robot.X     = {self.robot.X.T}  bad={bad_X}\n"
                f"  u_ref       = {u_ref.T}          bad={bad_u}\n"
                f"  nearest_obs = {nearest_obs}       bad={bad_obs}"
            )
            self._stop()
            return

        control_ref = {
            'state_machine': 'track',
            'u_ref':          u_ref,
            'goal':           local_goal,      # VFH* waypoint
        }

        try:
            u = self.pos_controller.solve_control_problem(
                self.robot.X, control_ref, nearest_obs)
        except Exception as e:
            rospy.logerr(
                f"[cbf_qp_node] Controller failed: {e}\n"
                f"  robot.X     = {self.robot.X.T}\n"
                f"  u_ref       = {u_ref.T}\n"
                f"  nearest_obs = {nearest_obs}"
            )
            self._stop()
            return

        if self.pos_controller.status != 'optimal':
            rospy.logwarn_throttle(
                1.0, "[cbf_qp_node] QP infeasible — stopping for safety.")
            self._stop()
            return

        # ------------------------------------------------------------------
        # 3. Publish cmd_vel
        # ------------------------------------------------------------------
        # accel = float(u[0, 0])
        # omega = float(u[1, 0])

        # current_v = float(self.robot.X[3, 0])
        # desired_v = current_v + accel * self.dt
        # desired_v = np.clip(desired_v, 0.0, self.robot_spec['v_max'])

        # cmd = Twist()
        # cmd.linear.x  = desired_v
        # cmd.angular.z = np.clip(omega,
        #                         -self.robot_spec['w_max'],
        #                          self.robot_spec['w_max'])
        # self.cmd_pub.publish(cmd)
        cmd = Twist()
        cmd.linear.x  = float(np.clip(u[0, 0],
                                       0.0, self.robot_spec['v_max']))
        cmd.angular.z = float(np.clip(u[1, 0],
                                       -self.robot_spec['w_max'],
                                        self.robot_spec['w_max']))
        self.cmd_pub.publish(cmd)

        rospy.logdebug(
            f"[cbf_qp_node] v={cmd.linear.x:.3f}  w={cmd.angular.z:.3f}  "
            f"n_obs={self.obs.shape[0]}"
        )



# ---------------------------------------------------------------------------

if __name__ == '__main__':
    try:
        node = CBFQPRosNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass