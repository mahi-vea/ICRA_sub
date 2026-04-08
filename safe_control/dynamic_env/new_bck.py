#!/usr/bin/env python3
"""
cbf_qp_ros_node.py
==================
ROS 1 node that ports the LocalTrackingControllerDyn obstacle-avoidance
logic to a real robot / Gazebo simulation.

Subscribes
----------
/tracked_objects      (std_msgs/Float32MultiArray)
    Flat array: [id, x, y, vx, vy,  id, x, y, vx, vy, ...]
    Positions are expected in base_link frame and are transformed
    to the world (odom) frame internally.

/front/scan           (sensor_msgs/LaserScan)
    LIDAR scan for static obstacles (walls, furniture, etc.).
    Points are converted to world frame and merged with tracked objects.

/odometry/filtered    (nav_msgs/Odometry)
    Robot state — x, y, yaw, v (linear), omega (angular).

Publishes
---------
/cmd_vel              (geometry_msgs/Twist)

Parameters (rosparam)
---------------------
~init_position   : [x, y, theta]   default [-11, 0, 3.14]
~goal_position   : [x, y]          default [31, 0]       (RELATIVE to init)
~controller_type : str             default 'cbf_qp'
~dt              : float            default 0.005
~obs_radius      : float            default 0.1
~robot_radius    : float            default 0.25
~v_max           : float            default 7.5
~w_max           : float            default 5.0
~a_max           : float            default 7.5
~num_constraints : int              default 5
~goal_tolerance  : float            default 0.5
"""

import math
import threading

import rospy
import numpy as np
import transforms3d.euler as t3d_euler

from std_msgs.msg      import Float32MultiArray
from nav_msgs.msg      import Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg   import LaserScan          # <-- was missing


# ---------------------------------------------------------------------------
# Main node
# ---------------------------------------------------------------------------

class CBFQPRosNode:

    def __init__(self):
        rospy.init_node('cbf_qp_avoidance_node', anonymous=False)

        # ------------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------------
        self.init_pos = rospy.get_param('~init_position', [-11.0, 0.0, 3.14])
        goal_rel      = rospy.get_param('~goal_position', [31.0, 0.0])

        # goal_position is RELATIVE to init — convert to absolute world frame
        self.goal = np.array([
            self.init_pos[0] + goal_rel[0],
            self.init_pos[1] + goal_rel[1],
        ], dtype=float)
        self.target_absolute_position = (self.goal[0], self.goal[1])

        self.dt      = rospy.get_param('~dt',              0.005)
        self.obs_r   = rospy.get_param('~obs_radius',      0.1)
        num_con      = rospy.get_param('~num_constraints',  5)
        goal_tol     = rospy.get_param('~goal_tolerance',   0.5)

        self.pos_controller_type = rospy.get_param('~controller_type', 'cbf_qp')

        robot_spec = {
            'model':  'DynamicUnicycle2D_DPCBF',
            'w_max':  rospy.get_param('~w_max',        5.0),
            'a_max':  rospy.get_param('~a_max',        7.5),
            'v_max':  rospy.get_param('~v_max',        7.5),
            'radius': rospy.get_param('~robot_radius', 0.25),
        }

        self.robot_spec      = robot_spec
        self.num_constraints = num_con
        self.goal_tol        = goal_tol

        # ------------------------------------------------------------------
        # Robot
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
        self.current_position = (x0, y0)
        self.current_heading  = 0.0
        self.odom_received    = False

        # ------------------------------------------------------------------
        # Obstacles  — separate arrays, merged at control time
        # ------------------------------------------------------------------
        self._obs_lock       = threading.Lock()
        self._static_obs     = np.empty((0, 5), dtype=float)   # from LaserScan
        self._dynamic_obs    = np.empty((0, 5), dtype=float)   # from tracked_objects

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
        rospy.Subscriber('/tracked_objects', Float32MultiArray,
                         self._tracked_obs_cb, queue_size=1)
        rospy.Subscriber('front/scan', LaserScan,
                         self._scan_cb, queue_size=1)

        # ------------------------------------------------------------------
        # Control loop timer
        # ------------------------------------------------------------------
        self.timer = rospy.Timer(rospy.Duration(self.dt), self._control_loop)

        rospy.loginfo(
            f"[cbf_qp_node] Ready.  controller={self.pos_controller_type}  "
            f"goal={self.goal}  obs_r={self.obs_r}  "
            f"init={self.init_pos}  goal_rel={goal_rel}"
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

        # Normalise yaw to [-pi, pi]
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
        self.current_heading  = yaw_deg
        self.odom_received    = True

        rospy.loginfo_throttle(1.0,
            f"[ODOM] x={x:.2f} y={y:.2f} yaw={yaw:.3f} v={v:.3f}")

    def _scan_cb(self, msg: LaserScan):
        """Convert LIDAR points to world-frame obstacle rows [x, y, r, 0, 0]."""
        obs_list = []
        cos_y = np.cos(self.robot.X[2, 0])
        sin_y = np.sin(self.robot.X[2, 0])
        rob_x = self.robot.X[0, 0]
        rob_y = self.robot.X[1, 0]

        DOWNSAMPLE = 5
        MAX_RANGE  = 4.0
        MIN_RANGE  = 0.05

        for i, r in enumerate(msg.ranges):
            if i % DOWNSAMPLE != 0:
                continue
            if not np.isfinite(r) or r < MIN_RANGE or r > MAX_RANGE:
                continue
            angle_i = msg.angle_min + i * msg.angle_increment
            bx = r * math.cos(angle_i)
            by = r * math.sin(angle_i)
            wx = rob_x + cos_y * bx - sin_y * by
            wy = rob_y + sin_y * bx + cos_y * by
            obs_list.append([wx, wy, self.obs_r, 0.0, 0.0])

        static = np.array(obs_list, dtype=float) if obs_list \
                 else np.empty((0, 5), dtype=float)

        with self._obs_lock:
            self._static_obs = static

    def _tracked_obs_cb(self, msg: Float32MultiArray):
        """Convert tracked-object detections to world-frame obstacle rows."""
        data = msg.data
        fields_per_obj = 5

        if len(data) == 0 or len(data) % fields_per_obj != 0:
            with self._obs_lock:
                self._dynamic_obs = np.empty((0, 5), dtype=float)
            return

        n_obs    = len(data) // fields_per_obj
        obs_list = []

        cos_y = np.cos(self.robot.X[2, 0])
        sin_y = np.sin(self.robot.X[2, 0])
        rob_x = self.robot.X[0, 0]
        rob_y = self.robot.X[1, 0]

        for i in range(n_obs):
            base         = i * fields_per_obj
            bx,  by  = data[base + 1], data[base + 2]
            bvx, bvy = data[base + 3], data[base + 4]

            wx  = rob_x + cos_y * bx  - sin_y * by
            wy  = rob_y + sin_y * bx  + cos_y * by
            wvx = cos_y * bvx - sin_y * bvy
            wvy = sin_y * bvx + cos_y * bvy

            obs_list.append([wx, wy, self.obs_r, wvx, wvy])

        dynamic = np.array(obs_list, dtype=float) if obs_list \
                  else np.empty((0, 5), dtype=float)

        with self._obs_lock:
            self._dynamic_obs = dynamic

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _get_merged_obs(self):
        """Return the combined static + dynamic obstacle array (thread-safe)."""
        with self._obs_lock:
            static  = self._static_obs
            dynamic = self._dynamic_obs

        parts = [p for p in (static, dynamic) if p.shape[0] > 0]
        if not parts:
            return np.empty((0, 5), dtype=float)
        return np.vstack(parts)

    def _get_nearest_obs(self):
        """Return the N closest obstacles from the merged set."""
        merged = self._get_merged_obs()
        if merged.shape[0] == 0:
            return None
        robot_xy = self.robot.X[:2, 0]
        dists    = np.linalg.norm(merged[:, :2] - robot_xy, axis=1)
        n        = min(self.num_constraints, len(dists))
        idx      = np.argsort(dists)[:n]
        return merged[idx]

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

        # 1. Nominal input toward goal
        if self.pos_controller_type == 'optimal_decay_cbf_qp':
            u_ref = self.robot.nominal_input(
                self.goal, k_omega=3.0, k_a=0.5, k_v=0.5)
        else:
            u_ref = self.robot.nominal_input(self.goal)

        # 2. CBF-QP safety filter
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
            'goal':           self.goal,
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

        # 3. Publish cmd_vel
        #    DynamicUnicycle2D_DPCBF: u = [acceleration, angular_velocity]
        #    cmd_vel expects linear velocity + angular velocity, so we
        #    integrate acceleration over dt and clip to [0, v_max].
        a_cmd = float(u[0, 0])
        w_cmd = float(u[1, 0])

        current_v = float(self.robot.X[3, 0])
        v_cmd = current_v + a_cmd * self.dt
        v_cmd = np.clip(v_cmd, 0.0, self.robot_spec['v_max'])
        w_cmd = np.clip(w_cmd, -self.robot_spec['w_max'],
                                 self.robot_spec['w_max'])

        cmd = Twist()
        cmd.linear.x  = v_cmd
        cmd.angular.z = w_cmd
        self.cmd_pub.publish(cmd)

        rospy.logdebug(
            f"[cbf_qp_node] a={a_cmd:.3f} v_cmd={v_cmd:.3f}  "
            f"w={w_cmd:.3f}  n_obs_static={self._static_obs.shape[0]}  "
            f"n_obs_dynamic={self._dynamic_obs.shape[0]}"
        )


# ---------------------------------------------------------------------------

if __name__ == '__main__':
    try:
        node = CBFQPRosNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass