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
~dt              : float            default 0.05
~obs_radius      : float            default 0.5
~robot_radius    : float            default 0.25
~v_max           : float            default 1.0
~w_max           : float            default 0.5
~a_max           : float            default 0.5
~num_constraints : int              default 3
~goal_tolerance  : float            default 0.3
"""

import math
import rospy
import numpy as np
import transforms3d.euler as t3d_euler

from std_msgs.msg      import Float32MultiArray
from nav_msgs.msg      import Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan


# ---------------------------------------------------------------------------
# Main node
# ---------------------------------------------------------------------------

class CBFQPRosNode:

    def __init__(self):
        rospy.init_node('cbf_qp_avoidance_node', anonymous=False)

        # ------------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------------
        self.init_pos = rospy.get_param('~init_position', [2.0,-3.0,-1.57])
        goal_rel = rospy.get_param('~goal_position', [0.0, 10.0])
        self.target_absolute_position = (
            self.init_pos[0] + goal_rel[0],
            self.init_pos[1] + goal_rel[1],
        )
        

        self.dt      = rospy.get_param('~dt',              0.01)
        self.obs_r   = rospy.get_param('~obs_radius',      0.15)
        num_con      = rospy.get_param('~num_constraints', 15)
        goal_tol     = rospy.get_param('~goal_tolerance',  0.5)

        # mirrors the original: controller_type={'pos': 'cbf_qp'}
        self.pos_controller_type = rospy.get_param('~controller_type', 'cbf_qp')

        # robot_spec = {
        #     'model':  'DynamicUnicycle2D_DPCBF',
        #     'w_max':  rospy.get_param('~w_max',        2.0),
        #     'a_max':  rospy.get_param('~a_max',        2.0),
        #     'v_max':  rospy.get_param('~v_max',        5.0),
        #     'radius': rospy.get_param('~robot_radius', 0.5),
        # }

        robot_spec = {
            'model': 'DynamicUnicycle2D',
            'w_max': 0.5,
            'a_max': 0.5,
            'v_max': 1.0,      # ← ADD THIS — was missing, causes KeyError at publish
            'radius': 0.25
        }

        self.robot_spec      = robot_spec
        self.num_constraints = num_con
        self.goal_tol        = goal_tol
        # self.goal            = np.array(goal_pos, dtype=float)
        # self.goal = np.array(self.target_absolute_position, dtype=float)
        self.goal = np.array(rospy.get_param('~goal_position', [10.0, 0.0]), dtype=float)


        # ------------------------------------------------------------------
        # Robot — use BaseRobotDyn so the controller gets all dynamics
        # methods (f, g, dt, nominal_input ...).
        # BaseRobot.__init__ unconditionally calls ax.add_patch, so we
        # create a throwaway offscreen figure to satisfy it, then close it.
        # robot.X is overwritten from odometry every tick — Gazebo is truth.
        # ------------------------------------------------------------------
        import matplotlib
        matplotlib.use('Agg')          # non-interactive: no window ever opens
        import matplotlib.pyplot as plt
        _fig, _ax = plt.subplots()     # throwaway axes — never shown/saved

        from safe_control.dynamic_env.robot import BaseRobotDyn
        x0, y0, theta0 = self.init_pos
        X0 = np.array([[x0], [y0], [theta0], [0.0]])  # [x, y, theta, v]
        self.robot = BaseRobotDyn(X0, robot_spec, self.dt, ax=_ax)
        plt.close(_fig)                # free memory immediately
        self.odom_received = False

        # ------------------------------------------------------------------
        # Obstacles — shape (n, 5): [x, y, r, vx, vy] in world (odom) frame.
        # step_dyn_obs() is NOT used — live positions come from the tracker.
        # ------------------------------------------------------------------
        self.obs = np.empty((0, 5), dtype=float)

        # ------------------------------------------------------------------
        # Position controller — mirrors LocalTrackingController selection
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
        # rospy.Subscriber('/tracked_objects', Float32MultiArray,
        #                  self._tracked_obs_cb, queue_size=1)
        rospy.Subscriber('front/scan', LaserScan, self._scan_cb, queue_size=1)

        # ------------------------------------------------------------------
        # Control loop timer
        # ------------------------------------------------------------------
        self.timer = rospy.Timer(rospy.Duration(self.dt), self._control_loop)

        rospy.loginfo(
            f"[cbf_qp_node] Ready.  controller={self.pos_controller_type}  "
            f"goal={self.goal}  obs_r={self.obs_r}"
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
        # yaw = (yaw + math.pi) % (2 * math.pi) - math.pi


        yaw_deg = math.degrees(yaw_rad)
        if yaw_deg < -180:
            yaw_deg += 360
        elif yaw_deg > 180:
            yaw_deg -= 360
        yaw = math.radians(yaw_deg)

        # ← Remove the init_pos offset entirely
        lin = msg.twist.twist.linear
        v   = np.hypot(lin.x, lin.y)

        self.robot.X[0, 0] = x
        self.robot.X[1, 0] = y
        self.robot.X[2, 0] = yaw
        self.robot.X[3, 0] = v
        self.current_position = (x, y)
        self.odom_received = True

        rospy.loginfo_throttle(1.0,
        f"[ODOM] x={x:.2f} y={y:.2f} yaw={yaw:.3f} v={v:.3f}")

    def _tracked_obs_cb(self, msg: Float32MultiArray):
        """
        Parse /tracked_objects flat array:
            [id, x, y, vx, vy,  id, x, y, vx, vy, ...]
        Positions arrive in base_link frame; rotate to odom frame using
        the robot's current yaw from odometry (no tf library needed).
        """
        data = msg.data
        fields_per_obj = 5  # id, x, y, vx, vy

        if len(data) == 0 or len(data) % fields_per_obj != 0:
            self.obs = np.empty((0, 5), dtype=float)
            return

        n_obs    = len(data) // fields_per_obj
        obs_list = []

        # R(theta): base_link -> odom
        # p_world = R @ p_body + robot_pos
        cos_y = np.cos(self.robot.X[2, 0])
        sin_y = np.sin(self.robot.X[2, 0])
        rob_x = self.robot.X[0, 0]
        rob_y = self.robot.X[1, 0]

        for i in range(n_obs):
            base         = i * fields_per_obj
            # data[base] = object id — unused
            bx,  by  = data[base + 1], data[base + 2]
            bvx, bvy = data[base + 3], data[base + 4]

            # Rotate position
            wx  = rob_x + cos_y * bx  - sin_y * by
            wy  = rob_y + sin_y * bx  + cos_y * by
            # Rotate velocity
            wvx = cos_y * bvx - sin_y * bvy
            wvy = sin_y * bvx + cos_y * bvy

            obs_list.append([wx, wy, self.obs_r, wvx, wvy])

        self.obs = np.array(obs_list, dtype=float) if obs_list \
                   else np.empty((0, 5), dtype=float)

    def _scan_cb(self, msg: LaserScan):
        """
        Convert raw laser scan to obstacle array (n, 5): [x, y, r, vx, vy]
        in world (odom) frame. Downsample to keep QP tractable.
        """
        obs_list = []
        angle = msg.angle_min
        cos_y = np.cos(self.robot.X[2, 0])
        sin_y = np.sin(self.robot.X[2, 0])
        rob_x = self.robot.X[0, 0]
        rob_y = self.robot.X[1, 0]

        # Downsample: only take every Nth beam
        DOWNSAMPLE = 5          # tune this
        MAX_RANGE  = 4.0        # ignore far points (meters)
        MIN_RANGE  = 0.05       # ignore points too close (sensor noise)

        for i, r in enumerate(msg.ranges):
            angle_i = msg.angle_min + i * msg.angle_increment
            if i % DOWNSAMPLE != 0:
                continue
            if not np.isfinite(r) or r < MIN_RANGE or r > MAX_RANGE:
                continue

            # Point in base_link frame
            bx = r * math.cos(angle_i)
            by = r * math.sin(angle_i)

            # Rotate to world frame
            wx = rob_x + cos_y * bx - sin_y * by
            wy = rob_y + sin_y * bx + cos_y * by

            obs_list.append([wx, wy, self.obs_r, 0.0, 0.0])  # static: vx=vy=0

        self.obs = np.array(obs_list, dtype=float) if obs_list \
                else np.empty((0, 5), dtype=float)

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
        # 1. Nominal input toward goal
        # ------------------------------------------------------------------
        if self.pos_controller_type == 'optimal_decay_cbf_qp':
            u_ref = self.robot.nominal_input(
                self.goal, k_omega=3.0, k_a=0.5, k_v=0.5)
        else:
            u_ref = self.robot.nominal_input(self.goal)

        # ------------------------------------------------------------------
        # 2. CBF-QP safety filter
        # ------------------------------------------------------------------
        nearest_obs = self._get_nearest_obs()

        # --- NaN/Inf guard: log what is bad before handing to QP ----------
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

        # ------------------------------------------------------------------
        # 3. Publish cmd_vel
        #    For DynamicUnicycle2D_DPCBF: u = [a (linear accel), w (angular vel)]
        #    We send v directly — the real robot's low-level controller
        #    handles its own acceleration loop.
        # ------------------------------------------------------------------
        accel = float(u[0, 0])
        omega = float(u[1, 0])

        # Integrate accel to get velocity command
        current_v = float(self.robot.X[3, 0])
        desired_v = current_v + accel * self.dt
        desired_v = np.clip(desired_v, 0.0, self.robot_spec['v_max'])

        cmd = Twist()
        cmd.linear.x  = desired_v
        cmd.angular.z = np.clip(omega, -self.robot_spec['w_max'], self.robot_spec['w_max'])
        self.cmd_pub.publish(cmd)


# ---------------------------------------------------------------------------

if __name__ == '__main__':
    try:
        node = CBFQPRosNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass