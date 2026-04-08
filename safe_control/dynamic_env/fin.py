#!/usr/bin/env python3
"""
cbf_qp_ros_node.py
==================
Unified ROS 1 node that ports the LocalTrackingControllerDyn obstacle-avoidance
logic to a real robot / Gazebo simulation.

Automatically selects **static** or **dynamic** environment mode:
  - static  (init_position == [-2, 3, 1.57]):  Uses VFH* for local heading
    selection from LIDAR, producing a short-range waypoint for the CBF-QP
    controller.
  - dynamic (any other init_position):  Uses tracked-object detections
    (with velocities) merged with LIDAR static obstacles; drives directly
    toward the goal with CBF-QP safety filtering.

Subscribes
----------
/front/scan           (sensor_msgs/LaserScan)
    LIDAR scan — used in both modes for static obstacle extraction.
    In static mode also feeds the VFH* heading selector.

/odometry/filtered    (nav_msgs/Odometry)
    Robot state — x, y, yaw, v (linear), omega (angular).
    Assumed to start at (0, 0, 0) in the odom frame.

/tracked_objects      (std_msgs/Float32MultiArray)   [dynamic mode only]
    Flat array: [id, x, y, vx, vy,  id, x, y, vx, vy, ...]
    Positions in base_link frame; transformed to world frame internally.

Publishes
---------
/cmd_vel              (geometry_msgs/Twist)

Parameters (rosparam)
---------------------
~init_position   : [x, y, theta]   default [-2, 3, 1.57]
~goal_position   : [x, y]          default [10, 0]  (RELATIVE to init, in robot's
                                    initial heading frame — i.e. "10m forward")
~goal_theta      : float           default 0.0  (desired final heading in world, rad)
~heading_tolerance : float         default 0.15 (~8.6 degrees)
~controller_type : str             default 'cbf_qp'
                                   options: cbf_qp | optimal_decay_cbf_qp | mpc_cbf
~dt              : float            default 0.01  (static) / 0.005 (dynamic)
~obs_radius      : float            default 0.01  (static) / 0.1   (dynamic)
~robot_radius    : float            default 0.25
~v_max           : float            default 1.5   (static) / 7.5   (dynamic)
~w_max           : float            default 1.0   (static) / 5.0   (dynamic)
~a_max           : float            default 0.5   (static) / 7.5   (dynamic)
~num_constraints : int              default 15    (static) / 5     (dynamic)
~goal_tolerance  : float            default 0.5
~waypoint_dist   : float            default 2.0   (VFH* lookahead, static only)
~reverse_speed   : float            default 0.3   (reverse speed on infeasible)
~reverse_duration: float            default 1.0   (seconds to reverse)
~reverse_omega   : float            default 0.5   (turn rate while reversing)
~rotate_k_omega  : float            default 2.0   (P-gain for rotate-in-place)
"""

import math
import threading
import random

import rospy
import numpy as np
import transforms3d.euler as t3d_euler

from std_msgs.msg      import Float32MultiArray
from nav_msgs.msg      import Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg   import LaserScan


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def normalize_angle(angle):
    """Wrap angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


# ---------------------------------------------------------------------------
# Main node
# ---------------------------------------------------------------------------

class CBFQPRosNode:

    # ===================================================================
    # Initialisation
    # ===================================================================

    def __init__(self):
        rospy.init_node('cbf_qp_avoidance_node', anonymous=False)

        # ------------------------------------------------------------------
        # Determine environment mode
        # ------------------------------------------------------------------
        self.init_pos = rospy.get_param('~init_position', [-2.0, 3.0, 1.57])
        self.environment_mode = (
            'static' if self.init_pos == [-2, 3, 1.57] else 'dynamic'
        )
        rospy.loginfo(f"[cbf_qp_node] Environment mode: {self.environment_mode}")

        # ------------------------------------------------------------------
        # Mode-dependent defaults
        # ------------------------------------------------------------------
        if self.environment_mode == 'static':
            default_goal    = [0.0, -10.0]
            default_dt      = 0.01
            default_obs_r   = 0.01
            default_v_max   = 2.5
            default_w_max   = 1.0
            default_a_max   = 1.0
            default_num_con = 15
            default_model   = 'DynamicUnicycle2D'
        else:  # dynamic
            default_goal    = [20.0, 0.0]
            default_dt      = 0.005
            default_obs_r   = 0.1
            default_v_max   = 7.5
            default_w_max   = 5.0
            default_a_max   = 7.5
            default_num_con = 5
            default_model   = 'DynamicUnicycle2D_DPCBF'

        # ------------------------------------------------------------------
        # Parameters (user overrides win; otherwise mode defaults apply)
        # ------------------------------------------------------------------
        goal_rel = rospy.get_param('~goal_position', default_goal)

        # *** FIX: rotate goal_rel by the initial heading ***
        # goal_rel is in the robot's initial body frame (e.g. [10, 0] means
        # "10 m forward").  We rotate it by init_pos[2] (theta) to get the
        # displacement in the world frame, then add init_pos XY.
        theta0 = self.init_pos[2]
        cos_t  = math.cos(theta0)
        sin_t  = math.sin(theta0)
        goal_world_x = self.init_pos[0] + cos_t * goal_rel[0] - sin_t * goal_rel[1]
        goal_world_y = self.init_pos[1] + sin_t * goal_rel[0] + cos_t * goal_rel[1]

        self.goal = np.array([goal_world_x, goal_world_y], dtype=float)
        self.target_absolute_position = (goal_world_x, goal_world_y)

        rospy.loginfo(
            f"[cbf_qp_node] goal_rel={goal_rel}  init_theta={theta0:.3f} rad  "
            f"=> world goal=({goal_world_x:.2f}, {goal_world_y:.2f})")

        # Goal heading
        self.goal_theta     = rospy.get_param('~goal_theta', 0.0)
        self.heading_tol    = rospy.get_param('~heading_tolerance', 0.15)  # ~8.6 deg
        self.rotate_k_omega = rospy.get_param('~rotate_k_omega', 2.0)

        self.dt    = rospy.get_param('~dt',              default_dt)
        self.obs_r = rospy.get_param('~obs_radius',      default_obs_r)
        num_con    = rospy.get_param('~num_constraints',  default_num_con)
        goal_tol   = rospy.get_param('~goal_tolerance',   0.5)

        self.pos_controller_type = rospy.get_param('~controller_type', 'cbf_qp')

        robot_spec = {
            'model':  default_model,
            'w_max':  rospy.get_param('~w_max',        default_w_max),
            'a_max':  rospy.get_param('~a_max',        default_a_max),
            'v_max':  rospy.get_param('~v_max',        default_v_max),
            'radius': rospy.get_param('~robot_radius', 0.25 if self.environment_mode == 'dynamic' else 0.2),
        }

        self.robot_spec      = robot_spec
        self.num_constraints = num_con
        self.goal_tol        = goal_tol

        # ------------------------------------------------------------------
        # VFH* parameters (static mode only — but harmless to define)
        # ------------------------------------------------------------------
        self.max_range      = 4.0
        self.sector_size    = 8
        self.filter_width   = 3
        self.vfh_threshold  = 0.9
        self.robotDim       = 0.6
        self.WidevalleyMin  = 22
        self.vfh_ds         = self.robotDim
        self.vfh_ng         = 2
        self.waypoint_dist  = rospy.get_param('~waypoint_dist', 2.0)
        self.prev_heading   = None
        self.alpha          = 0.5
        self.processed_lidar_ranges = None

        # Import VFH* utilities only when needed
        if self.environment_mode == 'static':
            from navigation_utils import (
                calcDanger, calc_h, calc_hp, calc_Hb,
                calc_Target, vfh_star_full,
            )
            self._vfh_calcDanger    = calcDanger
            self._vfh_calc_h        = calc_h
            self._vfh_calc_hp       = calc_hp
            self._vfh_calc_Hb       = calc_Hb
            self._vfh_calc_Target   = calc_Target
            self._vfh_star_full     = vfh_star_full

        # ------------------------------------------------------------------
        # Robot — offscreen matplotlib to satisfy BaseRobotDyn.__init__
        # ------------------------------------------------------------------
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        _fig, _ax = plt.subplots()

        from safe_control.dynamic_env.robot import BaseRobotDyn
        x0, y0 = self.init_pos[0], self.init_pos[1]
        X0 = np.array([[x0], [y0], [theta0], [0.0]])
        self.robot = BaseRobotDyn(X0, robot_spec, self.dt, ax=_ax)
        plt.close(_fig)

        self.current_position = (x0, y0)
        self.current_heading  = math.degrees(theta0)   # degrees, for VFH*
        self.current_yaw      = theta0                  # radians
        self.odom_received    = False

        # ------------------------------------------------------------------
        # State machine: 'navigate' -> 'rotate' -> 'done'
        # ------------------------------------------------------------------
        self.state = 'navigate'

        # ------------------------------------------------------------------
        # Reverse-on-infeasible state
        # ------------------------------------------------------------------
        self.reverse_speed    = rospy.get_param('~reverse_speed', 0.3)
        self.reverse_duration = rospy.get_param('~reverse_duration', 1.0)
        self.reverse_omega    = rospy.get_param('~reverse_omega', 0.5)
        self.reverse_start    = None   # rospy.Time when reverse began
        self.reverse_turn_dir = 1.0    # +1 or -1, randomised each time
        self.infeasible_count = 0      # consecutive infeasible ticks

        # ------------------------------------------------------------------
        # Obstacles
        # ------------------------------------------------------------------
        self._obs_lock    = threading.Lock()
        self._static_obs  = np.empty((0, 5), dtype=float)   # from LaserScan
        self._dynamic_obs = np.empty((0, 5), dtype=float)   # from tracked_objects

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

        # Dynamic-only subscriber
        if self.environment_mode == 'dynamic':
            rospy.Subscriber('/tracked_objects', Float32MultiArray,
                             self._tracked_obs_cb, queue_size=1)

        # ------------------------------------------------------------------
        # Control loop timer
        # ------------------------------------------------------------------
        self.timer = rospy.Timer(rospy.Duration(self.dt), self._control_loop)

        rospy.loginfo(
            f"[cbf_qp_node] Ready.  mode={self.environment_mode}  "
            f"controller={self.pos_controller_type}  "
            f"goal=({goal_world_x:.2f}, {goal_world_y:.2f})  "
            f"goal_theta={self.goal_theta:.3f} rad  "
            f"obs_r={self.obs_r}  init={self.init_pos}"
        )

    # ===================================================================
    # Callbacks
    # ===================================================================

    def _odom_cb(self, msg: Odometry):
        x = msg.pose.pose.position.x + self.init_pos[0]
        y = msg.pose.pose.position.y + self.init_pos[1]

        ori = msg.pose.pose.orientation
        q_ros = [ori.w, ori.x, ori.y, ori.z]
        _, _, yaw_rad = t3d_euler.quat2euler(q_ros, axes='sxyz')

        # Normalise yaw to [-pi, pi]
        yaw = normalize_angle(yaw_rad)

        lin = msg.twist.twist.linear
        v   = np.hypot(lin.x, lin.y)

        self.robot.X[0, 0] = x
        self.robot.X[1, 0] = y
        self.robot.X[2, 0] = yaw
        self.robot.X[3, 0] = v
        self.current_position = (x, y)
        self.current_yaw      = yaw                        # radians
        self.current_heading  = math.degrees(yaw)           # degrees for VFH*
        self.odom_received    = True

        rospy.loginfo_throttle(1.0,
            f"[ODOM] x={x:.2f} y={y:.2f} yaw={yaw:.3f} v={v:.3f}")

    def _scan_cb(self, msg: LaserScan):
        """
        Convert LIDAR points to world-frame obstacle rows [x, y, r, 0, 0].
        In static mode, also store processed ranges for VFH*.
        """
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

        # VFH* LIDAR processing (static mode only)
        if self.environment_mode == 'static':
            lidar_range  = np.array(msg.ranges)
            lidar_ranges = np.flip(lidar_range)
            lidar_ranges[lidar_ranges > self.max_range] = self.max_range
            self.processed_lidar_ranges = lidar_ranges

    def _tracked_obs_cb(self, msg: Float32MultiArray):
        """Convert tracked-object detections to world-frame obstacle rows.
        (Dynamic mode only — subscriber is not created in static mode.)
        """
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
            base     = i * fields_per_obj
            bx, by   = data[base + 1], data[base + 2]
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

    # ===================================================================
    # VFH* local waypoint (static mode only)
    # ===================================================================

    def _compute_vfh_waypoint(self):
        """
        Run VFH* on the current LIDAR data and return a local waypoint
        (x, y) in the world frame, placed `waypoint_dist` metres ahead
        along the chosen valley heading.

        Returns None if LIDAR data is not yet available.
        """
        if self.processed_lidar_ranges is None:
            return None

        m  = self._vfh_calcDanger(self.processed_lidar_ranges, self.max_range)
        h  = self._vfh_calc_h(m, self.sector_size)
        hp = self._vfh_calc_hp(h, self.filter_width)
        hb = self._vfh_calc_Hb(h, self.vfh_threshold)

        heading_sector = self._vfh_calc_Target(
            self.target_absolute_position,
            self.current_position,
            self.current_heading,          # degrees
        )

        prev_h = self.prev_heading if self.prev_heading is not None \
                 else heading_sector

        candidate_heading = self._vfh_star_full(
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

        # Convert VFH* sector heading to world-frame waypoint
        th_offset_deg   = (smoothed - 1.0) * 270.0 / 89.0 - 135.0
        world_angle_deg = self.current_heading - th_offset_deg
        world_angle_rad = math.radians(world_angle_deg)

        rx, ry = self.current_position
        wp_x = rx + self.waypoint_dist * math.cos(world_angle_rad)
        wp_y = ry + self.waypoint_dist * math.sin(world_angle_rad)

        rospy.loginfo_throttle(1.0,
            f"[VFH*] sector={smoothed:.1f}  offset={th_offset_deg:.1f}°  "
            f"wp=({wp_x:.2f}, {wp_y:.2f})")

        return np.array([wp_x, wp_y], dtype=float)

    # ===================================================================
    # Helpers
    # ===================================================================

    def _get_nearest_obs(self):
        """Return the N closest obstacles from the merged static+dynamic set."""
        with self._obs_lock:
            static  = self._static_obs
            dynamic = self._dynamic_obs

        parts = [p for p in (static, dynamic) if p.shape[0] > 0]
        if not parts:
            return None

        merged   = np.vstack(parts)
        robot_xy = self.robot.X[:2, 0]
        dists    = np.linalg.norm(merged[:, :2] - robot_xy, axis=1)
        n        = min(self.num_constraints, len(dists))
        idx      = np.argsort(dists)[:n]
        return merged[idx]

    def _position_reached(self):
        """Check if the robot is within goal_tol of the goal position."""
        dist = math.hypot(
            self.target_absolute_position[0] - self.current_position[0],
            self.target_absolute_position[1] - self.current_position[1],
        )
        return dist < self.goal_tol

    def _heading_reached(self):
        """Check if the robot heading is within heading_tol of goal_theta."""
        heading_err = normalize_angle(self.goal_theta - self.current_yaw)
        return abs(heading_err) < self.heading_tol

    def _stop(self):
        self.cmd_pub.publish(Twist())

    def _reverse(self):
        """Publish a reverse + slight turn command to back away from obstacles."""
        cmd = Twist()
        cmd.linear.x  = -self.reverse_speed
        cmd.angular.z = self.reverse_omega * self.reverse_turn_dir
        self.cmd_pub.publish(cmd)

    def _is_reversing(self):
        """Check if we are currently in a reverse manoeuvre."""
        if self.reverse_start is None:
            return False
        elapsed = (rospy.Time.now() - self.reverse_start).to_sec()
        if elapsed < self.reverse_duration:
            return True
        # Reverse finished — reset state
        self.reverse_start    = None
        self.infeasible_count = 0
        return False

    def _begin_reverse(self):
        """Start a timed reverse manoeuvre with a random turn direction."""
        self.reverse_start    = rospy.Time.now()
        self.reverse_turn_dir = random.choice([-1.0, 1.0])
        rospy.logwarn(
            f"[cbf_qp_node] Starting reverse manoeuvre "
            f"(v={-self.reverse_speed:.2f}, w_dir={self.reverse_turn_dir:+.0f}, "
            f"duration={self.reverse_duration:.1f}s)")

    def _rotate_in_place(self):
        """
        Publish a pure rotation command to align with goal_theta.
        Uses proportional control with clamping.
        """
        heading_err = normalize_angle(self.goal_theta - self.current_yaw)

        omega = self.rotate_k_omega * heading_err
        omega = float(np.clip(omega,
                              -self.robot_spec['w_max'],
                               self.robot_spec['w_max']))

        cmd = Twist()
        cmd.linear.x  = 0.0
        cmd.angular.z = omega
        self.cmd_pub.publish(cmd)

        rospy.loginfo_throttle(0.5,
            f"[ROTATE] heading_err={math.degrees(heading_err):.1f}°  "
            f"omega={omega:.3f}")

    # ===================================================================
    # Control loop
    # ===================================================================

    def _control_loop(self, event):
        if not self.odom_received:
            rospy.logwarn_throttle(2.0, "[cbf_qp_node] Waiting for odometry...")
            return

        # ==================================================================
        # State machine: navigate -> rotate -> done
        # ==================================================================

        # --- DONE ---
        if self.state == 'done':
            self._stop()
            return

        # --- ROTATE (position reached, aligning heading) ---
        if self.state == 'rotate':
            if self._heading_reached():
                rospy.loginfo("[cbf_qp_node] Goal pose reached (position + heading) — stopping.")
                self.state = 'done'
                self._stop()
                return
            self._rotate_in_place()
            return

        # --- NAVIGATE (driving toward goal position) ---
        # Check if position is reached -> transition to rotate
        if self._position_reached():
            rospy.loginfo(
                f"[cbf_qp_node] Position reached at "
                f"({self.current_position[0]:.2f}, {self.current_position[1]:.2f}). "
                f"Switching to rotate-in-place "
                f"(goal_theta={math.degrees(self.goal_theta):.1f}°)")
            self.state = 'rotate'
            if self._heading_reached():
                rospy.loginfo("[cbf_qp_node] Goal pose reached (position + heading) — stopping.")
                self.state = 'done'
                self._stop()
                return
            self._rotate_in_place()
            return

        # If we are in the middle of a reverse manoeuvre, keep reversing
        if self._is_reversing():
            self._reverse()
            return

        # ------------------------------------------------------------------
        # 0. Determine local goal
        # ------------------------------------------------------------------
        if self.environment_mode == 'static':
            vfh_wp     = self._compute_vfh_waypoint()
            local_goal = vfh_wp if vfh_wp is not None else self.goal
        else:
            local_goal = self.goal

        # ------------------------------------------------------------------
        # 1. Nominal input toward the goal / VFH* waypoint
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
            'goal':           local_goal,
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
            self.infeasible_count += 1
            rospy.logwarn_throttle(
                0.5,
                f"[cbf_qp_node] QP infeasible (count={self.infeasible_count}) "
                f"— reversing to escape.")
            self._begin_reverse()
            self._reverse()
            return

        # QP was feasible — reset infeasible counter
        self.infeasible_count = 0

        # ------------------------------------------------------------------
        # 3. Publish cmd_vel
        # ------------------------------------------------------------------
        if self.environment_mode == 'static':
            # Static model outputs [v, omega] directly
            cmd = Twist()
            cmd.linear.x  = float(np.clip(u[0, 0],
                                           0.0, self.robot_spec['v_max']))
            cmd.angular.z = float(np.clip(u[1, 0],
                                           -self.robot_spec['w_max'],
                                            self.robot_spec['w_max']))
            self.cmd_pub.publish(cmd)
        else:
            # Dynamic model outputs [acceleration, omega] — integrate to get v
            a_cmd = float(u[0, 0])
            w_cmd = float(u[1, 0])

            current_v = float(self.robot.X[3, 0])
            v_cmd = current_v + a_cmd * self.dt
            v_cmd = float(np.clip(v_cmd, 0.0, self.robot_spec['v_max']))
            w_cmd = float(np.clip(w_cmd, -self.robot_spec['w_max'],
                                          self.robot_spec['w_max']))

            cmd = Twist()
            cmd.linear.x  = v_cmd
            cmd.angular.z = w_cmd
            self.cmd_pub.publish(cmd)

        rospy.logdebug(
            f"[cbf_qp_node] mode={self.environment_mode}  "
            f"v={cmd.linear.x:.3f}  w={cmd.angular.z:.3f}  "
            f"n_obs_static={self._static_obs.shape[0]}  "
            f"n_obs_dynamic={self._dynamic_obs.shape[0]}"
        )


# ---------------------------------------------------------------------------

if __name__ == '__main__':
    try:
        node = CBFQPRosNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass