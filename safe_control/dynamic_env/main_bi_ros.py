#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ROS1 (Melodic) wrapper for LocalTrackingControllerDyn.

Subscribes:
    /tracked_objects     (std_msgs/Float32MultiArray)
        Format: [id, x, y, vx, vy,  id, x, y, vx, vy, ...]
        Arrives in base_link frame — transformed to world frame using
        the latest odometry pose (no tf2 required).

    /odometry/filtered   (nav_msgs/Odometry)
        Ground-truth robot pose + velocity; synced into the controller
        every control tick.

Publishes:
    /cmd_vel             (geometry_msgs/Twist)
        Linear.x  = longitudinal velocity [m/s]
        Angular.z = yaw rate              [rad/s]

ROS params (all under private namespace ~):
    init_position   [x, y, theta]   default: [-2.0, 3.0, 1.57]
    goal_position   [x, y]          default: [0.0, 10.0]
    dt              float           default: 0.05
    model           string          default: 'KinematicBicycle2D_DPCBF'
"""

import math
import threading
import numpy as np
import rospy

from std_msgs.msg import Float32MultiArray
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
# import tf.transformations as tft

# ── project imports (unchanged) ──────────────────────────────────────────────
from safe_control.tracking import LocalTrackingController


# ─────────────────────────────────────────────────────────────────────────────
# Minimal stub that replaces matplotlib-dependent BaseRobotDyn
# We keep the real robot class but silence its render calls.
# ─────────────────────────────────────────────────────────────────────────────
class InfeasibleError(Exception):
    def __init__(self, message="ERROR in QP or Collision"):
        self.message = message
        super().__init__(self.message)


# ─────────────────────────────────────────────────────────────────────────────
# ROS-aware controller
# ─────────────────────────────────────────────────────────────────────────────
class LocalTrackingControllerROS(LocalTrackingController):
    """
    Subclass of LocalTrackingController that:
      - removes all matplotlib rendering
      - receives obstacle data from /tracked_objects
      - receives robot state from /odometry/filtered
      - publishes cmd_vel
    """

    def __init__(self, X0, robot_spec,
                 controller_type=None,
                 dt=0.05,
                 raise_error=False):

        # Pass dummy ax/fig=None; parent may try to store them — that's fine,
        # we just never call anything that actually draws.
        class _DummyEnv:
            obs_circle = []

        super().__init__(
            X0, robot_spec,
            controller_type=controller_type,
            dt=dt,
            show_animation=False,
            save_animation=False,
            show_mpc_traj=False,
            enable_rotation=True,
            raise_error=raise_error,
            ax=None, fig=None, env=_DummyEnv()
        )
        # Obstacles come from /tracked_objects, not env
        self.obs = np.empty((0, 5))

        # Override position controller if needed
        if self.pos_controller_type == 'cbf_qp':
            from position_control.cbf_qp import CBFQP
            self.pos_controller = CBFQP(self.robot, self.robot_spec, num_obs=10)

        # ── internal state ────────────────────────────────────────────────────
        self._lock = threading.Lock()
        self._odom_received = False

        # obs array shape (N, 5): [x, y, r, vx, vy]  — world frame, no bounce
        # Populated from /tracked_objects; radius is fixed (see param below).
        self._obs_radius = rospy.get_param('~obs_radius', 0.1)

        # ── ROS I/O ───────────────────────────────────────────────────────────
        self._cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

        rospy.Subscriber('/odometry/filtered', Odometry,
                         self._odom_cb, queue_size=1)
        rospy.Subscriber('/tracked_objects', Float32MultiArray,
                         self._tracked_objects_cb, queue_size=1)

        rospy.loginfo("[TrackingControllerROS] Initialised. Waiting for odometry…")

    # ── robot setup (no matplotlib) ───────────────────────────────────────────
    def setup_robot(self, X0):
        from safe_control.dynamic_env.robot import BaseRobotDyn
        from safe_control.utils.headless_plot import NullAxes, NullFigure
        null_ax = NullAxes(NullFigure())
        self.robot = BaseRobotDyn(X0.reshape(-1, 1), self.robot_spec, self.dt, ax=null_ax)

    # ── odometry callback ─────────────────────────────────────────────────────
    def _odom_cb(self, msg: Odometry):
        """
        Sync robot state from odometry.
        Extracts [x, y, theta, v] and writes directly into self.robot.X
        so the controller always works with the real pose.
        """
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation

        # quaternion → yaw
        # q = [ori.x, ori.y, ori.z, ori.w]
        # _, _, yaw = tft.euler_from_quaternion(q)
        q = ori
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        # forward velocity (project body-frame linear vel onto world heading)
        vx_body = msg.twist.twist.linear.x
        vy_body = msg.twist.twist.linear.y
        v = math.hypot(vx_body, vy_body)

        with self._lock:
            model = self.robot_spec['model']
            if model in ['KinematicBicycle2D', 'KinematicBicycle2D_C3BF',
                         'KinematicBicycle2D_DPCBF', 'DynamicUnicycle2D']:
                # state: [x, y, theta, v]
                self.robot.X = np.array([[pos.x], [pos.y], [yaw], [v]])
            elif model == 'Unicycle2D':
                # state: [x, y, theta]
                self.robot.X = np.array([[pos.x], [pos.y], [yaw]])
            elif model in ['SingleIntegrator2D']:
                # state: [x, y]
                self.robot.X = np.array([[pos.x], [pos.y]])
            elif model in ['DoubleIntegrator2D']:
                # state: [x, y, vx, vy]
                vx_w = vx_body * math.cos(yaw) - vy_body * math.sin(yaw)
                vy_w = vx_body * math.sin(yaw) + vy_body * math.cos(yaw)
                self.robot.X = np.array([[pos.x], [pos.y], [vx_w], [vy_w]])
            else:
                # Fallback: at least keep x, y, yaw
                self.robot.X[:3] = np.array([[pos.x], [pos.y], [yaw]])

            self._odom_received = True

    # ── tracked objects callback ──────────────────────────────────────────────
    def _tracked_objects_cb(self, msg: Float32MultiArray):
        """
        Convert /tracked_objects (base_link frame) to world-frame obs array.

        Message layout: [id, x, y, vx, vy,  id, x, y, vx, vy, ...]
        We discard `id` and convert (x_bl, y_bl, vx_bl, vy_bl) → world frame
        using the current odometry pose (rotation only — no tf2 needed).
        """
        data = msg.data
        fields_per_obj = 5  # id, x, y, vx, vy

        if len(data) % fields_per_obj != 0:
            rospy.logwarn_throttle(
                5.0,
                f"[TrackingControllerROS] /tracked_objects length {len(data)} "
                f"not divisible by {fields_per_obj} — skipping."
            )
            return

        with self._lock:
            if not self._odom_received:
                return  # can't transform yet

            # Current robot yaw for base_link → world rotation
            yaw = float(self.robot.X[2, 0])
            cos_y, sin_y = math.cos(yaw), math.sin(yaw)
            rx, ry = float(self.robot.X[0, 0]), float(self.robot.X[1, 0])

            n_obs = len(data) // fields_per_obj
            obs_list = []
            for i in range(n_obs):
                base = i * fields_per_obj
                # _id  = data[base]       # not used
                x_bl = data[base + 1]
                y_bl = data[base + 2]
                vx_bl = data[base + 3]
                vy_bl = data[base + 4]

                # Rotate position from base_link → world
                x_w = rx + cos_y * x_bl - sin_y * y_bl
                y_w = ry + sin_y * x_bl + cos_y * y_bl

                # Rotate velocity from base_link → world
                vx_w = cos_y * vx_bl - sin_y * vy_bl
                vy_w = sin_y * vx_bl + cos_y * vy_bl

                # obs_list.append([x_w, y_w, self._obs_radius, vx_w, vy_w])
                obs_list.append([x_w, y_w, self._obs_radius, 0.0, 0.0, 0.0, 0])


            self.obs = np.array(obs_list) if obs_list else np.empty((0, 5))

    # ── no-op rendering overrides ─────────────────────────────────────────────
    def draw_plot(self, pause=0.0, force_save=False):
        pass  # matplotlib removed

    def draw_infeasible(self):
        pass

    def update_unknown_obs_visual(self, detected_obs):
        pass

    def render_dyn_obs(self):
        pass

    # ── dynamic obstacle stepping ─────────────────────────────────────────────
    # Gazebo/the tracker updates obstacle positions; we do NOT integrate them
    # ourselves.  The step_dyn_obs() from the parent is intentionally NOT called.
    def step_dyn_obs(self):
        pass  # positions come from /tracked_objects each tick

    # ── publish helpers ───────────────────────────────────────────────────────
    def _publish_cmd(self, u):
        """
        Map the controller output `u` to a Twist message.

        Convention depends on model:
          KinematicBicycle / DynamicUnicycle: u = [a, delta]  → v from state, omega computed
          Unicycle2D:                          u = [v, omega]
          SingleIntegrator2D:                 u = [vx, vy]   → forward + angular approx
        """
        twist = Twist()
        model = self.robot_spec['model']

        if model in ['KinematicBicycle2D', 'KinematicBicycle2D_C3BF',
                     'KinematicBicycle2D_DPCBF']:
            # u = [acceleration, steering_rate]
            # current speed from state
            v_cur = float(self.robot.X[3, 0])
            a = float(u[0])
            delta_dot = float(u[1])
            twist.linear.x = v_cur + a * self.dt
            twist.angular.z = delta_dot           # steering rate as proxy

        elif model in ['DynamicUnicycle2D']:
            # u = [a, omega]
            v_cur = float(self.robot.X[3, 0])
            twist.linear.x = v_cur + float(u[0]) * self.dt
            twist.angular.z = float(u[1])

        elif model in ['Unicycle2D']:
            # u = [v, omega]
            twist.linear.x = float(u[0])
            twist.angular.z = float(u[1])

        elif model in ['SingleIntegrator2D']:
            # u = [vx, vy] in world frame — approximate cmd_vel in body frame
            yaw = float(self.robot.X[2, 0]) if self.robot.X.shape[0] > 2 else 0.0
            vx_w, vy_w = float(u[0]), float(u[1])
            twist.linear.x = vx_w * math.cos(yaw) + vy_w * math.sin(yaw)
            twist.angular.z = 0.0  # no heading correction here

        elif model in ['DoubleIntegrator2D']:
            # u = [ax, ay] — integrate and rotate to body frame
            vx_w = float(self.robot.X[2, 0]) + float(u[0]) * self.dt
            vy_w = float(self.robot.X[3, 0]) + float(u[1]) * self.dt
            yaw = math.atan2(vy_w, vx_w) if (abs(vx_w) + abs(vy_w)) > 1e-3 else 0.0
            twist.linear.x = math.hypot(vx_w, vy_w)
            twist.angular.z = 0.0

        else:
            rospy.logwarn_throttle(10.0, f"[TrackingControllerROS] Unknown model '{model}' — zeroing cmd_vel.")

        self._cmd_pub.publish(twist)

    # ── main control tick ─────────────────────────────────────────────────────
    def control_step(self):
        """
        One tick of the control loop.  Mirrors the original control_step() but
        with all matplotlib calls removed and cmd_vel published at the end.

        Returns:
            -2  : QP infeasible or collision
            -1  : all waypoints reached
             0  : normal
             1  : visibility violation (if sensor footprint active)
        """
        # ── state machine ──────────────────────────────────────────────────────
        if self.state_machine == 'stop':
            if self.robot.has_stopped():
                if self.enable_rotation:
                    self.state_machine = 'rotate'
                else:
                    self.state_machine = 'track'
                self.goal = self.update_goal()
        else:
            self.goal = self.update_goal()

        # ── detect unknown obstacles ───────────────────────────────────────────
        detected_obs = self.robot.detect_unknown_obs(self.unknown_obs)
        self.update_unknown_obs_visual(detected_obs)   # no-op
        self.nearest_multi_obs = self.get_nearest_unpassed_obs(
            detected_obs, obs_num=self.num_constraints)
        if self.nearest_multi_obs is not None:
            self.nearest_obs = self.nearest_multi_obs[0].reshape(-1, 1)

        # ── NOTE: dynamic obstacle positions come from /tracked_objects ────────
        #    step_dyn_obs() is intentionally a no-op in this class.

        # ── nominal control input ──────────────────────────────────────────────
        if self.state_machine == 'rotate':
            goal_angle = np.arctan2(
                self.goal[1] - self.robot.X[1, 0],
                self.goal[0] - self.robot.X[0, 0])
            model = self.robot_spec['model']
            if model in ['SingleIntegrator2D', 'DoubleIntegrator2D']:
                self.u_att = self.robot.rotate_to(goal_angle)
                u_ref = self.robot.stop()
            else:
                u_ref = self.robot.rotate_to(goal_angle)
        elif self.goal is None:
            u_ref = self.robot.stop()
        else:
            if self.pos_controller_type == 'optimal_decay_cbf_qp':
                u_ref = self.robot.nominal_input(
                    self.goal, k_omega=3.0, k_a=0.5, k_v=0.5)
            else:
                u_ref = self.robot.nominal_input(self.goal)

        # ── solve CBF-QP ───────────────────────────────────────────────────────
        control_ref = {
            'state_machine': self.state_machine,
            'u_ref': u_ref,
            'goal': self.goal
        }
        u = self.pos_controller.solve_control_problem(
            self.robot.X, control_ref, self.nearest_multi_obs)

        # ── attitude controller (integrators only) ─────────────────────────────
        if self.state_machine == 'track' and self.att_controller is not None:
            self.u_att = self.att_controller.solve_control_problem(
                self.robot.X, self.robot.yaw, u)

        # ── infeasibility / collision check ────────────────────────────────────
        collide = self.is_collide_unknown()
        if self.pos_controller.status != 'optimal' or collide:
            cause = "Collision" if collide else "Infeasible QP"
            rospy.logwarn(f"[TrackingControllerROS] {cause} detected!")
            self.draw_infeasible()   # no-op
            if self.raise_error:
                raise InfeasibleError(f"{cause} detected!")
            # Publish zero velocity for safety
            self._cmd_pub.publish(Twist())
            return -2

        # ── step the robot model (internal state only) ─────────────────────────
        # robot.step() integrates the internal state; the real robot is driven
        # by cmd_vel.  Odometry will overwrite self.robot.X next tick anyway.
        self.robot.step(u, self.u_att)
        self.u_pos = u

        # ── publish cmd_vel ────────────────────────────────────────────────────
        self._publish_cmd(u)

        # ── sensing footprint check (optional) ────────────────────────────────
        if 'sensor' in self.robot_spec and self.robot_spec['sensor'] == 'rgbd':
            self.robot.update_sensing_footprints()
            self.robot.update_safety_area()
            beyond_flag = self.robot.is_beyond_sensing_footprints()
        else:
            beyond_flag = 0

        if self.goal is None and self.state_machine != 'stop':
            rospy.loginfo("[TrackingControllerROS] All waypoints reached.")
            self._cmd_pub.publish(Twist())  # stop the robot
            return -1

        return beyond_flag


# ─────────────────────────────────────────────────────────────────────────────
# ROS node entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    rospy.init_node('tracking_controller', anonymous=False)

    # ── parameters ─────────────────────────────────────────────────────────────
    dt     = rospy.get_param('~dt',    0.05)
    model  = rospy.get_param('~model', 'KinematicBicycle2D_DPCBF')
    init_p = rospy.get_param('~init_position', [-2.0, 3.0, 1.57])   # [x, y, theta]
    goal_p = rospy.get_param('~goal_position', [0.0, 10.0])          # [x, y]

    # ── robot spec ──────────────────────────────────────────────────────────────
    robot_specs = {
        'SingleIntegrator2D': {
            'model': 'SingleIntegrator2D', 'v_max': 1.0, 'radius': 0.25
        },
        'DoubleIntegrator2D': {
            'model': 'DoubleIntegrator2D', 'v_max': 1.0, 'a_max': 1.0, 'radius': 0.25
        },
        'DynamicUnicycle2D': {
            'model': 'DynamicUnicycle2D', 'w_max': 0.5, 'a_max': 0.5, 'radius': 0.25
        },
        'KinematicBicycle2D': {
            'model': 'KinematicBicycle2D', 'a_max': 0.5, 'radius': 0.5
        },
        'KinematicBicycle2D_C3BF': {
            'model': 'KinematicBicycle2D_C3BF', 'a_max': 5.0, 'radius': 0.3
        },
        'KinematicBicycle2D_DPCBF': {
            'model': 'KinematicBicycle2D_DPCBF', 'a_max': 5.0, 'radius': 0.3
        },
    }

    if model not in robot_specs:
        rospy.logfatal(f"[TrackingControllerROS] Unknown model: {model}")
        return

    robot_spec = robot_specs[model]

    # ── initial state ───────────────────────────────────────────────────────────
    if model in ['SingleIntegrator2D', 'DoubleIntegrator2D']:
        x_init = np.array(init_p[:2], dtype=np.float64)
    else:
        # [x, y, theta, v=0]
        x_init = np.array([init_p[0], init_p[1], init_p[2], 0.0], dtype=np.float64)

    # ── waypoints ───────────────────────────────────────────────────────────────
    # Single goal waypoint; extend this list for multi-point missions.
    if model in ['SingleIntegrator2D', 'DoubleIntegrator2D']:
        waypoints = np.array([[goal_p[0], goal_p[1]]], dtype=np.float64)
    else:
        waypoints = np.array([[goal_p[0], goal_p[1], 0.0]], dtype=np.float64)

    # ── instantiate controller ──────────────────────────────────────────────────
    controller = LocalTrackingControllerROS(
        x_init, robot_spec,
        controller_type={'pos': 'cbf_qp'},
        dt=dt,
        raise_error=False
    )

    # Initialise obstacle array (populated by /tracked_objects callback)
    controller.obs = np.empty((0, 7))
    controller.set_waypoints(waypoints)

    # ── control loop ────────────────────────────────────────────────────────────
    rate = rospy.Rate(1.0 / dt)

    rospy.loginfo(f"[TrackingControllerROS] Running at {1.0/dt:.1f} Hz | model: {model}")
    rospy.loginfo(f"[TrackingControllerROS] init: {init_p}  goal: {goal_p}")

    while not rospy.is_shutdown():
        if not controller._odom_received:
            rospy.loginfo_throttle(2.0, "[TrackingControllerROS] Waiting for /odometry/filtered …")
            rate.sleep()
            continue

        with controller._lock:
            ret = controller.control_step()

        if ret == -1:
            rospy.loginfo("[TrackingControllerROS] Goal reached. Node idle.")
            break
        elif ret == -2:
            rospy.logwarn_throttle(1.0, "[TrackingControllerROS] Infeasible/collision — holding.")

        rate.sleep()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass