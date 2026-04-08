#!/usr/bin/env python
"""
cbf_tracking_node.py
====================
ROS 1 (Melodic) wrapper around LocalTrackingControllerDyn.

Subscribes
----------
/tracked_objects      std_msgs/Float32MultiArray
    Flat array: [id, x, y, vx, vy,  id, x, y, vx, vy, ...]
    Expected in base_link frame; transformed to world frame internally
    using the current odometry pose.

/odometry/filtered    nav_msgs/Odometry
    Ground-truth robot state.  Synced into the controller every tick.

/waypoints            std_msgs/Float32MultiArray   (optional, overrides params)
    Flat array: [x0, y0,  x1, y1, ...]
    Publishing to this topic replaces the current waypoint list immediately.
    Current odometry position is used as the implicit new start.

Publishes
---------
/cmd_vel              geometry_msgs/Twist
    Linear.x = forward velocity,  angular.z = yaw rate.

ROS Parameters
--------------
~init_position      [x, y, theta]           default: [-2.0, 3.0, 1.57]
~goal_position      [x, y]                  default: [0.0, 10.0]
    Single-goal shorthand. Ignored if ~waypoints is set.
~waypoints          [x0,y0, x1,y1, ...]
    Multi-waypoint list (flat). Takes priority over ~goal_position.
    init_position is always prepended as the first waypoint.

~pos_controller     str                     default: 'cbf_qp'
    Position controller: 'cbf_qp' or 'mpc_cbf'
~att_controller     str                     default: ''
    Attitude controller: '' (none) or 'gatekeeper'

~robot_radius       float                   default: 0.25
~w_max              float                   default: 0.5
~a_max              float                   default: 0.5
~dt                 float                   default: 0.05  (must match timer)
~obs_radius         float                   default: 0.5
    Collision radius assigned to every object from /tracked_objects.
"""

import rospy
import numpy as np
import math
from std_msgs.msg      import Float32MultiArray
from nav_msgs.msg      import Odometry
from geometry_msgs.msg import Twist

# ── project imports ────────────────────────────────────────────────────────────
from safe_control.tracking import LocalTrackingController
# (matplotlib imports intentionally omitted)


# ══════════════════════════════════════════════════════════════════════════════
#  Minimal no-op stubs for every matplotlib / figure surface the base class
#  calls.  This lets us keep the controller logic untouched.
# ══════════════════════════════════════════════════════════════════════════════
class _NullLine:
    """Stub for a matplotlib Line2D."""
    def __getattr__(self, _):
        return lambda *a, **kw: None

class _NullAx:
    """Silently absorbs every axes method call.
    ax.plot() must return a tuple because the base class does: handle, = ax.plot(...)
    ax.fill() must return a list because it does: handle = ax.fill(...)[0]
    ax.scatter() returns a stub directly.
    Everything else returns self so chained calls work.
    """
    def plot(self, *a, **kw):
        return (_NullLine(),)
    def fill(self, *a, **kw):
        return [_NullLine()]
    def scatter(self, *a, **kw):
        return _NullLine()
    def add_patch(self, patch):
        return patch
    def __getattr__(self, _):
        return lambda *a, **kw: self

class _NullFig:
    number = 0
    canvas = type("C", (), {
        "draw_idle":    lambda *a: None,
        "flush_events": lambda *a: None,
    })()
    def __getattr__(self, _):
        return lambda *a, **kw: self

class _NullEnv:
    """Minimal env stub — base class reads env.obs_circle at init."""
    obs_circle = []          # no static obstacles; /tracked_objects fills self.obs at runtime
    def __getattr__(self, _):
        return []


# ══════════════════════════════════════════════════════════════════════════════
#  Headless controller
# ══════════════════════════════════════════════════════════════════════════════
class LocalTrackingControllerDynROS(LocalTrackingController):
    """
    LocalTrackingControllerDyn stripped of all rendering for ROS use.

    Changes vs the matplotlib version
    ----------------------------------
    - All drawing calls go to _NullAx / _NullFig (no-ops).
    - step_dyn_obs() is NOT called — Gazebo owns obstacle motion.
      Obstacle state is written into self.obs from /tracked_objects each tick.
    - Robot state is injected each tick via set_robot_state() from odometry.
    - solve() returns (v, omega) for direct publication on /cmd_vel.
    """

    def __init__(self, X0, robot_spec, controller_type, dt):
        super().__init__(
            X0, robot_spec,
            controller_type=controller_type,
            dt=dt,
            show_animation=False,
            save_animation=False,
            show_mpc_traj=False,
            enable_rotation=True,
            raise_error=False,
            ax=_NullAx(),
            fig=_NullFig(),
            env=_NullEnv(),
        )

        # When cbf_qp is chosen, swap in the dynamic-obstacle variant that
        # accepts velocity columns in the obs array.
        if self.pos_controller_type == 'cbf_qp':
            from position_control.cbf_qp import CBFQP
            self.pos_controller = CBFQP(self.robot, self.robot_spec, num_obs=10)

    # ------------------------------------------------------------------
    def setup_robot(self, X0):
        from safe_control.dynamic_env.robot import BaseRobotDyn
        self.robot = BaseRobotDyn(
            X0.reshape(-1, 1), self.robot_spec, self.dt, _NullAx())

    # ------------------------------------------------------------------
    def set_robot_state(self, x, y, theta, v=0.0):
        """Overwrite internal robot state with current odometry each tick."""
        # DynamicUnicycle2D state: [x, y, theta, v]
        self.robot.X[0, 0] = x
        self.robot.X[1, 0] = y
        self.robot.X[2, 0] = theta
        if self.robot.X.shape[0] > 3:
            self.robot.X[3, 0] = v

    # ------------------------------------------------------------------
    def solve(self):
        """
        Run one CBF-QP / MPC-CBF step.

        Returns
        -------
        (v, omega) : (float, float)   — linear and angular velocity command
        None                          — all waypoints reached
        """
        # ── state machine & goal update ────────────────────────────────
        if self.state_machine == 'stop':
            if self.robot.has_stopped():
                self.state_machine = 'rotate'
                self.goal = self.update_goal()
        else:
            self.goal = self.update_goal()

        if self.goal is None:
            return None  # done

        # ── obstacle pipeline ──────────────────────────────────────────
        detected_obs = self.robot.detect_unknown_obs(self.unknown_obs)
        self.update_unknown_obs_visual(detected_obs)   # no-op
        self.nearest_multi_obs = self.get_nearest_unpassed_obs(
            detected_obs, obs_num=self.num_constraints)
        if self.nearest_multi_obs is not None:
            self.nearest_obs = self.nearest_multi_obs[0].reshape(-1, 1)

        # step_dyn_obs() intentionally omitted —
        # /tracked_objects already carries current world-frame positions.

        # ── nominal reference ──────────────────────────────────────────
        if self.state_machine == 'rotate':
            goal_angle = math.atan2(
                self.goal[1] - self.robot.X[1, 0],
                self.goal[0] - self.robot.X[0, 0])
            u_ref = self.robot.rotate_to(goal_angle)
        else:
            u_ref = self.robot.nominal_input(self.goal)

        control_ref = {
            'state_machine': self.state_machine,
            'u_ref':         u_ref,
            'goal':          self.goal,
        }

        # ── solve ──────────────────────────────────────────────────────
        u = self.pos_controller.solve_control_problem(
            self.robot.X, control_ref, self.nearest_multi_obs)

        if self.pos_controller.status != 'optimal':
            rospy.logwarn("Controller infeasible — sending zero velocity.")
            return (0.0, 0.0)

        # ── advance internal model ─────────────────────────────────────
        self.robot.step(u, self.u_att)
        self.u_pos = u

        # DynamicUnicycle2D: u = [a, alpha]
        #   robot.X[3,0] = v after integration
        #   u[1,0]       = omega (angular rate)
        v     = float(self.robot.X[3, 0])
        omega = float(u[1, 0])
        return (v, omega)


# ══════════════════════════════════════════════════════════════════════════════
#  ROS node
# ══════════════════════════════════════════════════════════════════════════════
class CBFTrackingNode:

    def __init__(self):
        rospy.init_node('cbf_tracking_node')

        # ── parameters ────────────────────────────────────────────────
        self.dt         = rospy.get_param('~dt',         0.05)
        self.obs_radius = rospy.get_param('~obs_radius', 0.5)

        init_pos = rospy.get_param('~init_position', [-2.0, 3.0, 1.57])  # [x,y,θ]
        # goal_pos = rospy.get_param('~goal_position',  [0.0, 10.0])       # [x,y]
        goal_rel  = rospy.get_param('~goal_position', [-20.0, 0.0])

        # goal_position is relative to init — convert to absolute world frame
        goal_pos = np.array([
            init_pos[0] + goal_rel[0],
            init_pos[1] + goal_rel[1],
        ], dtype=float)
        
        wp_flat  = rospy.get_param('~waypoints',      None)              # flat list

        # Controller type — validated against known options
        pos_ctrl = rospy.get_param('~pos_controller', 'cbf_qp')
        att_ctrl = rospy.get_param('~att_controller', '')
        if pos_ctrl not in ('cbf_qp', 'mpc_cbf'):
            rospy.logwarn("Unknown ~pos_controller '%s', falling back to 'cbf_qp'",
                          pos_ctrl)
            pos_ctrl = 'cbf_qp'
        controller_type = {'pos': pos_ctrl}
        if att_ctrl:
            controller_type['att'] = att_ctrl

        robot_spec = {
            'model':  'DynamicUnicycle2D',
            'w_max':  rospy.get_param('~w_max',        0.5),
            'a_max':  rospy.get_param('~a_max',        0.5),
            'radius': rospy.get_param('~robot_radius', 0.25),
            # 'sensor' key omitted → sensing-footprint / rgbd checks skipped
        }

        # ── build initial waypoints ────────────────────────────────────
        waypoints = self._build_waypoints(wp_flat, init_pos, goal_pos)

        # ── initialise controller ──────────────────────────────────────
        x_init = np.array([init_pos[0], init_pos[1], init_pos[2], 0.0],
                          dtype=np.float64)

        self.controller = LocalTrackingControllerDynROS(
            x_init,
            robot_spec,
            controller_type=controller_type,
            dt=self.dt,
        )
        self.controller.set_waypoints(waypoints)
        self.controller.obs = np.empty((0, 5))   # [x, y, r, vx, vy]

        # ── robot state cache (written by /odometry/filtered) ──────────
        self._x             = float(init_pos[0])
        self._y             = float(init_pos[1])
        self._theta         = float(init_pos[2])
        self._v             = 0.0
        self._odom_received = False

        # ── publishers & subscribers ───────────────────────────────────
        self._cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

        rospy.Subscriber('/odometry/filtered', Odometry,
                         self._odom_cb,      queue_size=1)
        rospy.Subscriber('/tracked_objects',   Float32MultiArray,
                         self._tracked_cb,   queue_size=1)
        rospy.Subscriber('/waypoints',         Float32MultiArray,
                         self._waypoints_cb, queue_size=1)

        self._timer = rospy.Timer(rospy.Duration(self.dt), self._timer_cb)

        rospy.loginfo(
            "cbf_tracking_node ready | pos=%s att=%s | waypoints=%s",
            pos_ctrl, att_ctrl or 'none', waypoints[:, :2].tolist())

    # ------------------------------------------------------------------
    # Waypoint helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _build_waypoints(wp_flat, init_pos, goal_pos):
        """
        Return (N, 3) float64 array [x, y, heading=0].

        Priority:
          1. ~waypoints param  — flat [x0,y0, x1,y1, ...] list
          2. ~goal_position    — single [x, y]
        init_position is always the first row.
        """
        if wp_flat is not None and len(wp_flat) >= 2:
            pts = np.array(wp_flat, dtype=np.float64).reshape(-1, 2)
        else:
            pts = np.array([goal_pos[:2]], dtype=np.float64)

        start = np.array([[init_pos[0], init_pos[1]]], dtype=np.float64)
        pts   = np.vstack([start, pts])
        return np.hstack([pts, np.zeros((len(pts), 1))])   # add heading col

    # ------------------------------------------------------------------
    # /odometry/filtered
    # ------------------------------------------------------------------
    def _odom_cb(self, msg):
        pos  = msg.pose.pose.position
        quat = msg.pose.pose.orientation
        qx, qy, qz, qw = quat.x, quat.y, quat.z, quat.w
        yaw = math.atan2(2.0 * (qw * qz + qx * qy),
                         1.0 - 2.0 * (qy * qy + qz * qz))

        self._x             = pos.x
        self._y             = pos.y
        self._theta         = yaw
        self._v             = msg.twist.twist.linear.x
        self._odom_received = True

    # ------------------------------------------------------------------
    # /tracked_objects
    # ------------------------------------------------------------------
    def _tracked_cb(self, msg):
        """
        Parse [id, x, y, vx, vy, ...] in base_link frame.
        Transforms to world frame and stores in controller.obs as (N,5):
        [x_w, y_w, r, vx_w, vy_w].
        """
        data           = np.array(msg.data, dtype=np.float64)
        fields_per_obs = 5   # id, x, y, vx, vy
        n_obs          = len(data) // fields_per_obs

        if n_obs == 0:
            self.controller.obs = np.empty((0, 5))
            return

        cx, cy, ct   = self._x, self._y, self._theta
        cos_t, sin_t = math.cos(ct), math.sin(ct)

        obs_list = []
        for i in range(n_obs):
            b   = i * fields_per_obs
            bx  = data[b + 1]
            by  = data[b + 2]
            bvx = data[b + 3]
            bvy = data[b + 4]

            # Position: base_link → world (full rigid-body transform)
            wx  = cx + cos_t * bx - sin_t * by
            wy  = cy + sin_t * bx + cos_t * by

            # Velocity: rotation only (no translation for velocity vectors)
            wvx = cos_t * bvx - sin_t * bvy
            wvy = sin_t * bvx + cos_t * bvy

            obs_list.append([wx, wy, self.obs_radius, wvx, wvy])

        self.controller.obs = np.array(obs_list, dtype=np.float64)

    # ------------------------------------------------------------------
    # /waypoints  — runtime override
    # ------------------------------------------------------------------
    def _waypoints_cb(self, msg):
        """
        Replace the waypoint list on the fly.
        Format: flat Float32MultiArray [x0,y0, x1,y1, ...]
        Current robot position is automatically prepended as the start,
        so there is no discontinuity in the planned path.
        """
        data = np.array(msg.data, dtype=np.float64)
        if len(data) < 2:
            rospy.logwarn("/waypoints: need at least one [x, y] pair — ignored.")
            return

        pts   = data.reshape(-1, 2)
        start = np.array([[self._x, self._y]], dtype=np.float64)
        pts   = np.vstack([start, pts])
        waypoints = np.hstack([pts, np.zeros((len(pts), 1))])

        self.controller.set_waypoints(waypoints)
        rospy.loginfo("Waypoints updated via /waypoints topic: %s",
                      pts[1:].tolist())

    # ------------------------------------------------------------------
    # Control timer
    # ------------------------------------------------------------------
    def _timer_cb(self, _event):
        if not self._odom_received:
            return  # stall until first odometry arrives

        # Sync controller's internal model to ground-truth odometry
        self.controller.set_robot_state(
            self._x, self._y, self._theta, self._v)

        result = self.controller.solve()

        twist = Twist()
        if result is None:
            rospy.loginfo_once("Goal reached — publishing zero velocity.")
        else:
            twist.linear.x  = result[0]   # v
            twist.angular.z = result[1]   # omega

        self._cmd_pub.publish(twist)


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    try:
        node = CBFTrackingNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass