#!/usr/bin/env python

from sensor_msgs.msg import LaserScan
# from nav_msgs.msg import Odometry

import numpy as np
from geometry_msgs.msg import Pose2D


def euler_from_quaternion(quaternion):
    """
    Convert quaternion (x, y, z, w) to euler angles (roll, pitch, yaw).
    Pure NumPy implementation — no tf dependency.
    """
    x, y, z, w = quaternion

    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)  # use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return (roll, pitch, yaw)


def lidar_scan(msgScan):
    """
    Convert LaserScan msg to array
    """
    distances = np.array([])
    angles = np.array([])
    information = np.array([])

    for i in range(len(msgScan.ranges)):
        ang = i * msgScan.angle_increment

        if msgScan.ranges[i] > msgScan.range_max:
            dist = msgScan.range_max
        elif msgScan.ranges[i] < msgScan.range_min:
            dist = msgScan.range_min
        else:
            dist = msgScan.ranges[i]

        inf = ((msgScan.range_max - dist) / msgScan.range_max) ** 2

        distances = np.append(distances, dist)
        angles = np.append(angles, ang)
        information = np.append(information, inf)

    return (distances, angles, information)


def lidar_scan_xy(distances, angles, x_odom, y_odom, theta_odom):
    """
    Lidar measurements in X-Y plane
    """
    distances_x = np.array([])
    distances_y = np.array([])

    for (dist, ang) in zip(distances, angles):
        distances_x = np.append(distances_x, x_odom + dist * np.cos(ang + theta_odom))
        distances_y = np.append(distances_y, y_odom + dist * np.sin(ang + theta_odom))

    return (distances_x, distances_y)


def transform_orientation(orientation_q):
    """
    Transform theta to [radians] from [quaternion orientation]
    """
    orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
    (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
    if yaw < 0:
        yaw = 2 * np.pi + yaw  # 0->360 degrees >> 0->2pi
    return yaw


def get_odom_orientation(msgOdom):
    """
    Get theta from Odometry msg in [radians]
    """
    orientation_q = msgOdom.pose.pose.orientation
    theta = transform_orientation(orientation_q)
    return theta


def get_odom_position(msgOdom):
    """
    Get (x,y) coordinates from Odometry msg in [m]
    """
    x = msgOdom.pose.pose.position.x
    y = msgOdom.pose.pose.position.y
    return (x, y)


def get_pose2d(msgPose):
    """
    Get (x, y, theta) from Pose2D msg
    x, y in [m], theta in [radians]
    """
    x = msgPose.x
    y = msgPose.y
    theta = msgPose.theta  # already in radians
    return (x, y, theta)