"""
train.launch.py — Launch Gazebo + Robot for TD3 Training
=========================================================
Place this file in:
  ~/ros2_ws/src/td3_training/launch/train.launch.py
"""

import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node


def generate_launch_description():

    # ── Paths ─────────────────────────────────────────────────────────
    pkg_dir   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    world_path = os.path.join(pkg_dir, "worlds", "td3.world")
    urdf_path  = os.path.join(pkg_dir, "urdf",   "td_robot.urdf")

    with open(urdf_path, "r") as f:
        robot_description = f.read()

    return LaunchDescription([

        # 1. Launch Gazebo with TD3 world
        ExecuteProcess(
            cmd=[
                "gazebo", "--verbose", world_path,
                "-s", "libgazebo_ros_factory.so",
                "-s", "libgazebo_ros_init.so",
            ],
            output="screen",
        ),

        # 2. Publish robot URDF to /robot_description topic
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            parameters=[{"robot_description": robot_description}],
            output="screen",
        ),

        # 3. Spawn robot at origin after 3 seconds (Gazebo needs to start first)
        TimerAction(
            period=3.0,
            actions=[
                Node(
                    package="gazebo_ros",
                    executable="spawn_entity.py",
                    name="spawn_robot",
                    arguments=[
                        "-topic", "robot_description",
                        "-entity", "td_robot",
                        "-x", "0.0",
                        "-y", "0.0",
                        "-z", "0.05",
                    ],
                    output="screen",
                ),
            ],
        ),

    ])
