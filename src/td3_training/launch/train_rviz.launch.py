"""
train_rviz.launch.py
=====================
Launches everything at once:
  1. Gazebo  with td3.world
  2. Robot   spawned at origin
  3. RViz    with pre-configured td3_training.rviz

Usage:
  ros2 launch td3_training train_rviz.launch.py
  # then in another terminal:
  python3 ~/ros2_ws/src/td3_training/td3_training/train.py
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node


def generate_launch_description():

    pkg_share  = get_package_share_directory('td3_training')
    world_path = os.path.join(pkg_share, 'worlds', 'td3.world')
    urdf_path  = os.path.join(pkg_share, 'urdf',   'td_robot.urdf')
    rviz_path  = os.path.join(pkg_share, 'config', 'td3_training.rviz')

    with open(urdf_path, 'r') as f:
        robot_description = f.read()

    return LaunchDescription([

        # ── 1. Gazebo ──────────────────────────────────────────────────
        ExecuteProcess(
            cmd=[
                'gazebo', '--verbose', world_path,
                '-s', 'libgazebo_ros_factory.so',
                '-s', 'libgazebo_ros_init.so',
            ],
            output='screen',
        ),

        # ── 2. Robot description publisher ────────────────────────────
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[{'robot_description': robot_description}],
            output='screen',
        ),

        # ── 3. Spawn robot after 5s (give Gazebo time to fully load) ──
        TimerAction(
            period=5.0,
            actions=[
                Node(
                    package='gazebo_ros',
                    executable='spawn_entity.py',
                    name='spawn_robot',
                    arguments=[
                        '-topic', 'robot_description',
                        '-entity', 'td_robot',
                        '-x', '0.0',
                        '-y', '0.0',
                        '-z', '0.05',
                    ],
                    output='screen',
                ),
            ],
        ),

        # ── 4. RViz after 6s ──────────────────────────────────────────
        TimerAction(
            period=6.0,
            actions=[
                Node(
                    package='rviz2',
                    executable='rviz2',
                    name='rviz2',
                    arguments=['-d', rviz_path],
                    output='screen',
                ),
            ],
        ),

    ])
