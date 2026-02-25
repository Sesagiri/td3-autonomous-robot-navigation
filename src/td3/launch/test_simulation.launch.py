#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    world_file_name = 'td3.world'
    
    # Paths
    pkg_td3 = get_package_share_directory('td3')
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    
    world = os.path.join(pkg_td3, 'worlds', world_file_name)
    launch_file_dir = os.path.join(pkg_td3, 'launch')
    rviz_file = os.path.join(pkg_td3, 'launch', 'pioneer3dx.rviz')

    return LaunchDescription([
        # 1. Start Gazebo Server with your TD3 World
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')
            ),
            launch_arguments={'world': world}.items(),
        ),

        # 2. Start Gazebo Client (The UI)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py')
            ),
        ),

        # 3. CHANGE: Launch the ACTUAL Training Node
        Node(
            package='td3',
            executable='train_4wheel_node.py',
            output='screen'
        ),

        # 4. NEW: Spawn the 4-wheel Robot into the World
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=['-topic', 'robot_description', '-entity', 'r1'],
            output='screen'
        ),

        # 5. Robot State Publisher (Provides TF transforms)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([launch_file_dir, '/robot_state_publisher.launch.py']),
            launch_arguments={'use_sim_time': use_sim_time}.items(),
        ),

        # 6. RViz for Visualizing LiDAR and Goals
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',  
            arguments=['-d', rviz_file],
            output='screen'
        ),
    ])
