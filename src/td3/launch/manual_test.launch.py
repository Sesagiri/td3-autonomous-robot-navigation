import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    pkg_td3 = get_package_share_directory('td3')
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    
    world = os.path.join(pkg_td3, 'worlds', 'td3.world')
    launch_file_dir = os.path.join(pkg_td3, 'launch')

    return LaunchDescription([
        # 1. Start Gazebo Server with the world
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')),
            launch_arguments={'world': world}.items(),
        ),

        # 2. Start Gazebo Client
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py')),
        ),

        # 3. Spawn the Robot
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=['-topic', 'robot_description', '-entity', 'r1'],
            output='screen'
        ),

        # 4. Robot State Publisher
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([launch_file_dir, '/robot_state_publisher.launch.py']),
            launch_arguments={'use_sim_time': 'true'}.items(),
        ),
    ])
