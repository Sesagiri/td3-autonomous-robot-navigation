import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    use_sim_time = LaunchConfiguration('use_sim_time', default='true')  # FIX 1: was false

    urdf_file_name = 'td_robot.urdf'
    urdf = os.path.join(
        get_package_share_directory('td3'),
        'urdf',
        urdf_file_name
    )

    with open(urdf, 'r') as f:
        robot_desc = f.read()

    return LaunchDescription([

        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',   # FIX 1: was 'false' — must match Gazebo sim clock
            description='Use Gazebo simulation clock'
        ),

        # Publishes TF for: base_footprint→base_link, base_link→lidar, base_link→wheels
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'robot_description': robot_desc,
            }]
        ),

        # FIX 2: ADDED — publishes /joint_states so RSP can compute wheel TF
        # Without this, wheels are missing from TF tree when publish_wheel_tf=false
        Node(
            package='joint_state_publisher',
            executable='joint_state_publisher',
            name='joint_state_publisher',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
            }]
        ),

    ])
