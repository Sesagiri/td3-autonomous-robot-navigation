"""
Shared pytest configuration and fixtures.
Mocks ROS2 dependencies so tests run without
a ROS2 installation or running robot.
"""
import sys
from unittest.mock import MagicMock

# Mock all ROS2 modules — allows importing navigator logic
# on any machine without ROS2 installed
ROS2_MODULES = [
    'rclpy', 'rclpy.node', 'rclpy.qos',
    'geometry_msgs', 'geometry_msgs.msg',
    'sensor_msgs', 'sensor_msgs.msg',
    'nav_msgs', 'nav_msgs.msg',
    'std_msgs', 'std_msgs.msg',
]
for mod in ROS2_MODULES:
    sys.modules[mod] = MagicMock()