"""
rviz_visualizer.py
==================
Publishes extra topics that RViz needs to show:
  1. /goal_marker_rviz  → green sphere showing current goal position
  2. /odom_path         → blue trail of where robot has been
  3. /training_status   → text overlay showing episode/reward/result

Run alongside training:
  Terminal 3: ros2 run td3_training rviz_viz
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker
from std_msgs.msg import String
import math


class RVizVisualizer(Node):
    def __init__(self):
        super().__init__("rviz_visualizer")

        # Publishers
        self.goal_pub   = self.create_publisher(Marker,  "/goal_marker_rviz", 10)
        self.path_pub   = self.create_publisher(Path,    "/odom_path",        10)
        self.status_pub = self.create_publisher(Marker,  "/training_status",  10)

        # Subscriber — listen to odom to build path trail
        self.odom_sub = self.create_subscription(
            Odometry, "/odom", self._odom_cb, 10
        )

        # Subscribe to goal updates from training env
        self.goal_sub = self.create_subscription(
            String, "/current_goal", self._goal_cb, 10
        )

        self._path_msg = Path()
        self._path_msg.header.frame_id = "odom"

        self._goal_x = 0.75
        self._goal_y = 0.0

        self._episode   = 0
        self._reward    = 0.0
        self._result    = "---"

        # Subscribe to training stats
        self.stats_sub = self.create_subscription(
            String, "/training_stats", self._stats_cb, 10
        )

        # Publish at 5 Hz
        self.create_timer(0.2, self._publish_all)

        self.get_logger().info("RViz Visualizer running ✅")
        self.get_logger().info("Open RViz and load: config/td3_training.rviz")

    def _odom_cb(self, msg):
        """Append current robot pose to path trail."""
        pose = PoseStamped()
        pose.header = msg.header
        pose.pose   = msg.pose.pose
        self._path_msg.poses.append(pose)

        # Keep last 500 poses to avoid memory bloat
        if len(self._path_msg.poses) > 500:
            self._path_msg.poses.pop(0)

    def _goal_cb(self, msg):
        """Parse 'x,y' string from training env."""
        try:
            parts = msg.data.split(",")
            self._goal_x = float(parts[0])
            self._goal_y = float(parts[1])
        except Exception:
            pass

    def _stats_cb(self, msg):
        """Parse 'episode,reward,result' from training."""
        try:
            parts = msg.data.split(",")
            self._episode = int(parts[0])
            self._reward  = float(parts[1])
            self._result  = parts[2]
        except Exception:
            pass

    def _publish_all(self):
        now = self.get_clock().now().to_msg()

        # ── 1. Goal sphere ─────────────────────────────────────────────
        goal_marker = Marker()
        goal_marker.header.frame_id = "odom"
        goal_marker.header.stamp    = now
        goal_marker.ns     = "goal"
        goal_marker.id     = 0
        goal_marker.type   = Marker.CYLINDER
        goal_marker.action = Marker.ADD

        goal_marker.pose.position.x = self._goal_x
        goal_marker.pose.position.y = self._goal_y
        goal_marker.pose.position.z = 0.01
        goal_marker.pose.orientation.w = 1.0

        goal_marker.scale.x = 0.40   # 0.20m radius × 2
        goal_marker.scale.y = 0.40
        goal_marker.scale.z = 0.02

        goal_marker.color.r = 0.0
        goal_marker.color.g = 1.0
        goal_marker.color.b = 0.0
        goal_marker.color.a = 0.7

        self.goal_pub.publish(goal_marker)

        # ── 2. Odom path trail ─────────────────────────────────────────
        self._path_msg.header.stamp = now
        self.path_pub.publish(self._path_msg)

        # ── 3. Training status text ────────────────────────────────────
        status_marker = Marker()
        status_marker.header.frame_id = "odom"
        status_marker.header.stamp    = now
        status_marker.ns     = "status"
        status_marker.id     = 1
        status_marker.type   = Marker.TEXT_VIEW_FACING
        status_marker.action = Marker.ADD

        status_marker.pose.position.x = -0.9
        status_marker.pose.position.y =  0.9
        status_marker.pose.position.z =  0.3
        status_marker.pose.orientation.w = 1.0

        status_marker.scale.z = 0.08   # text height

        # Color by result
        if self._result == "goal":
            status_marker.color.r = 0.0
            status_marker.color.g = 1.0
            status_marker.color.b = 0.0
        elif self._result == "collision":
            status_marker.color.r = 1.0
            status_marker.color.g = 0.0
            status_marker.color.b = 0.0
        else:
            status_marker.color.r = 1.0
            status_marker.color.g = 1.0
            status_marker.color.b = 0.0
        status_marker.color.a = 1.0

        status_marker.text = (
            f"Episode: {self._episode}\n"
            f"Reward:  {self._reward:.1f}\n"
            f"Result:  {self._result}\n"
            f"Goal: ({self._goal_x:.2f}, {self._goal_y:.2f})"
        )

        self.status_pub.publish(status_marker)


def main(args=None):
    rclpy.init(args=args)
    node = RVizVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
