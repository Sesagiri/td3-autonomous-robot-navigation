"""
td3_env.py — Gazebo Training Environment with Randomized Goals
==============================================================

KEY FEATURE: After every episode (success OR fail), the goal
randomizes to a new safe position inside the arena.
This teaches the robot to navigate to ANY goal, not just one.

Arena:   2m × 2m  (x: -1.0 to +1.0,  y: -1.0 to +1.0)
Robot:   starts at (0, 0) each episode
Obstacles: (0.5,0.5)  (-0.6,0.4)  (0.4,-0.6)  — all 0.2×0.2m boxes
Goal:    random each episode, guaranteed not inside obstacle or wall
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import EntityState
from geometry_msgs.msg import Pose
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import math, numpy as np, random, time

# ── Arena bounds (keep robot well inside walls) ────────────────────────
ARENA_MIN = -0.80
ARENA_MAX =  0.80

# ── Goal settings ──────────────────────────────────────────────────────
GOAL_TOLERANCE    = 0.20   # metres — robot centre must be within this
GOAL_MIN_DIST     = 0.40   # goal must be at least this far from robot start
GOAL_MARKER_NAME  = "goal_marker"   # name in your .world file

# ── Obstacle positions + half-sizes (for safe goal sampling) ──────────
OBSTACLES = [
    (0.5,  0.5,  0.20),   # (x, y, half_size_with_margin)
    (-0.6, 0.4,  0.20),
    (0.4, -0.6,  0.20),
]

# ── LiDAR ─────────────────────────────────────────────────────────────
NUM_LIDAR_BINS = 24          # 15° per bin — MUST match inference
MAX_LIDAR_DIST = 3.5

# ── Safety ─────────────────────────────────────────────────────────────
COLLISION_DIST = 0.15        # metres — any bin closer → collision

# ── Episode ────────────────────────────────────────────────────────────
MAX_STEPS = 500

# ── Reward ─────────────────────────────────────────────────────────────
R_GOAL      =  200.0
R_COLLISION = -100.0
R_STEP      =   -0.5
R_PROGRESS  =   30.0


def _safe_goal():
    """
    Sample a random (x, y) inside the arena that is:
      - Not inside or too close to any obstacle
      - Not too close to robot start (0,0)
      - Not too close to arena walls
    """
    for _ in range(1000):
        x = random.uniform(ARENA_MIN + 0.1, ARENA_MAX - 0.1)
        y = random.uniform(ARENA_MIN + 0.1, ARENA_MAX - 0.1)

        # Too close to start?
        if math.hypot(x, y) < GOAL_MIN_DIST:
            continue

        # Too close to any obstacle?
        too_close = False
        for ox, oy, margin in OBSTACLES:
            if math.hypot(x - ox, y - oy) < margin + GOAL_TOLERANCE + 0.05:
                too_close = True
                break
        if too_close:
            continue

        return x, y

    # Fallback (should never happen with this arena)
    return 0.75, 0.0


class TD3Env(Node):
    """
    Gym-like ROS2 node.
    Usage:
        env = TD3Env()
        obs = env.reset()
        obs, reward, done, info = env.step(action)
    """

    def __init__(self):
        super().__init__("td3_env")

        qos = QoSProfile(depth=10,
                         reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.VOLATILE)

        self.cmd_pub  = self.create_publisher(Twist, "/cmd_vel", 10)
        self.scan_sub = self.create_subscription(LaserScan, "/scan", self._scan_cb, qos)
        self.odom_sub = self.create_subscription(Odometry,  "/odom", self._odom_cb, 10)

        self.reset_world_srv = self.create_client(Empty,          "/reset_world")
        self.set_state_srv   = self.create_client(SetEntityState, "/set_entity_state")

        # Wait for services
        self.get_logger().info("Waiting for Gazebo services...")
        self.reset_world_srv.wait_for_service(timeout_sec=10.0)
        self.set_state_srv.wait_for_service(timeout_sec=10.0)
        self.get_logger().info("Gazebo services ready ✅")

        # Sensor state
        self._lidar_bins  = [MAX_LIDAR_DIST] * NUM_LIDAR_BINS
        self._pos_x       = 0.0
        self._pos_y       = 0.0
        self._heading     = 0.0   # radians
        self._scan_ready  = False
        self._odom_ready  = False

        # Episode state
        self._goal_x      = 0.75
        self._goal_y      = 0.0
        self._prev_dist   = 0.0
        self._step_count  = 0

        self.get_logger().info("TD3Env ready ✅")

    # ── Sensor callbacks ───────────────────────────────────────────────
    def _scan_cb(self, msg):
        bins      = [MAX_LIDAR_DIST] * NUM_LIDAR_BINS
        bin_width = 360.0 / NUM_LIDAR_BINS
        for i, d in enumerate(msg.ranges):
            if not math.isfinite(d) or d <= 0.01:
                continue
            a   = math.degrees(msg.angle_min + i * msg.angle_increment) % 360
            idx = int(a / bin_width) % NUM_LIDAR_BINS
            bins[idx] = min(bins[idx], d)
        self._lidar_bins = [min(b, MAX_LIDAR_DIST) / MAX_LIDAR_DIST for b in bins]
        self._scan_ready = True

    def _odom_cb(self, msg):
        self._pos_x = msg.pose.pose.position.x
        self._pos_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self._heading    = math.atan2(
            2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
        self._odom_ready = True

    # ── Utilities ──────────────────────────────────────────────────────
    def _spin_wait(self, secs=0.5):
        deadline = time.time() + secs
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)

    def _wait_sensors(self):
        self._scan_ready = False
        self._odom_ready = False
        for _ in range(40):           # up to 2 seconds
            rclpy.spin_once(self, timeout_sec=0.05)
            if self._scan_ready and self._odom_ready:
                return
        self.get_logger().warn("Sensor timeout — using last values")

    def _goal_dist(self):
        return math.hypot(self._goal_x - self._pos_x,
                          self._goal_y - self._pos_y)

    def _goal_angle(self):
        abs_a = math.atan2(self._goal_y - self._pos_y,
                           self._goal_x - self._pos_x)
        rel_a = abs_a - self._heading
        return (rel_a + math.pi) % (2*math.pi) - math.pi   # [-π, π]

    def _build_obs(self):
        """26-float observation — identical layout must be used in td3_inference.py"""
        dist_norm  = min(self._goal_dist(), 3.0) / 3.0
        angle_norm = self._goal_angle() / math.pi
        return np.array(self._lidar_bins + [dist_norm, angle_norm], dtype=np.float32)

    def _collision(self):
        # Already normalised — multiply back to get metres
        return min(self._lidar_bins) * MAX_LIDAR_DIST < COLLISION_DIST

    # ── Move goal marker in Gazebo (visual only) ───────────────────────
    def _move_goal_marker(self, x, y):
        if not self.set_state_srv.service_is_ready():
            return
        req = SetEntityState.Request()
        req.state = EntityState()
        req.state.name = GOAL_MARKER_NAME
        req.state.pose = Pose()
        req.state.pose.position.x = x
        req.state.pose.position.y = y
        req.state.pose.position.z = 0.01
        req.state.reference_frame = "world"
        self.set_state_srv.call_async(req)   # fire-and-forget

    # ── Public API ─────────────────────────────────────────────────────
    def reset(self):
        """
        1. Stop robot
        2. Pick NEW random goal
        3. Reset Gazebo (robot back to origin)
        4. Move goal marker to new position
        5. Return first observation
        """
        # 1. Stop
        self._pub_cmd(0.0, 0.0)
        time.sleep(0.1)

        # 2. New random goal ← THE KEY FEATURE
        self._goal_x, self._goal_y = _safe_goal()
        self.get_logger().info(
            f"🎯 New goal: ({self._goal_x:.2f}, {self._goal_y:.2f})"
        )

        # 3. Reset sim
        req    = Empty.Request()
        future = self.reset_world_srv.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        time.sleep(0.3)

        # 4. Move goal marker in Gazebo
        self._move_goal_marker(self._goal_x, self._goal_y)

        # 5. Wait for sensors
        self._wait_sensors()
        self._prev_dist  = self._goal_dist()
        self._step_count = 0

        return self._build_obs()

    def step(self, action):
        """
        action: np.array([linear_vel, angular_vel])  each in [-1, 1]
        linear  maps to ±0.30 m/s
        angular maps to ±1.50 rad/s
        """
        lin = float(np.clip(action[0], -1, 1))
        ang = float(np.clip(action[1], -1, 1))

        self._pub_cmd(lin * 0.30, ang * 1.50)
        self._spin_wait(0.15)
        self._wait_sensors()

        self._step_count += 1
        obs    = self._build_obs()
        dist   = self._goal_dist()
        reward = R_STEP

        # Progress reward
        progress = (self._prev_dist - dist) / max(self._prev_dist, 0.01)
        reward  += R_PROGRESS * progress
        self._prev_dist = dist

        done   = False
        info   = {}

        if dist < GOAL_TOLERANCE:
            reward += R_GOAL
            done    = True
            info    = {"result": "goal"}
            self.get_logger().info(
                f"✅ GOAL ({self._goal_x:.2f},{self._goal_y:.2f}) "
                f"reached in {self._step_count} steps!"
            )
        elif self._collision():
            reward += R_COLLISION
            done    = True
            info    = {"result": "collision"}
            self.get_logger().warn("💥 Collision!")
        elif self._step_count >= MAX_STEPS:
            done = True
            info = {"result": "timeout"}

        return obs, reward, done, info

    def _pub_cmd(self, linear, angular):
        msg = Twist()
        msg.linear.x  = float(linear)
        msg.angular.z = float(angular)
        self.cmd_pub.publish(msg)
