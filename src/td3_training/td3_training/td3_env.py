"""
td3_env.py — Gazebo Training Environment with Randomized Goals

KEY FEATURE: After every episode (success OR fail), the goal
randomizes to a new safe position inside the arena.
This teaches the robot to navigate to ANY goal, not just one.
td3_env.py — Gazebo Training Environment with Randomized Goals + RViz support

KEY FEATURE: After every episode (success OR fail), the goal
randomizes to a new safe position inside the arena.

Arena:   2m × 2m  (x: -1.0 to +1.0,  y: -1.0 to +1.0)
Robot:   starts at (0, 0) each episode
Obstacles: (0.5,0.5)  (-0.6,0.4)  (0.4,-0.6)  — all 0.2×0.2m boxes
Goal:    random each episode, guaranteed not inside obstacle or wall

RViz topics published:
  /current_goal    → goal x,y for RViz visualizer
  /training_stats  → episode,reward,result for RViz text overlay
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
from gazebo_msgs.srv import SpawnEntity, DeleteEntity
from geometry_msgs.msg import Pose
from std_msgs.msg import String
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import math, numpy as np, random, time

# ── Arena bounds — keep goals well away from walls ─────────────────────
# Walls are at ±1.0m. Robot body is 0.21×0.135m.
# Keep goals at least 0.30m from walls so robot has room to stop
ARENA_MIN = -0.70
ARENA_MAX =  0.70

# ── Goal ───────────────────────────────────────────────────────────────
GOAL_TOLERANCE   = 0.20
GOAL_MIN_DIST    = 0.40
GOAL_MARKER_NAME = "goal_marker"

# ── Obstacles (x, y, clearance_radius) ────────────────────────────────
# Plate obstacles — longer but thinner than boxes
# clearance = half plate length (0.20m) + robot half-width (0.07m) + margin
OBSTACLES = [
    ( 0.45,  0.45, 0.25),   # plate 1: top-right,   rotated 45°, 0.40m long
    (-0.55,  0.40, 0.22),   # plate 2: top-left,    straight,    0.35m long
    ( 0.15, -0.55, 0.20),   # plate 3: bottom-centre, rotated 45°, 0.30m long
]

# ── LiDAR ─────────────────────────────────────────────────────────────
NUM_LIDAR_BINS = 24
MAX_LIDAR_DIST = 3.5

# ── Safety ────────────────────────────────────────────────────────────
# Collision distance analysis:
#   Arena walls are at 1.0m from centre
#   ARENA_MIN = -0.70, so robot can be at most 0.70m from centre
#   Distance to wall from robot at x=-0.70: 1.0 - 0.70 = 0.30m
#   Obstacle box half-size = 0.10m
#   Robot half-width = 0.0675m
#   Real contact distance ≈ 0.10 + 0.0675 = 0.168m from obstacle centre
#   Use 0.20m threshold — catches real hits, safe from wall false alarms
COLLISION_DIST        = 0.20   # metres
COLLISION_BINS_NEEDED = 2      # 2 bins minimum to confirm real obstacle

# ── Episode ───────────────────────────────────────────────────────────
MAX_STEPS = 500

# ── Reward ────────────────────────────────────────────────────────────
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

_goal_quadrant_idx = 0   # global counter for strict rotation

# Goals placed directly BEHIND each plate obstacle from robot start (0,0)
# Forces robot to navigate around the plate to reach the goal
FORCED_OBSTACLE_GOALS = [
    ( 0.65,  0.65),   # behind plate_1 at (0.45, 0.45)
    (-0.65,  0.60),   # behind plate_2 at (-0.55, 0.40)
    ( 0.30, -0.68),   # behind plate_3 at (0.15, -0.55)
]
_forced_goal_idx = 0   # cycles through forced goals


def _safe_goal():
    """
    Mix of:
    - 50% forced obstacle-avoidance goals (robot MUST go around obstacle)
    - 50% random quadrant goals (generalization)
    """
    global _goal_quadrant_idx, _forced_goal_idx

    # Every other episode use a forced obstacle goal
    if _goal_quadrant_idx % 2 == 0:
        goal = FORCED_OBSTACLE_GOALS[_forced_goal_idx % len(FORCED_OBSTACLE_GOALS)]
        _forced_goal_idx     += 1
        _goal_quadrant_idx   += 1
        return goal

    # Odd episodes: random quadrant goal
    quadrants = [
        ( 0.10,  ARENA_MAX,  0.10,  ARENA_MAX),  # Q1: top-right
        ( 0.10,  ARENA_MAX,  ARENA_MIN, -0.10),  # Q4: bottom-right
        ( ARENA_MIN, -0.10,  0.10,  ARENA_MAX),  # Q2: top-left
        ( ARENA_MIN, -0.10,  ARENA_MIN, -0.10),  # Q3: bottom-left
    ]

    q = _goal_quadrant_idx % 4
    _goal_quadrant_idx += 1

    xmin, xmax, ymin, ymax = quadrants[q]

    for _ in range(500):
        x = random.uniform(xmin, xmax)
        y = random.uniform(ymin, ymax)

        if math.hypot(x, y) < GOAL_MIN_DIST:
            continue

        too_close = any(
            math.hypot(x - ox, y - oy) < margin + GOAL_TOLERANCE + 0.05
            for ox, oy, margin in OBSTACLES
        )
        if not too_close:
            return x, y

    fallbacks = [(0.55, 0.20), (0.55, -0.20), (-0.55, 0.20), (-0.55, -0.20)]
    return fallbacks[q]


class TD3Env(Node):
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
        # Robot control
        self.cmd_pub  = self.create_publisher(Twist, "/cmd_vel", 10)

        # RViz publishers
        self.goal_pub  = self.create_publisher(String, "/current_goal",   10)
        self.stats_pub = self.create_publisher(String, "/training_stats", 10)

        # Sensors
        self.scan_sub = self.create_subscription(LaserScan, "/scan", self._scan_cb, qos)
        self.odom_sub = self.create_subscription(Odometry,  "/odom", self._odom_cb, 10)

        # Gazebo services
        self.reset_world_srv = self.create_client(Empty,        "/reset_world")
        self.spawn_srv       = self.create_client(SpawnEntity,  "/spawn_entity")
        self.delete_srv      = self.create_client(DeleteEntity, "/delete_entity")

        self.get_logger().info("Waiting for Gazebo services...")
        self.reset_world_srv.wait_for_service(timeout_sec=10.0)
        self.spawn_srv.wait_for_service(timeout_sec=10.0)
        self.delete_srv.wait_for_service(timeout_sec=10.0)
        self.get_logger().info("Gazebo services ready ✅")

        # Sensor state
        self._lidar_bins = [MAX_LIDAR_DIST] * NUM_LIDAR_BINS
        self._pos_x      = 0.0
        self._pos_y      = 0.0
        self._heading    = 0.0
        self._scan_ready = False
        self._odom_ready = False

        # Episode state
        self._goal_x     = 0.75
        self._goal_y     = 0.0
        self._prev_dist  = 0.0
        self._step_count = 0
        self._episode    = 0

        self.get_logger().info("TD3Env ready ✅")

    # ── Sensor callbacks ───────────────────────────────────────────────
    def _scan_cb(self, msg):
        bins      = [MAX_LIDAR_DIST] * NUM_LIDAR_BINS
        bin_width = 360.0 / NUM_LIDAR_BINS
        for i, d in enumerate(msg.ranges):
            if not math.isfinite(d) or d <= 0.01:
                continue
            # Filter LiDAR min_range self-detections
            # LiDAR is on top, but robot body can reflect at very close range
            if d < 0.15:   # anything below 0.15m is the robot itself
                continue
            a   = math.degrees(msg.angle_min + i * msg.angle_increment) % 360
            idx = int(a / bin_width) % NUM_LIDAR_BINS
            bins[idx] = min(bins[idx], d)
        self._lidar_bins = [min(b, MAX_LIDAR_DIST) / MAX_LIDAR_DIST for b in bins]
        self._scan_ready = True

    def _odom_cb(self, msg):
        self._pos_x = msg.pose.pose.position.x
        self._pos_y = msg.pose.pose.position.y
        self._pos_x  = msg.pose.pose.position.x
        self._pos_y  = msg.pose.pose.position.y
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
        deadline = time.time() + 2.0   # hard 2 second max wait
        for _ in range(40):
            if time.time() > deadline:
                break
            rclpy.spin_once(self, timeout_sec=0.05)
            if self._scan_ready and self._odom_ready:
                return
        # Only warn once every 10 calls to avoid log spam
        if not hasattr(self, '_timeout_count'):
            self._timeout_count = 0
        self._timeout_count += 1
        if self._timeout_count % 10 == 1:
            self.get_logger().warn("Sensor timeout — check Gazebo is running!")

    def _goal_dist(self):
        return math.hypot(self._goal_x - self._pos_x, self._goal_y - self._pos_y)

    def _goal_angle(self):
        abs_a = math.atan2(self._goal_y - self._pos_y, self._goal_x - self._pos_x)
        rel_a = abs_a - self._heading
        return (rel_a + math.pi) % (2*math.pi) - math.pi

    def _build_obs(self):
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
        # Skip only the very first step (step_count=1) after reset
        # because LiDAR can still have stale data from previous episode
        if self._step_count <= 1:
            return False
        threshold  = COLLISION_DIST / MAX_LIDAR_DIST
        close_bins = sum(1 for b in self._lidar_bins if b < threshold)
        return close_bins >= COLLISION_BINS_NEEDED

    def _move_goal_marker(self, x, y):
        """Delete old marker and spawn new one at goal position."""
        # SDF for a green cylinder (no collision — robot passes through)
        sdf = f"""<?xml version='1.0'?>
<sdf version='1.6'>
  <model name='goal_marker'>
    <static>true</static>
    <link name='link'>
      <pose>{x} {y} 0.01 0 0 0</pose>
      <visual name='visual'>
        <geometry>
          <cylinder><radius>0.20</radius><length>0.03</length></cylinder>
        </geometry>
        <material>
          <ambient>0 1 0 0.8</ambient>
          <diffuse>0 1 0 0.8</diffuse>
          <emissive>0 0.5 0 1</emissive>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""
        # Step 1: Delete existing marker (ignore errors if it doesn't exist)
        try:
            del_req = DeleteEntity.Request()
            del_req.name = GOAL_MARKER_NAME
            future = self.delete_srv.call_async(del_req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)
        except Exception:
            pass

        time.sleep(0.1)

        # Step 2: Spawn new marker at goal position
        try:
            spawn_req = SpawnEntity.Request()
            spawn_req.name            = GOAL_MARKER_NAME
            spawn_req.xml             = sdf
            spawn_req.reference_frame = "world"
            future = self.spawn_srv.call_async(spawn_req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
            self.get_logger().info(f"  ✅ Goal marker spawned at ({x:.2f}, {y:.2f})")
        except Exception as e:
            self.get_logger().warn(f"  ⚠ Goal marker spawn failed: {e}")

    def _publish_rviz_goal(self):
        msg      = String()
        msg.data = f"{self._goal_x:.3f},{self._goal_y:.3f}"
        self.goal_pub.publish(msg)

    def _publish_rviz_stats(self, reward, result):
        msg      = String()
        msg.data = f"{self._episode},{reward:.1f},{result}"
        self.stats_pub.publish(msg)

    # ── Public API ─────────────────────────────────────────────────────
    def reset(self):
        self._episode += 1

        # Stop robot
        self._pub_cmd(0.0, 0.0)
        time.sleep(0.1)

        # NEW random goal every episode ← KEY FEATURE
        self._goal_x, self._goal_y = _safe_goal()
        self.get_logger().info(
            f"🎯 Ep {self._episode} — New goal: ({self._goal_x:.2f}, {self._goal_y:.2f})"
        )

        # Reset Gazebo world (resets robot position + physics)
        future = self.reset_world_srv.call_async(Empty.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        time.sleep(0.8)
        self._spin_wait(0.3)

        # Spawn goal marker at new position (delete old + spawn new)
        self._move_goal_marker(self._goal_x, self._goal_y)

        # Publish goal to RViz
        for _ in range(5):
            self._publish_rviz_goal()
            rclpy.spin_once(self, timeout_sec=0.05)

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
        lin = float(np.clip(action[0], -1, 1))
        ang = float(np.clip(action[1], -1, 1))

        # Force forward-only movement (no reversing)
        # Map lin from [-1,1] to [0,1] so robot always moves forward
        # This prevents the network learning backward navigation
        lin_fwd = (lin + 1.0) / 2.0   # converts [-1,1] → [0,1]

        self._pub_cmd(lin_fwd * 0.30, ang * 1.50)
        self._spin_wait(0.15)
        self._wait_sensors()

        self._step_count += 1
        obs    = self._build_obs()
        dist   = self._goal_dist()
        reward = R_STEP

        # Progress reward

        reward = R_STEP
        progress = (self._prev_dist - dist) / max(self._prev_dist, 0.01)
        reward  += R_PROGRESS * progress
        self._prev_dist = dist

        done   = False
        info   = {}
        done = False
        info = {}

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
                f"✅ GOAL reached in {self._step_count} steps! "
                f"goal=({self._goal_x:.2f},{self._goal_y:.2f})"
            )
        elif self._collision():
            # Debug: show which bins triggered collision
            threshold  = COLLISION_DIST / MAX_LIDAR_DIST
            bad_bins   = [(i, round(b * MAX_LIDAR_DIST, 3))
                          for i, b in enumerate(self._lidar_bins) if b < threshold]
            reward += R_COLLISION
            done    = True
            info    = {"result": "collision"}
            self.get_logger().warn(
                f"💥 Collision at step {self._step_count}! "
                f"Close bins (idx, metres): {bad_bins}"
            )
        elif self._step_count >= MAX_STEPS:
            done = True
            info = {"result": "timeout"}

        if done:
            self._publish_rviz_stats(reward, info.get("result", "timeout"))

        return obs, reward, done, info

    def _pub_cmd(self, linear, angular):
        msg = Twist()
        msg.linear.x  = float(linear)
        msg.angular.z = float(angular)
        self.cmd_pub.publish(msg)
