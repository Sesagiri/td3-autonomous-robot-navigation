"""
td3_env.py — Gazebo Training Environment with Randomized Goals + RViz support
==============================================================================

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
from gazebo_msgs.srv import SpawnEntity, DeleteEntity
from geometry_msgs.msg import Pose
from std_msgs.msg import String
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import math, numpy as np, random, time

# ── Arena bounds — keep goals well away from walls ─────────────────────
# Arena: 2.4m x 2.4m, walls at +-1.2m
# Robot spawns at (0,0) = bottom-left area
# Usable area: x=[-1.0, +1.0], y=[-1.0, +1.0]
ARENA_MIN_X = -1.00
ARENA_MAX_X =  1.00
ARENA_MIN_Y = -1.00
ARENA_MAX_Y =  1.00
ARENA_MIN   = ARENA_MIN_X
ARENA_MAX   = ARENA_MAX_X

# ── Goal ───────────────────────────────────────────────────────────────
GOAL_TOLERANCE   = 0.15
GOAL_MIN_DIST    = 0.35
GOAL_MARKER_NAME = "goal_marker"

# ── Obstacles ──────────────────────────────────────────────────────────
# PLATE_A: y=+0.40, x=-1.20 to +0.25  gap on RIGHT (x > +0.25)  blocks LEFT
# PLATE_B: y=-0.30, x=-0.20 to +1.20  gap on LEFT  (x < -0.20)  blocks RIGHT
#
# Path types robot must learn:
#   Straight:  goal in bottom-left  → go straight through PLATE_B left gap
#   1-turn:    goal in top-right    → dodge PLATE_B right, pass PLATE_A right gap
#   S-curve:   goal in top-left     → pass PLATE_B left, pass PLATE_A left (blocked!)
#              Actually must go RIGHT gap of PLATE_A after going left of PLATE_B
OBSTACLES = [
    (-0.475,  0.40, 0.40),   # PLATE_A centre + clearance
    ( 0.500, -0.30, 0.40),   # PLATE_B centre + clearance
]
PLATE_A = dict(y=0.40,  x_min=-1.20, x_max=0.25)   # gap right of x=+0.25
PLATE_B = dict(y=-0.30, x_min=-0.20, x_max=1.20)   # gap left  of x=-0.20

# ── LiDAR ─────────────────────────────────────────────────────────────
NUM_LIDAR_BINS = 24
MAX_LIDAR_DIST = 3.5

# ── Safety ────────────────────────────────────────────────────────────
COLLISION_DIST        = 0.18
COLLISION_BINS_NEEDED = 2

# ── Episode ───────────────────────────────────────────────────────────
MAX_STEPS = 500

# ── Reward ────────────────────────────────────────────────────────────
R_GOAL      =  200.0
R_COLLISION = -100.0
R_STEP      =   -0.3   # 500 steps = -150 total, close to collision cost
R_PROGRESS  =   30.0


_goal_zone_idx = 0
_current_episode = 0

def set_training_episode(ep):
    """Called by train.py each episode so goal difficulty scales up."""
    global _current_episode
    _current_episode = ep

def _safe_goal():
    """
    CURRICULUM LEARNING — verified against full arena geometry.

    Arena layout (top view):
      y=+1.0  ─────────────── north wall
      y=+0.40  [PLATE_A: x=-1.2 to +0.25]  gap RIGHT (x > +0.25)
      y= 0.00  ✦ ROBOT SPAWN (faces +X)
      y=-0.30  [PLATE_B: x=-0.20 to +1.2]  gap LEFT  (x < -0.20)
      y=-1.0  ─────────────── south wall

    Open zones (NO gap crossing needed):
      Corridor:    -0.18 < y < +0.28,  any x  (between both plates)
      Bottom-left: y < -0.42, x < -0.20       (fully below PLATE_B, left side)

    1-gap zones (ONE gap crossing needed):
      Bottom-left: y < -0.42, x < -0.20  → pass PLATE_B left gap
      Top-right:   y > +0.52, x > +0.25  → pass PLATE_A right gap

    2-gap S-curve (hardest — needs full maze navigation):
      Top-left:    y > +0.52, x < +0.25  → must go: left gap B → right gap A

    ── Phase 1 (ep 1–500): OPEN ONLY ─────────────────────────────────
      Goals in corridor (-0.18 < y < +0.28) AND bottom-left (verified).
      Covers ALL compass directions at 0.55–0.92m distance.
      Zero gap crossings needed. Robot learns:
        • Orient toward any angle
        • Drive forward
        • Stop at goal
      WHY 500 eps: needs ~300 to stabilise, 200 buffer before introducing gaps.

    ── Phase 2 (ep 501–1000): 1-GAP ONLY ─────────────────────────────
      Mixes open goals (50%) with 1-gap goals (50%).
      Open goals kept so network doesn't forget Phase 1.
      New skill: find and pass through a gap.

    ── Phase 3 (ep 1001–2000): ALL GOALS ──────────────────────────────
      Adds top-left S-curve goals (25%) alongside all previous types.
      Random sampling across full arena with plate exclusion.
      Network must combine both gap skills into one path.

    ── Phase 4 (ep 2001–3000): RANDOM FULL ────────────────────────────
      Pure random across entire arena — no fixed lists.
      Maximises generalisation and removes any fixed-goal memorisation.
    """
    global _goal_zone_idx
    _goal_zone_idx += 1
    ep = _current_episode

    # ── Phase 1: ALL DIRECTIONS, close range, ZERO gap crossings ─────
    # ALL goals are inside the corridor: -0.18 < y < +0.28
    # This band sits between PLATE_B (y=-0.30) and PLATE_A (y=+0.40)
    # so ZERO gap crossings are needed regardless of x position.
    #
    # Covers 8 compass directions from (0,0):
    #   RIGHT      0°  (+x,  0)
    #   UP-RIGHT  +22° (+x, +y)  — kept y<+0.28 to stay below PLATE_A
    #   UP-LEFT  +156° (-x, +y)  — same
    #   LEFT      180° (-x,  0)
    #   DOWN-LEFT-156° (-x, -y)  — kept y>-0.18 to stay above PLATE_B
    #   DOWN-RIGHT-11° (+x, -y)  — same
    #
    # VERIFIED by geometry checker — no goal hits any plate or wall.
    if ep <= 500:
        open_goals = [
            # RIGHT — robot faces this directly, easiest start
            ( 0.60,  0.00),   # 0°,    dist=0.60
            ( 0.65,  0.10),   # +9°,   dist=0.66
            # UPPER-RIGHT — small left turn needed
            ( 0.50,  0.20),   # +22°,  dist=0.54  y=0.20 ✅
            ( 0.45,  0.20),   # +24°,  dist=0.49  y=0.20 ✅
            # UPPER-LEFT — large left turn needed
            (-0.50,  0.15),   # +163°, dist=0.52  y=0.15 ✅
            (-0.45,  0.20),   # +156°, dist=0.49  y=0.20 ✅
            # LEFT — 180° turn
            (-0.55,  0.00),   # 180°,  dist=0.55
            (-0.60,  0.00),   # 180°,  dist=0.60
            # LOWER-LEFT — lower-left turn
            (-0.50, -0.15),   # -163°, dist=0.52  y=-0.15 ✅
            (-0.45, -0.15),   # -162°, dist=0.47  y=-0.15 ✅
            # LOWER-RIGHT — slight right-down, above PLATE_B
            ( 0.55, -0.15),   # -15°,  dist=0.57  y=-0.15 ✅
            ( 0.50, -0.10),   # -11°,  dist=0.51  y=-0.10 ✅
        ]
        return open_goals[_goal_zone_idx % len(open_goals)]

    # ── Phase 2: OPEN(50%) + 1-GAP(50%) ───────────────────────────────
    # Open goals kept to prevent catastrophic forgetting of Phase 1.
    # 1-gap goals verified: y < -0.42 OR (y > +0.52 and x > +0.25)
    # REMOVED: (-0.80,-0.80) and (-0.85,-0.65) — path crosses PLATE_B
    elif ep <= 1000:
        if _goal_zone_idx % 2 == 0:
            # Open — identical subset to Phase 1 keeps skill alive
            open_goals = [
                ( 0.60,  0.00), (-0.55,  0.00),
                ( 0.50,  0.20), (-0.50,  0.15),
                (-0.50, -0.15), ( 0.55, -0.15),
            ]
            return open_goals[_goal_zone_idx % len(open_goals)]
        else:
            # 1-gap verified goals only
            one_gap_goals = [
                # Bottom-left: below PLATE_B, left side (left gap)
                (-0.55, -0.65),   # dist=0.85 ✅
                (-0.70, -0.60),   # dist=0.92 ✅
                (-0.60, -0.75),   # dist=0.96 ✅
                (-0.75, -0.55),   # dist=0.93 ✅
                (-0.65, -0.75),   # dist=0.99 ✅
                # Top-right: above PLATE_A, right side (right gap)
                ( 0.60,  0.75),   # dist=0.96 ✅
                ( 0.75,  0.65),   # dist=0.99 ✅
                ( 0.80,  0.80),   # dist=1.13 ✅
                ( 0.50,  0.75),   # dist=0.90 ✅
                ( 0.90,  0.70),   # dist=1.14 ✅
                ( 0.65,  0.85),   # dist=1.07 ✅
            ]
            return one_gap_goals[_goal_zone_idx % len(one_gap_goals)]

    # ── Phase 3: ALL TYPES — 25% each ──────────────────────────────────
    # S-curve top-left: y shifted to +0.75 (was 0.70) to ensure they
    # are clearly above PLATE_A (y=0.40) with good clearance.
    # Direct path from (0,0) to (-0.60,+0.75) DOES cross PLATE_A ✅
    # so robot genuinely must navigate around it.
    elif ep <= 2000:
        r = _goal_zone_idx % 4
        if r == 0:
            # Open — foundation kept alive
            return [( 0.60,  0.00), (-0.55,  0.00),
                    ( 0.50,  0.20), (-0.50, -0.15)][_goal_zone_idx % 4]
        elif r == 1:
            # Bottom 1-gap
            return [(-0.55, -0.65), (-0.70, -0.60), (-0.60, -0.75),
                    (-0.75, -0.55), (-0.65, -0.75)][_goal_zone_idx % 5]
        elif r == 2:
            # Top-right 1-gap
            return [( 0.60,  0.75), ( 0.75,  0.65), ( 0.80,  0.80),
                    ( 0.50,  0.75), ( 0.90,  0.70)][_goal_zone_idx % 5]
        else:
            # S-curve top-left — y=+0.75 ensures above PLATE_A
            return [(-0.60,  0.75), (-0.70,  0.65), (-0.80,  0.80),
                    (-0.50,  0.75), (-0.90,  0.65)][_goal_zone_idx % 5]

    # ── Phase 4: FULL RANDOM — pure generalisation ─────────────────────
    else:
        for _ in range(2000):
            x = random.uniform(ARENA_MIN_X, ARENA_MAX_X)
            y = random.uniform(ARENA_MIN_Y, ARENA_MAX_Y)
            if math.hypot(x, y) < GOAL_MIN_DIST:
                continue
            if abs(y - PLATE_A['y']) < 0.12 and PLATE_A['x_min'] < x < PLATE_A['x_max']:
                continue
            if abs(y - PLATE_B['y']) < 0.12 and PLATE_B['x_min'] < x < PLATE_B['x_max']:
                continue
            return x, y
        # Guaranteed safe fallbacks — one per zone
        fallbacks = [
            ( 0.60,  0.00),   # open corridor
            (-0.55, -0.60),   # bottom-left open
            (-0.70, -0.70),   # bottom 1-gap
            ( 0.75,  0.75),   # top-right 1-gap
            (-0.70,  0.70),   # S-curve
        ]
        return fallbacks[_goal_zone_idx % len(fallbacks)]


class TD3Env(Node):
    def __init__(self):
        super().__init__("td3_env")

        qos = QoSProfile(depth=10,
                         reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.VOLATILE)

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
          <cylinder><radius>0.15</radius><length>0.03</length></cylinder>
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
        progress = (self._prev_dist - dist) / max(self._prev_dist, 0.01)
        reward  += R_PROGRESS * progress
        self._prev_dist = dist

        done = False
        info = {}

        if dist < GOAL_TOLERANCE:
            reward += R_GOAL
            done    = True
            info    = {"result": "goal"}
            self.get_logger().info(
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
