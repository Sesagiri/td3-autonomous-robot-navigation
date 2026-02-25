#!/usr/bin/env python3

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import rclpy
from rclpy.node import Node
import threading
import math

from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from squaternion import Quaternion
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker, MarkerArray

# ─────────────────────────────────────────────
# CONSTANTS  — must match train.py exactly
# ─────────────────────────────────────────────
GOAL_REACHED_DIST = 0.3       # metres
COLLISION_DIST    = 0.25      # metres  (raw lidar metres)
TIME_DELTA        = 0.2       # seconds
ENVIRONMENT_DIM   = 20

# ─────────────────────────────────────────────
# GLOBALS
# ─────────────────────────────────────────────
last_odom  = None
lidar_data = np.ones(ENVIRONMENT_DIM) * 10.0   # raw metres

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────
# ACTOR  — identical architecture to train.py
# ─────────────────────────────────────────────
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2 = nn.Linear(800, 600)
        self.layer_3 = nn.Linear(600, action_dim)
        self.tanh    = nn.Tanh()

    def forward(self, s):
        s = F.relu(self.layer_1(s))
        s = F.relu(self.layer_2(s))
        return self.tanh(self.layer_3(s))


class td3(object):
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim).to(device)

    def get_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        with torch.no_grad():
            return self.actor(state).cpu().numpy().flatten()

    def load(self, filename, directory):
        path = f"{directory}/{filename}_actor.pth"
        self.actor.load_state_dict(
            torch.load(path, map_location=device)
        )
        self.actor.eval()
        print(f"✅  Model loaded from: {path}")


# ─────────────────────────────────────────────
# OBSTACLE CHECK
# ─────────────────────────────────────────────
def check_pos(x, y):
    if -3.8 > x > -6.2 and 6.2 > y > 3.8:   return False
    if -1.3 > x > -2.7 and 4.7 > y > -0.2:  return False
    if -0.3 > x > -4.2 and 2.7 > y > 1.3:   return False
    if -0.8 > x > -4.2 and -2.3 > y > -4.2: return False
    if -1.3 > x > -3.7 and -0.8 > y > -2.7: return False
    if  4.2 > x >  0.8 and -1.8 > y > -3.2: return False
    if  4.0 > x >  2.5 and  0.7 > y > -3.2: return False
    if  6.2 > x >  3.8 and -3.3 > y > -4.2: return False
    if  4.2 > x >  1.3 and  3.7 > y >  1.5: return False
    if -3.0 > x > -7.2 and  0.5 > y > -1.5: return False
    if x > 4.5 or x < -4.5 or y > 4.5 or y < -4.5: return False
    return True


# ─────────────────────────────────────────────
# GAZEBO ENVIRONMENT
# ─────────────────────────────────────────────
class GazeboEnv(Node):

    def __init__(self):
        super().__init__('env')
        self.environment_dim = ENVIRONMENT_DIM
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.goal_x = 1.0
        self.goal_y = 0.0
        self.upper  = 5.0
        self.lower  = -5.0

        self.set_self_state = ModelState()
        self.set_self_state.model_name       = "r1"
        self.set_self_state.pose.orientation.w = 1.0

        self.vel_pub   = self.create_publisher(Twist,      "/cmd_vel",                1)
        self.set_state = self.create_publisher(ModelState, "gazebo/set_model_state", 10)
        self.unpause   = self.create_client(Empty, "/unpause_physics")
        self.pause     = self.create_client(Empty, "/pause_physics")
        self.reset_proxy = self.create_client(Empty, "/reset_world")
        self.publisher = self.create_publisher(MarkerArray, "goal_point", 3)
        self.publisher2 = self.create_publisher(MarkerArray, "linear_velocity", 1)
        self.publisher3 = self.create_publisher(MarkerArray, "angular_velocity", 1)

    def step(self, action):
        global lidar_data, last_odom
        if last_odom is None:
            return np.zeros(self.environment_dim + 4), 0.0, False, False

        # Send velocity command
        vel_cmd           = Twist()
        vel_cmd.linear.x  = float(action[0])
        vel_cmd.angular.z = float(action[1])
        self.vel_pub.publish(vel_cmd)
        self.publish_markers(action)

        # Unpause
        while not self.unpause.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for /unpause_physics ...")
        self.unpause.call_async(Empty.Request())
        time.sleep(TIME_DELTA)

        # Pause
        while not self.pause.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for /pause_physics ...")
        self.pause.call_async(Empty.Request())

        # Collision check — lidar_data is raw metres, COLLISION_DIST is metres ✅
        done, collision, min_laser = self.observe_collision(lidar_data)

        # Odometry
        self.odom_x = last_odom.pose.pose.position.x
        self.odom_y = last_odom.pose.pose.position.y
        quaternion  = Quaternion(
            last_odom.pose.pose.orientation.w,
            last_odom.pose.pose.orientation.x,
            last_odom.pose.pose.orientation.y,
            last_odom.pose.pose.orientation.z,
        )
        angle = round(quaternion.to_euler(degrees=False)[2], 4)

        distance = np.linalg.norm([self.odom_x - self.goal_x, self.odom_y - self.goal_y])
        skew_x   = self.goal_x - self.odom_x
        skew_y   = self.goal_y - self.odom_y
        mag1     = math.sqrt(skew_x**2 + skew_y**2) + 1e-8
        beta     = math.acos(skew_x / mag1)
        if skew_y < 0:
            beta = -beta
        theta = beta - angle
        if theta >  math.pi: theta -= 2 * math.pi
        if theta < -math.pi: theta += 2 * math.pi

        target = False
        if distance < GOAL_REACHED_DIST:
            self.get_logger().info("✅  GOAL REACHED!")
            target = True
            done   = True

        # ── CRITICAL: normalise lidar before feeding to network ──
        # The model was trained on lidar/10.0, so test must do the same
        lidar_norm  = lidar_data / 10.0
        robot_state = [distance, theta, action[0], action[1]]
        state       = np.append(lidar_norm, robot_state)

        reward = self.get_reward(target, collision, action, min_laser)
        return state, reward, done, target

    def reset(self):
        global lidar_data

        while not self.reset_proxy.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for /reset_world ...")
        self.reset_proxy.call_async(Empty.Request())

        angle      = np.random.uniform(-math.pi, math.pi)
        quaternion = Quaternion.from_euler(0.0, 0.0, angle)
        obj        = self.set_self_state

        x, y = 0.0, 0.0
        while not check_pos(x, y):
            x = np.random.uniform(-4.5, 4.5)
            y = np.random.uniform(-4.5, 4.5)

        obj.pose.position.x    = x
        obj.pose.position.y    = y
        obj.pose.orientation.x = quaternion.x
        obj.pose.orientation.y = quaternion.y
        obj.pose.orientation.z = quaternion.z
        obj.pose.orientation.w = quaternion.w
        self.set_state.publish(obj)

        self.odom_x = x
        self.odom_y = y
        self.change_goal()
        self.random_box()
        self.publish_markers([0.0, 0.0])

        while not self.unpause.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for /unpause_physics ...")
        self.unpause.call_async(Empty.Request())
        time.sleep(TIME_DELTA)

        while not self.pause.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for /pause_physics ...")
        self.pause.call_async(Empty.Request())

        distance = np.linalg.norm([self.odom_x - self.goal_x, self.odom_y - self.goal_y])
        skew_x   = self.goal_x - self.odom_x
        skew_y   = self.goal_y - self.odom_y
        mag1     = math.sqrt(skew_x**2 + skew_y**2) + 1e-8
        beta     = math.acos(skew_x / mag1)
        if skew_y < 0:
            beta = -beta
        theta = beta - angle
        if theta >  math.pi: theta -= 2 * math.pi
        if theta < -math.pi: theta += 2 * math.pi

        # ── CRITICAL: normalise lidar ──
        lidar_norm  = lidar_data / 10.0
        robot_state = [distance, theta, 0.0, 0.0]
        state       = np.append(lidar_norm, robot_state)
        return state

    def change_goal(self):
        goal_ok = False
        while not goal_ok:
            self.goal_x = np.random.uniform(-4.5, 4.5)
            self.goal_y = np.random.uniform(-4.5, 4.5)
            goal_ok     = check_pos(self.goal_x, self.goal_y)
            if np.linalg.norm([self.goal_x - self.odom_x, self.goal_y - self.odom_y]) < 1.0:
                goal_ok = False

    def random_box(self):
        for i in range(4):
            x, y   = 0.0, 0.0
            box_ok = False
            while not box_ok:
                x      = np.random.uniform(-6, 6)
                y      = np.random.uniform(-6, 6)
                box_ok = check_pos(x, y)
                if np.linalg.norm([x - self.odom_x, y - self.odom_y]) < 1.5: box_ok = False
                if np.linalg.norm([x - self.goal_x, y - self.goal_y]) < 1.5: box_ok = False
            bs                    = ModelState()
            bs.model_name         = f"cardboard_box_{i}"
            bs.pose.position.x    = x
            bs.pose.position.y    = y
            bs.pose.orientation.w = 1.0
            self.set_state.publish(bs)

    def publish_markers(self, action):
        # Goal
        ma1    = MarkerArray()
        m      = Marker()
        m.header.frame_id   = "odom"
        m.type              = Marker.CYLINDER
        m.action            = Marker.ADD
        m.scale.x = m.scale.y = 0.1
        m.scale.z           = 0.01
        m.color.a = m.color.g = 1.0
        m.pose.orientation.w = 1.0
        m.pose.position.x   = self.goal_x
        m.pose.position.y   = self.goal_y
        ma1.markers.append(m)
        self.publisher.publish(ma1)

        for pub, val, offset in [
            (self.publisher2, abs(action[0]), 0.0),
            (self.publisher3, abs(action[1]), 0.2),
        ]:
            ma  = MarkerArray()
            mk  = Marker()
            mk.header.frame_id  = "odom"
            mk.type             = Marker.CUBE
            mk.action           = Marker.ADD
            mk.scale.x          = float(val)
            mk.scale.y          = 0.1
            mk.scale.z          = 0.01
            mk.color.a          = 1.0
            mk.color.r          = 1.0
            mk.pose.orientation.w = 1.0
            mk.pose.position.x  = 5.0
            mk.pose.position.y  = offset
            ma.markers.append(mk)
            pub.publish(ma)

    @staticmethod
    def observe_collision(laser_data):
        """laser_data is raw metres. COLLISION_DIST is metres. Comparison is correct."""
        min_laser = float(np.min(laser_data))
        if min_laser < COLLISION_DIST:
            env.get_logger().info(f"💥 Collision! min_laser={min_laser:.3f} m")
            return True, True, min_laser
        return False, False, min_laser

    @staticmethod
    def get_reward(target, collision, action, min_laser):
        if target:
            return 100.0
        if collision:
            return -100.0
        r3 = lambda x: 1 - x if x < 1 else 0.0
        return action[0] / 2 - abs(action[1]) / 2 - r3(min_laser) / 2


# ─────────────────────────────────────────────
# SUBSCRIBER NODES
# ─────────────────────────────────────────────
class Odom_subscriber(Node):
    def __init__(self):
        super().__init__('odom_subscriber')
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

    def odom_callback(self, od_data):
        global last_odom
        last_odom = od_data


class Lidar_subscriber(Node):
    def __init__(self):
        super().__init__('lidar_subscriber')
        self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)

    def lidar_callback(self, msg):
        global lidar_data
        raw = np.array(msg.ranges, dtype=np.float32)
        raw[np.isinf(raw)] = 10.0
        raw[np.isnan(raw)] = 10.0
        raw = np.clip(raw, 0.0, 10.0)
        # Same indices as train.py — front 160°
        indices    = np.linspace(100, 260, ENVIRONMENT_DIM, dtype=int)
        lidar_data = raw[indices]   # raw metres, normalisation happens in step()/reset()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == '__main__':

    rclpy.init(args=None)

    seed           = 0
    max_ep         = 500
    file_name      = "td3_green_4wheel"
    # ✅ Correct path matching your pytorch_models folder
    model_dir      = "/home/shesha/robot_ws/src/td3/scripts/pytorch_models"
    environment_dim = ENVIRONMENT_DIM
    robot_dim       = 4
    state_dim       = environment_dim + robot_dim   # 24
    action_dim      = 2

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load network
    network = td3(state_dim, action_dim)
    try:
        network.load(file_name, model_dir)
    except Exception as e:
        raise ValueError(f"Could not load model from {model_dir}: {e}")

    # Create nodes
    env              = GazeboEnv()
    odom_subscriber  = Odom_subscriber()
    lidar_subscriber = Lidar_subscriber()   # ✅ Fixed: was lidar_subscriber() (lowercase)

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(odom_subscriber)
    executor.add_node(lidar_subscriber)
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    # Wait for sensors
    env.get_logger().info("Waiting for Gazebo sensors ...")
    while last_odom is None or np.all(lidar_data == 10.0):
        time.sleep(0.1)
    env.get_logger().info("Sensors ready. Starting test loop.")

    # ── Test loop ─────────────────────────────
    done              = True
    episode_timesteps = 0
    episode_num       = 0
    state             = None

    try:
        while rclpy.ok():
            if done:
                env.get_logger().info(
                    f"Episode {episode_num} done. Resetting ..."
                )
                state             = env.reset()
                done              = False
                episode_timesteps = 0
                episode_num      += 1

            # Get action — NO exploration noise during testing
            action = network.get_action(np.array(state))
            a_in   = [(action[0] + 1) / 2, action[1]]   # linear: [-1,1]→[0,1]

            next_state, reward, done, target = env.step(a_in)

            env.get_logger().info(
                f"Ep {episode_num} | Step {episode_timesteps:3d} | "
                f"Reward: {reward:6.2f} | Done: {bool(done)} | Target: {target}"
            )

            done = 1 if episode_timesteps + 1 == max_ep else int(done)

            state              = next_state
            episode_timesteps += 1

    except KeyboardInterrupt:
        env.get_logger().info("Test interrupted by user.")

    finally:
        # Stop robot safely
        stop = Twist()
        env.vel_pub.publish(stop)
        rclpy.shutdown()
