# TD3 Autonomous Robot Navigation
### Sim-to-Real Deep Reinforcement Learning on ROS2

Trained TD3 (Twin Delayed DDPG) agent in Gazebo simulation and
deployed reactive LiDAR navigator on a physical 4-wheeled skid-steer
robot running ROS2 Jazzy on Raspberry Pi 5.

---

## Demo
🎥 **[Watch demo — YouTube][https://youtu.be/4_c40mNnxdI]**

| Physical Robot |
|---|
| <img width="786" height="670" alt="image" src="https://github.com/user-attachments/assets/3c43aea7-3472-4966-a338-f53885cf0ef8" />|

---

## System Architecture

```
Training Machine (ROS2 Humble + Gazebo)
         ↓  trained model weights
Raspberry Pi 5 (ROS2 Jazzy)
    ├── RPLIDAR A1    →  /scan
    ├── MPU6050 IMU   →  /imu/data
    ├── LM393 Encoders → /odom
    └── Arduino Motor Shield ← /cmd_vel
```

**Deployment repo (all Pi 5 ROS2 nodes):**
→ [td3-robot-deployment-ros2](https://github.com/Sesagiri/td3-robot-deployment-ros2)

---

## Hardware

| Component | Details |
|---|---|
| Compute | Raspberry Pi 5 (8GB), ROS2 Jazzy |
| LiDAR | RPLIDAR A1M8 — 360°, 8000 samples/sec |
| IMU | MPU6050 — 6-axis, I2C |
| Motor Controller | Arduino Uno + Motor Shield (/dev/ttyACM0) |
| Encoders | LM393 — rear wheels, GPIO 24/25 |
| Chassis | 4-wheel skid-steer, 3D-printed mounts and heat inserts |

---

## Algorithm — TD3

**Observation space (26-dim):**
- 24 LiDAR range bins (360° scan, normalized 0–1)
- Normalized distance to goal
- Bearing angle to goal

**Action space (2-dim):** Linear velocity, Angular velocity

**Reward:** +100 goal reached · −100 collision · distance shaping

---

## Training Setup

| Parameter | Value |
|---|---|
| Algorithm | TD3 (Twin Delayed DDPG) |
| Simulator | Gazebo, ROS2 Humble |
| Arena | 2×2m |
| Curriculum stages | 4 progressive environments |
| Training episodes | 3000 |
| Training machine | Ryz 7, 16GB RAM, RTX 3050 |
| Framework | PyTorch |

---

## Navigation State Machine

Deployed navigator uses a reactive state machine —
no map, no global planner, just LiDAR + IMU.

```
ALIGN → FORWARD → WALL → BACKUP → ARRIVED
```

| State | Behaviour |
|---|---|
| ALIGN | Rotate toward goal using IMU yaw |
| FORWARD | Drive toward goal, monitor LiDAR |
| WALL | Turn away from obstacle |
| BACKUP | Reverse if trapped |
| ARRIVED | Stop within 0.15m of goal |

Obstacle thresholds: **0.55m front, 0.40m diagonal**

---

## EKF Sensor Fusion

6-state EKF: `[x, y, yaw, vx, vy, yaw_rate]`
- Prediction: IMU acceleration + angular rate
- Correction: LiDAR scan-match odometry
- Output: `/ekf/odom` (nav_msgs/Odometry)

Replaces dead-reckoning for accurate pose during turns.

---

## HIL Deployment

This project implements a Hardware-in-the-Loop (HIL) setup:

```
Physical Sensors (RPLIDAR + MPU6050)
         ↓  real sensor data via ROS2 topics
Navigator Control Loop (Pi 5)
         ↓  velocity commands
Arduino Motor Shield → Physical Robot Motion
         ↑
LM393 Encoders → feedback to /odom
```

Simulation training (Gazebo) = SIL (Software-in-the-Loop)
Physical deployment (Pi 5) = HIL (Hardware-in-the-Loop)

---

## Sim-to-Real Bugs Resolved

| Bug | Symptom | Fix |
|---|---|---|
| LiDAR upside-down | Robot spun on startup | Negated angle calculation |
| Dead-zone at bin 22 (330°) | False obstacle at 330° | Filtered bin 22 |
| Motor order wrong | Wrong wheels moved | Corrected to [BR, BL, FL, FR] |
| Reverse PWM stuck | Emergency stop never cleared | Rewrote with negative PWM values |
| Encoder indices wrong | Odometry drifted immediately | Fixed to rear indices 2 and 3 (GPIO 24/25) |

---

## How to Run

### Training (ROS2 Humble + Gazebo)
```bash
git clone https://github.com/Sesagiri/td3-autonomous-robot-navigation
cd td3-autonomous-robot-navigation
colcon build && source install/setup.bash
ros2 launch td3 train.launch.py
```

### Evaluation
```bash
ros2 run td3 goal_gui.py
```

### Real robot → see deployment repo
[github.com/Sesagiri/td3-robot-deployment-ros2](https://github.com/Sesagiri/td3-robot-deployment-ros2)

---

## Team

| Member |
|---|
| Sesagiri K R |
| B S Yaathish Kanna |
| Hariharan T |
| Pranav K |

Guide: Dr. K. I. Ramachandran
B.Tech Automation & Robotics Engineering
Amrita School of Engineering, Coimbatore — Batch 2026
