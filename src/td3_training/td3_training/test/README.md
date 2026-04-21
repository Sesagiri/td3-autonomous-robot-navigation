# Navigator Test Suite

pytest tests for the LiDAR reactive navigator state machine.

## What's tested

| Class | Tests | Covers |
|---|---|---|
| TestAlignState | 4 | ALIGN→FORWARD transitions, heading thresholds |
| TestObstacleDetection | 5 | Front/diagonal thresholds, boundary values |
| TestEmergencyStop | 2 | Velocity zeroing on obstacle and goal |
| TestGoalReaching | 3 | ARRIVED state, goal priority over obstacles |
| TestWallRecovery | 2 | WALL→FORWARD when path clears |
| TestBackupState | 3 | BACKUP behaviour and reverse velocity |

**Total: 19 test cases**

## Run

```bash
# From repo root
pip install pytest
pytest tests/ -v
```

## Design

Tests use a pure-Python `NavigatorLogic` class that mirrors the
state machine in `navigator.py` without any ROS2 dependencies.
This means the suite runs on any machine — no ROS2 install,
no robot, no simulation needed.

`conftest.py` mocks all ROS2 modules at the session level.