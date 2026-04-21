"""
pytest suite for LiDAR reactive navigator state machine.

Tests cover:
- State transitions (ALIGN, FORWARD, WALL, BACKUP, ARRIVED)
- Obstacle detection thresholds (0.55m front, 0.40m diagonal)
- Emergency stop and velocity zeroing
- Goal reaching logic
- Boundary conditions at exact threshold values

Run with: pytest tests/ -v
"""
import pytest


class NavigatorLogic:
    """
    Pure-Python state machine extracted from navigator.py.
    No ROS2 dependencies — fully testable standalone.
    Mirrors the logic deployed on Raspberry Pi 5.
    """
    FRONT_THRESHOLD = 0.55   # metres
    DIAG_THRESHOLD  = 0.40   # metres
    GOAL_THRESHOLD  = 0.15   # metres
    HEADING_THRESH  = 10.0   # degrees

    def __init__(self):
        self.state        = "ALIGN"
        self.heading_error = 0.0
        self.front_dist   = 10.0
        self.diag_dist    = 10.0
        self.goal_dist    = 10.0
        self.linear_vel   = 0.0
        self.angular_vel  = 0.0

    def step(self):
        """Single control loop tick."""
        # Goal check takes priority over everything
        if self.goal_dist < self.GOAL_THRESHOLD:
            self.state       = "ARRIVED"
            self.linear_vel  = 0.0
            self.angular_vel = 0.0
            return

        if self.state == "ALIGN":
            if abs(self.heading_error) < self.HEADING_THRESH:
                self.state = "FORWARD"

        elif self.state == "FORWARD":
            self.linear_vel = 0.15
            if (self.front_dist < self.FRONT_THRESHOLD or
                    self.diag_dist < self.DIAG_THRESHOLD):
                self.state      = "WALL"
                self.linear_vel = 0.0

        elif self.state == "WALL":
            self.angular_vel = 0.5
            if (self.front_dist >= self.FRONT_THRESHOLD and
                    self.diag_dist >= self.DIAG_THRESHOLD):
                self.state = "FORWARD"

        elif self.state == "BACKUP":
            self.linear_vel = -0.1
            if self.front_dist >= self.FRONT_THRESHOLD:
                self.state = "ALIGN"


# ── Fixtures ────────────────────────────────────────────────────

@pytest.fixture
def nav():
    """Fresh NavigatorLogic instance for each test."""
    return NavigatorLogic()

@pytest.fixture
def nav_forward(nav):
    """Navigator already in FORWARD state, path clear."""
    nav.state      = "FORWARD"
    nav.front_dist = 1.0
    nav.diag_dist  = 1.0
    nav.goal_dist  = 5.0
    return nav


# ── ALIGN state ─────────────────────────────────────────────────

class TestAlignState:
    def test_transitions_to_forward_when_aligned(self, nav):
        """ALIGN → FORWARD when heading error < 10°."""
        nav.heading_error = 5.0
        nav.step()
        assert nav.state == "FORWARD"

    def test_stays_align_when_heading_too_large(self, nav):
        """ALIGN stays when heading error ≥ 10°."""
        nav.heading_error = 25.0
        nav.step()
        assert nav.state == "ALIGN"

    def test_transitions_just_below_threshold(self, nav):
        """ALIGN → FORWARD at 9.9° (just under 10° threshold)."""
        nav.heading_error = 9.9
        nav.step()
        assert nav.state == "FORWARD"

    def test_stays_at_exact_threshold(self, nav):
        """ALIGN stays at exactly 10.0° (boundary — not below)."""
        nav.heading_error = 10.0
        nav.step()
        assert nav.state == "ALIGN"


# ── Obstacle detection ───────────────────────────────────────────

class TestObstacleDetection:
    def test_front_obstacle_triggers_wall(self, nav_forward):
        """FORWARD → WALL when front obstacle < 0.55m."""
        nav_forward.front_dist = 0.40
        nav_forward.step()
        assert nav_forward.state == "WALL"

    def test_diagonal_obstacle_triggers_wall(self, nav_forward):
        """FORWARD → WALL when diagonal obstacle < 0.40m."""
        nav_forward.diag_dist = 0.30
        nav_forward.step()
        assert nav_forward.state == "WALL"

    def test_clear_path_stays_forward(self, nav_forward):
        """FORWARD stays FORWARD with no obstacles."""
        nav_forward.step()
        assert nav_forward.state == "FORWARD"

    def test_front_at_exact_threshold_is_clear(self, nav_forward):
        """0.55m exactly is NOT an obstacle (threshold is strict <)."""
        nav_forward.front_dist = 0.55
        nav_forward.step()
        assert nav_forward.state == "FORWARD"

    def test_diagonal_at_exact_threshold_is_clear(self, nav_forward):
        """0.40m exactly is NOT an obstacle (threshold is strict <)."""
        nav_forward.diag_dist = 0.40
        nav_forward.step()
        assert nav_forward.state == "FORWARD"


# ── Emergency stop ───────────────────────────────────────────────

class TestEmergencyStop:
    def test_linear_vel_zero_when_obstacle_detected(self, nav_forward):
        """Linear velocity stops immediately on obstacle."""
        nav_forward.front_dist = 0.30
        nav_forward.step()
        assert nav_forward.linear_vel == 0.0

    def test_both_velocities_zero_on_arrival(self, nav_forward):
        """Both linear and angular velocity zero at goal."""
        nav_forward.goal_dist = 0.10
        nav_forward.step()
        assert nav_forward.linear_vel  == 0.0
        assert nav_forward.angular_vel == 0.0


# ── Goal reaching ────────────────────────────────────────────────

class TestGoalReaching:
    def test_arrived_within_threshold(self, nav_forward):
        """ARRIVED when within 0.15m of goal."""
        nav_forward.goal_dist = 0.10
        nav_forward.step()
        assert nav_forward.state == "ARRIVED"

    def test_not_arrived_outside_threshold(self, nav_forward):
        """Not ARRIVED when outside 0.15m."""
        nav_forward.goal_dist = 0.20
        nav_forward.step()
        assert nav_forward.state != "ARRIVED"

    def test_goal_overrides_obstacle(self, nav_forward):
        """Goal reached takes priority even if obstacle present."""
        nav_forward.goal_dist  = 0.05
        nav_forward.front_dist = 0.20   # would normally trigger WALL
        nav_forward.step()
        assert nav_forward.state == "ARRIVED"


# ── WALL recovery ────────────────────────────────────────────────

class TestWallRecovery:
    def test_wall_clears_to_forward(self, nav):
        """WALL → FORWARD when both front and diagonal clear."""
        nav.state      = "WALL"
        nav.front_dist = 1.0
        nav.diag_dist  = 1.0
        nav.step()
        assert nav.state == "FORWARD"

    def test_wall_stays_while_blocked(self, nav):
        """WALL stays when front still blocked."""
        nav.state      = "WALL"
        nav.front_dist = 0.30
        nav.step()
        assert nav.state == "WALL"


# ── BACKUP state ─────────────────────────────────────────────────

class TestBackupState:
    def test_backup_returns_to_align_when_clear(self, nav):
        """BACKUP → ALIGN when front distance clears."""
        nav.state      = "BACKUP"
        nav.front_dist = 1.0
        nav.step()
        assert nav.state == "ALIGN"

    def test_backup_stays_when_still_blocked(self, nav):
        """BACKUP stays when still too close to obstacle."""
        nav.state      = "BACKUP"
        nav.front_dist = 0.30
        nav.step()
        assert nav.state == "BACKUP"

    def test_backup_velocity_is_negative(self, nav):
        """Backup moves in reverse (negative linear velocity)."""
        nav.state      = "BACKUP"
        nav.front_dist = 0.30
        nav.step()
        assert nav.linear_vel < 0