"""
goal_gui.py — Interactive Goal Setter GUI for TD3 Robot
========================================================
Run AFTER training is complete. Replaces test_td3.py.

Features:
  - 2D arena map with obstacles drawn
  - LEFT CLICK anywhere on map = set goal, robot navigates there
  - Live robot dot moves as robot navigates in Gazebo
  - Live goal marker appears in Gazebo simultaneously
  - Status panel: position, distance, steps, reward, result
  - Session stats: success rate across all goals you set
  - RESET button: resets robot to (0,0) in Gazebo
  - STOP button: stops robot immediately

Usage:
  # Terminal 1: Launch Gazebo
  ros2 launch td3_training train_rviz.launch.py

  # Terminal 2: Run GUI
  cd ~/ros2_ws/src/td3_training/td3_training/
  python3 goal_gui.py

  # Then click anywhere on the map to send robot to that goal!

Requirements:
  pip install matplotlib --break-system-packages
"""

import rclpy
import threading
import math
import time
import numpy as np
import os

import matplotlib
matplotlib.use('TkAgg')  # works headless-friendly with X11 forwarding too
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
import matplotlib.gridspec as gridspec

from td3_env   import TD3Env, ARENA_MIN_X, ARENA_MAX_X, ARENA_MIN_Y, ARENA_MAX_Y
from td3_env   import PLATE_A, PLATE_B, GOAL_TOLERANCE
from td3_agent import TD3

# ── Config ────────────────────────────────────────────────────────────
MODEL_PATH  = "./models/best"
MAX_STEPS   = 600
UPDATE_HZ   = 10   # GUI refresh rate

# ── Arena geometry for drawing ─────────────────────────────────────────
WALL_X_MIN  = -1.2
WALL_X_MAX  =  1.2
WALL_Y_MIN  = -1.2
WALL_Y_MAX  =  1.2

# Plate A: y=0.40, x=-1.20 to +0.25  (white)
# Plate B: y=-0.30, x=-0.20 to +1.20 (brown)
PLATE_A_DRAW = dict(x=-1.20, y=0.37, w=1.45, h=0.06, color='#e0e0e0', label='PLATE A')
PLATE_B_DRAW = dict(x=-0.20, y=-0.33, w=1.40, h=0.06, color='#c8a060', label='PLATE B')


# ═══════════════════════════════════════════════════════════════════════
#  GUI Class
# ═══════════════════════════════════════════════════════════════════════
class GoalGUI:
    def __init__(self, env, agent):
        self.env   = env
        self.agent = agent

        # ── State ──────────────────────────────────────────────────────
        self.goal_x      = None
        self.goal_y      = None
        self.robot_x     = 0.0
        self.robot_y     = 0.0
        self.robot_head  = 0.0
        self.status      = "Click on map to set a goal"
        self.status_color = 'white'
        self.distance    = 0.0
        self.steps       = 0
        self.ep_reward   = 0.0
        self.running     = False
        self.stop_flag   = False

        # Session stats
        self.total_goals      = 0
        self.total_successes  = 0
        self.total_collisions = 0
        self.total_timeouts   = 0
        self.history          = []   # list of (gx, gy, result)

        # ── Build figure ───────────────────────────────────────────────
        self._build_figure()

    # ──────────────────────────────────────────────────────────────────
    def _build_figure(self):
        self.fig = plt.figure(figsize=(12, 8), facecolor='#1a1a2e')
        self.fig.canvas.manager.set_window_title('TD3 Robot Navigator — Goal GUI')

        gs = gridspec.GridSpec(
            2, 2,
            width_ratios=[2.5, 1],
            height_ratios=[3, 1],
            hspace=0.35,
            wspace=0.3,
            left=0.06, right=0.97,
            top=0.93,  bottom=0.08,
        )

        # ── Arena map (left, tall) ─────────────────────────────────────
        self.ax_map = self.fig.add_subplot(gs[:, 0])
        self._draw_arena()

        # ── Status panel (top right) ───────────────────────────────────
        self.ax_info = self.fig.add_subplot(gs[0, 1])
        self.ax_info.set_facecolor('#0f0f23')
        self.ax_info.axis('off')
        self._init_info_text()

        # ── History panel (bottom right) ───────────────────────────────
        self.ax_hist = self.fig.add_subplot(gs[1, 1])
        self.ax_hist.set_facecolor('#0f0f23')
        self.ax_hist.axis('off')
        self._init_hist_text()

        # ── Buttons ────────────────────────────────────────────────────
        self.ax_btn_reset = plt.axes([0.72, 0.04, 0.11, 0.04])
        self.ax_btn_stop  = plt.axes([0.85, 0.04, 0.11, 0.04])

        from matplotlib.widgets import Button
        self.btn_reset = Button(self.ax_btn_reset, '⟳ RESET',
                                color='#2d4a7a', hovercolor='#3a6090')
        self.btn_stop  = Button(self.ax_btn_stop,  '■ STOP',
                                color='#7a2d2d', hovercolor='#903a3a')
        self.btn_reset.label.set_color('white')
        self.btn_stop.label.set_color('white')
        self.btn_reset.on_clicked(self._on_reset)
        self.btn_stop.on_clicked(self._on_stop)

        # ── Click handler ──────────────────────────────────────────────
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)

        # ── Live elements (updated each frame) ────────────────────────
        # Robot arrow
        self.robot_arrow = None
        # Goal marker
        self.goal_dot,  = self.ax_map.plot([], [], '*', color='#00ff88',
                                            markersize=18, zorder=6)
        # Goal circle (tolerance ring)
        self.goal_circle = plt.Circle((0, 0), GOAL_TOLERANCE,
                                       color='#00ff88', fill=False,
                                       linewidth=1.5, linestyle='--',
                                       alpha=0.6, zorder=5)
        self.ax_map.add_patch(self.goal_circle)
        self.goal_circle.set_visible(False)

        # Robot trail
        self.trail_x = []
        self.trail_y = []
        self.trail_line, = self.ax_map.plot([], [], '-',
                                             color='#4488ff', linewidth=1.2,
                                             alpha=0.5, zorder=4)

        # History dots on map (✅ / 💥 markers)
        self.hist_dots_goal, = self.ax_map.plot([], [], 'o',
                                                  color='#00cc66',
                                                  markersize=6, alpha=0.5,
                                                  zorder=3)
        self.hist_dots_fail, = self.ax_map.plot([], [], 'x',
                                                  color='#ff4444',
                                                  markersize=7, alpha=0.5,
                                                  zorder=3)

        plt.ion()
        plt.show()

    # ──────────────────────────────────────────────────────────────────
    def _draw_arena(self):
        ax = self.ax_map
        ax.set_facecolor('#0d1117')
        ax.set_xlim(WALL_X_MIN - 0.05, WALL_X_MAX + 0.05)
        ax.set_ylim(WALL_Y_MIN - 0.05, WALL_Y_MAX + 0.05)
        ax.set_aspect('equal')
        ax.set_title('Arena Map  —  Left click to set goal',
                     color='white', fontsize=11, pad=8)
        ax.tick_params(colors='#888888')
        ax.spines[:].set_color('#333333')
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_color('#888888')

        # Grid
        ax.grid(True, color='#222233', linewidth=0.5, zorder=0)

        # Arena walls (white rectangle)
        wall = patches.Rectangle(
            (WALL_X_MIN, WALL_Y_MIN),
            WALL_X_MAX - WALL_X_MIN,
            WALL_Y_MAX - WALL_Y_MIN,
            linewidth=3, edgecolor='#dddddd',
            facecolor='none', zorder=2
        )
        ax.add_patch(wall)

        # PLATE A (white)
        p = PLATE_A_DRAW
        plate_a = patches.Rectangle(
            (p['x'], p['y']), p['w'], p['h'],
            linewidth=1, edgecolor='#aaaaaa',
            facecolor=p['color'], zorder=3, alpha=0.9
        )
        ax.add_patch(plate_a)
        ax.text(p['x'] + p['w']/2, p['y'] + p['h'] + 0.04,
                p['label'], color='#aaaaaa', fontsize=7,
                ha='center', zorder=4)

        # Gap annotation for Plate A
        ax.annotate('', xy=(WALL_X_MAX - 0.05, p['y'] + p['h']/2),
                    xytext=(p['x'] + p['w'] + 0.03, p['y'] + p['h']/2),
                    arrowprops=dict(arrowstyle='<->', color='#44aa88',
                                   lw=1.2), zorder=4)
        ax.text(WALL_X_MAX - 0.38, p['y'] + p['h'] + 0.04,
                'gap →', color='#44aa88', fontsize=7, zorder=4)

        # PLATE B (brown)
        p = PLATE_B_DRAW
        plate_b = patches.Rectangle(
            (p['x'], p['y']), p['w'], p['h'],
            linewidth=1, edgecolor='#996633',
            facecolor=p['color'], zorder=3, alpha=0.9
        )
        ax.add_patch(plate_b)
        ax.text(p['x'] + p['w']/2, p['y'] - 0.08,
                p['label'], color='#c8a060', fontsize=7,
                ha='center', zorder=4)

        # Gap annotation for Plate B
        ax.annotate('', xy=(WALL_X_MIN + 0.05, p['y'] + p['h']/2),
                    xytext=(p['x'] - 0.03, p['y'] + p['h']/2),
                    arrowprops=dict(arrowstyle='<->', color='#44aa88',
                                   lw=1.2), zorder=4)
        ax.text(WALL_X_MIN + 0.08, p['y'] - 0.08,
                '← gap', color='#44aa88', fontsize=7, zorder=4)

        # Robot spawn marker
        ax.plot(0, 0, 's', color='#aaaaff', markersize=8,
                alpha=0.4, zorder=2)
        ax.text(0.05, -0.08, 'spawn', color='#aaaaff',
                fontsize=7, alpha=0.6, zorder=2)

        # Axis labels
        ax.set_xlabel('X (metres)', color='#888888', fontsize=9)
        ax.set_ylabel('Y (metres)', color='#888888', fontsize=9)

    # ──────────────────────────────────────────────────────────────────
    def _init_info_text(self):
        ax = self.ax_info
        ax.set_title('Status', color='white', fontsize=10, pad=6)
        self.info_text = ax.text(
            0.05, 0.95, self._info_str(),
            transform=ax.transAxes,
            color='white', fontsize=9,
            verticalalignment='top',
            fontfamily='monospace'
        )

    def _init_hist_text(self):
        ax = self.ax_hist
        ax.set_title('Session Stats', color='white', fontsize=10, pad=6)
        self.hist_text = ax.text(
            0.05, 0.95, self._hist_str(),
            transform=ax.transAxes,
            color='white', fontsize=9,
            verticalalignment='top',
            fontfamily='monospace'
        )

    # ──────────────────────────────────────────────────────────────────
    def _info_str(self):
        gx = f'{self.goal_x:+.2f}' if self.goal_x is not None else '  ---'
        gy = f'{self.goal_y:+.2f}' if self.goal_y is not None else '  ---'
        lines = [
            f'Status:',
            f'  {self.status}',
            f'',
            f'Robot pos:',
            f'  x = {self.robot_x:+.3f} m',
            f'  y = {self.robot_y:+.3f} m',
            f'  θ = {math.degrees(self.robot_head):+.1f}°',
            f'',
            f'Goal:',
            f'  x = {gx} m',
            f'  y = {gy} m',
            f'',
            f'Distance: {self.distance:.3f} m',
            f'Steps:    {self.steps}',
            f'Reward:   {self.ep_reward:+.1f}',
        ]
        return '\n'.join(lines)

    def _hist_str(self):
        total = self.total_goals
        sr = (self.total_successes / total * 100) if total > 0 else 0
        lines = [
            f'Goals set:   {total}',
            f'✅ Success:  {self.total_successes}  ({sr:.0f}%)',
            f'💥 Crash:    {self.total_collisions}',
            f'⏱  Timeout:  {self.total_timeouts}',
            f'',
        ]
        # Last 4 results
        if self.history:
            lines.append('Last results:')
            for gx, gy, res in self.history[-4:][::-1]:
                icon = '✅' if res == 'goal' else ('💥' if res == 'collision' else '⏱ ')
                lines.append(f'  {icon} ({gx:+.2f},{gy:+.2f})')
        return '\n'.join(lines)

    # ──────────────────────────────────────────────────────────────────
    def _on_click(self, event):
        """Left click on arena map sets new goal."""
        if event.inaxes != self.ax_map:
            return
        if event.button != 1:  # left click only
            return
        if self.running:
            print("⚠️  Robot is still navigating — stop it first or wait")
            return

        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return

        # Clamp to usable arena area
        x = max(ARENA_MIN_X, min(ARENA_MAX_X, x))
        y = max(ARENA_MIN_Y, min(ARENA_MAX_Y, y))

        # Check not inside a plate
        if (abs(y - PLATE_A['y']) < 0.12 and
                PLATE_A['x_min'] < x < PLATE_A['x_max']):
            self._set_status("❌ That's inside PLATE A! Click elsewhere", 'orange')
            self._refresh_gui()
            return
        if (abs(y - PLATE_B['y']) < 0.12 and
                PLATE_B['x_min'] < x < PLATE_B['x_max']):
            self._set_status("❌ That's inside PLATE B! Click elsewhere", 'orange')
            self._refresh_gui()
            return

        print(f"\n🎯 Goal set: ({x:+.3f}, {y:+.3f})")
        self.goal_x = x
        self.goal_y = y
        self.trail_x.clear()
        self.trail_y.clear()

        # Launch navigation in background thread
        t = threading.Thread(target=self._run_episode, daemon=True)
        t.start()

    # ──────────────────────────────────────────────────────────────────
    def _on_reset(self, event):
        """Reset robot to (0,0) in Gazebo."""
        if self.running:
            self.stop_flag = True
            time.sleep(0.3)
        self._set_status("Resetting robot to (0,0)...", 'cyan')
        self._refresh_gui()
        threading.Thread(target=self._do_reset, daemon=True).start()

    def _do_reset(self):
        self.env._pub_cmd(0.0, 0.0)
        self.env.reset()
        self.robot_x = self.env._pos_x
        self.robot_y = self.env._pos_y
        self.trail_x.clear()
        self.trail_y.clear()
        self.steps     = 0
        self.ep_reward = 0.0
        self.goal_x    = None
        self.goal_y    = None
        self._set_status("Reset done! Click to set a new goal", 'white')

    def _on_stop(self, event):
        """Stop robot immediately."""
        self.stop_flag = True
        self.env._pub_cmd(0.0, 0.0)
        self._set_status("⏹  Stopped. Click to set new goal.", 'orange')

    # ──────────────────────────────────────────────────────────────────
    def _run_episode(self):
        """Run one navigation episode to the clicked goal."""
        self.running   = True
        self.stop_flag = False
        self.steps     = 0
        self.ep_reward = 0.0
        result         = 'timeout'

        gx, gy = self.goal_x, self.goal_y
        self.total_goals += 1

        self._set_status(f"🟢 Navigating to ({gx:+.2f}, {gy:+.2f})", '#00ff88')

        # Inject goal into environment
        obs = self.env.reset(goal_override=(gx, gy))

        for step in range(MAX_STEPS):
            if self.stop_flag:
                result = 'stopped'
                break

            self.steps = step + 1

            # Get robot position for trail
            self.robot_x   = self.env._pos_x
            self.robot_y   = self.env._pos_y
            self.robot_head = self.env._heading
            self.trail_x.append(self.robot_x)
            self.trail_y.append(self.robot_y)
            self.distance = math.hypot(gx - self.robot_x, gy - self.robot_y)

            # Pure policy — no noise
            action = self.agent.select_action(obs)
            obs, reward, done, info = self.env.step(action)
            self.ep_reward += reward
            result = info.get('result', 'timeout')

            if done:
                break

        # Episode finished
        self.robot_x    = self.env._pos_x
        self.robot_y    = self.env._pos_y
        self.robot_head = self.env._heading
        self.env._pub_cmd(0.0, 0.0)

        # Update stats
        self.history.append((gx, gy, result))
        if result == 'goal':
            self.total_successes += 1
            self._set_status(
                f"✅ GOAL REACHED!  {self.steps} steps  reward={self.ep_reward:+.0f}",
                '#00ff88'
            )
        elif result == 'collision':
            self.total_collisions += 1
            self._set_status(
                f"💥 COLLISION at step {self.steps}  reward={self.ep_reward:+.0f}",
                '#ff4444'
            )
        else:
            self.total_timeouts += 1
            self._set_status(
                f"⏱  TIMEOUT ({self.steps} steps)  reward={self.ep_reward:+.0f}",
                '#ffaa00'
            )

        self.running = False

    # ──────────────────────────────────────────────────────────────────
    def _set_status(self, msg, color='white'):
        self.status       = msg
        self.status_color = color

    # ──────────────────────────────────────────────────────────────────
    def _refresh_gui(self):
        """Update all live elements on the plot."""

        # ── Robot arrow ───────────────────────────────────────────────
        if self.robot_arrow:
            self.robot_arrow.remove()
            self.robot_arrow = None

        arrow_len = 0.12
        dx = arrow_len * math.cos(self.robot_head)
        dy = arrow_len * math.sin(self.robot_head)
        self.robot_arrow = FancyArrowPatch(
            (self.robot_x, self.robot_y),
            (self.robot_x + dx, self.robot_y + dy),
            arrowstyle='->', color='#66aaff',
            mutation_scale=15, linewidth=2, zorder=7
        )
        self.ax_map.add_patch(self.robot_arrow)

        # Robot body dot
        if not hasattr(self, 'robot_dot'):
            self.robot_dot, = self.ax_map.plot(
                [self.robot_x], [self.robot_y],
                'o', color='#4488ff', markersize=12, zorder=7
            )
        else:
            self.robot_dot.set_data([self.robot_x], [self.robot_y])

        # ── Goal star + tolerance circle ──────────────────────────────
        if self.goal_x is not None:
            self.goal_dot.set_data([self.goal_x], [self.goal_y])
            self.goal_circle.set_center((self.goal_x, self.goal_y))
            self.goal_circle.set_visible(True)
        else:
            self.goal_dot.set_data([], [])
            self.goal_circle.set_visible(False)

        # ── Trail ─────────────────────────────────────────────────────
        self.trail_line.set_data(self.trail_x, self.trail_y)

        # ── History dots ──────────────────────────────────────────────
        gx_ok = [h[0] for h in self.history if h[2] == 'goal']
        gy_ok = [h[1] for h in self.history if h[2] == 'goal']
        gx_fail = [h[0] for h in self.history if h[2] != 'goal']
        gy_fail = [h[1] for h in self.history if h[2] != 'goal']
        self.hist_dots_goal.set_data(gx_ok, gy_ok)
        self.hist_dots_fail.set_data(gx_fail, gy_fail)

        # ── Status color ──────────────────────────────────────────────
        self.info_text.set_text(self._info_str())
        self.info_text.set_color(self.status_color)
        self.hist_text.set_text(self._hist_str())

        # ── Title bar success rate ─────────────────────────────────────
        sr = (self.total_successes / self.total_goals * 100
              if self.total_goals > 0 else 0)
        self.fig.suptitle(
            f'TD3 Robot Navigator    '
            f'Session: {self.total_successes}/{self.total_goals} goals  '
            f'({sr:.0f}% success)',
            color='white', fontsize=11, y=0.97
        )

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    # ──────────────────────────────────────────────────────────────────
    def run(self):
        """Main GUI loop — keeps refreshing at UPDATE_HZ."""
        print("\n" + "="*60)
        print("  TD3 Goal GUI Ready!")
        print("  Left click on the arena map to set a goal")
        print("  Robot will navigate there in Gazebo + shown on map")
        print("  RESET button → robot returns to (0,0)")
        print("  STOP button  → immediately stop robot")
        print("="*60 + "\n")

        try:
            while plt.fignum_exists(self.fig.number):
                # Sync robot position from env even when not navigating
                if not self.running:
                    try:
                        rclpy.spin_once(self.env, timeout_sec=0.02)
                        self.robot_x    = self.env._pos_x
                        self.robot_y    = self.env._pos_y
                        self.robot_head = self.env._heading
                    except Exception:
                        pass

                self._refresh_gui()
                plt.pause(1.0 / UPDATE_HZ)

        except KeyboardInterrupt:
            pass
        finally:
            print("\n👋 GUI closed. Stopping robot...")
            self.env._pub_cmd(0.0, 0.0)
            self.env.destroy_node()
            rclpy.shutdown()


# ═══════════════════════════════════════════════════════════════════════
#  Patch TD3Env to accept goal_override in reset()
# ═══════════════════════════════════════════════════════════════════════
def _patched_reset(self, goal_override=None):
    """
    Extended reset that accepts an optional (x, y) goal override.
    When goal_override is given, skips _safe_goal() and uses that position.
    """
    import math as _math
    from rclpy.node import Node
    from std_srvs.srv import Empty
    import rclpy as _rclpy

    self._episode += 1
    self._step_count = 0

    # Stop robot
    self._pub_cmd(0.0, 0.0)

    # Set goal BEFORE reset so marker spawns at right place
    if goal_override is not None:
        self._goal_x, self._goal_y = goal_override
    else:
        from td3_env import _safe_goal
        self._goal_x, self._goal_y = _safe_goal()

    self.get_logger().info(
        f"🎯 Ep {self._episode} — New goal: ({self._goal_x:.2f}, {self._goal_y:.2f})"
    )

    # Reset Gazebo world
    future = self.reset_world_srv.call_async(Empty.Request())
    _rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
    self._spin_wait(0.8)

    # Spawn goal marker at clicked position
    self._move_goal_marker(self._goal_x, self._goal_y)

    # Wait for sensors
    self._wait_sensors()

    self._prev_dist  = _math.hypot(
        self._goal_x - self._pos_x,
        self._goal_y - self._pos_y
    )

    return self._build_obs()


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    # ── Check model exists ────────────────────────────────────────────
    if not os.path.exists(f"{MODEL_PATH}/TD3_actor.pth"):
        print(f"\n❌ No model found at {MODEL_PATH}/TD3_actor.pth")
        print("   Train first with: python3 train.py")
        print("   Or specify path:  MODEL_PATH = './models'  in goal_gui.py")
        return

    # ── Init ROS2 + environment ───────────────────────────────────────
    rclpy.init()
    env   = TD3Env()
    agent = TD3()
    agent.load(MODEL_PATH)

    # Patch reset to support goal_override
    import types
    env.reset = types.MethodType(_patched_reset, env)

    print(f"\n✅ Model loaded from: {MODEL_PATH}")
    print(f"   Arena: {ARENA_MIN_X} to {ARENA_MAX_X} m (X),"
          f" {ARENA_MIN_Y} to {ARENA_MAX_Y} m (Y)")
    print(f"   Goal tolerance: {GOAL_TOLERANCE} m")

    # ── Launch GUI ────────────────────────────────────────────────────
    gui = GoalGUI(env, agent)
    gui.run()


if __name__ == "__main__":
    main()
