"""
test_td3.py — Test Saved TD3 Model in Gazebo
=============================================
Run this to evaluate your saved model WITHOUT training.

Shows:
  - Success rate over N test episodes
  - Average steps to goal
  - Average reward
  - Which goals it fails on

Usage:
  python3 test_td3.py                    # test best model, 20 episodes
  python3 test_td3.py --episodes 50      # test 50 episodes
  python3 test_td3.py --model ./models/  # test specific model

Run Gazebo first:
  ros2 launch td3_training train_rviz.launch.py
"""

import rclpy
import numpy as np
import argparse
import os

from td3_env   import TD3Env
from td3_agent import TD3
from actor_critic import ReplayBuffer, STATE_DIM, ACTION_DIM

# ── Test config ───────────────────────────────────────────────────────
DEFAULT_EPISODES  = 20
DEFAULT_MODEL     = "./models/best"   # use best saved model
MAX_STEPS         = 500


def test(model_path, num_episodes):
    rclpy.init()
    env   = TD3Env()
    agent = TD3()

    # Load saved model
    if not os.path.exists(f"{model_path}/TD3_actor.pth"):
        print(f"❌ No model found at {model_path}/TD3_actor.pth")
        print(f"   Available models:")
        for d in ["./models/best", "./models"]:
            if os.path.exists(f"{d}/TD3_actor.pth"):
                print(f"   ✅ {d}/TD3_actor.pth")
        return

    agent.load(model_path)
    print(f"\n✅ Loaded model from: {model_path}")
    print(f"   Testing for {num_episodes} episodes (NO training, NO noise)\n")
    print("="*65)
    print(f"{'Ep':>4} {'Goal':>14} {'Steps':>6} {'Reward':>8} {'Result':>10}")
    print("="*65)

    # ── Stats tracking ─────────────────────────────────────────────────
    results      = []
    all_rewards  = []
    all_steps    = []
    goal_count   = 0
    collision_count = 0
    timeout_count   = 0

    for ep in range(1, num_episodes + 1):
        obs      = env.reset()
        gx, gy   = env._goal_x, env._goal_y
        ep_reward = 0.0
        ep_steps  = 0
        result    = "timeout"

        for _ in range(MAX_STEPS):
            ep_steps += 1

            # Pure policy — NO exploration noise during testing
            action = agent.select_action(obs)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            result     = info.get("result", "timeout")

            if done:
                break

        # Track stats
        all_rewards.append(ep_reward)
        all_steps.append(ep_steps)
        results.append(result)

        if result == "goal":
            goal_count += 1
            status = "✅ GOAL"
        elif result == "collision":
            collision_count += 1
            status = "💥 CRASH"
        else:
            timeout_count += 1
            status = "⏱  TIME"

        success_rate = goal_count / ep * 100
        print(
            f"{ep:>4}  ({gx:+.2f},{gy:+.2f})  "
            f"{ep_steps:>5}  {ep_reward:>8.1f}  {status}   "
            f"[{success_rate:.0f}%]"
        )

    # ── Final summary ──────────────────────────────────────────────────
    print("="*65)
    print(f"\n📊 TEST RESULTS ({num_episodes} episodes)")
    print(f"   Model:         {model_path}")
    print(f"   Success rate:  {goal_count/num_episodes*100:.1f}%  "
          f"({goal_count}/{num_episodes} goals reached)")
    print(f"   Collisions:    {collision_count/num_episodes*100:.1f}%  "
          f"({collision_count} episodes)")
    print(f"   Timeouts:      {timeout_count/num_episodes*100:.1f}%  "
          f"({timeout_count} episodes)")
    print(f"   Avg reward:    {np.mean(all_rewards):.1f}  "
          f"(best: {max(all_rewards):.1f})")

    # Only count steps for successful episodes
    success_steps = [all_steps[i] for i, r in enumerate(results) if r == "goal"]
    if success_steps:
        print(f"   Avg steps:     {np.mean(success_steps):.0f} steps to goal  "
              f"(fastest: {min(success_steps)})")

    # ── Deployment readiness ───────────────────────────────────────────
    print()
    sr = goal_count / num_episodes * 100
    if sr >= 80:
        print("🚀 READY FOR PI 5 DEPLOYMENT! Success rate ≥ 80%")
    elif sr >= 60:
        print("⚠️  BORDERLINE — consider training more episodes (target 80%+)")
    else:
        print("❌ NOT READY — train more episodes before deploying to Pi 5")

    print()
    env._pub_cmd(0.0, 0.0)
    env.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test TD3 model in Gazebo")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES,
                        help="Number of test episodes")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="Path to model directory")
    args = parser.parse_args()

    test(args.model, args.episodes)
