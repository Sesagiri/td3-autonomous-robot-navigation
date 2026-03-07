"""
train.py — Main TD3 Training Script
=====================================
Run:  python3 train.py

What happens each episode:
  1. Robot resets to (0, 0)
  2. A NEW random goal is picked inside the arena
  3. Robot tries to reach it using current policy
  4. Neural network improves from the experience
  5. Repeat 500+ times

After training you get:
  models/TD3_actor.pth   ← copy this to Pi 5
  models/best/TD3_actor.pth ← best performing version
"""

import rclpy
import numpy as np
import os

from td3_env   import TD3Env
from td3_agent import TD3
from actor_critic import ReplayBuffer, STATE_DIM, ACTION_DIM

# ── Hyperparameters ───────────────────────────────────────────────────
<<<<<<< HEAD
MAX_EPISODES      = 600      # increase if needed
BATCH_SIZE        = 256
REPLAY_START      = 1000     # random exploration before training starts
EXPLORATION_NOISE = 0.15     # noise added to actions during training
SAVE_EVERY        = 10       # save checkpoint every N episodes
RESUME_TRAINING   = False    # set True to continue from checkpoint
MAX_EPISODES      = 1500     # enough for obstacle avoidance to develop
=======
MAX_EPISODES      = 2000     # enough for obstacle avoidance to develop
>>>>>>> 16e6eea (modified world , training)
BATCH_SIZE        = 256
REPLAY_START      = 500      # reduced: start learning after 500 steps not 1000
EXPLORATION_NOISE = 0.20     # noise added to actions during training
SAVE_EVERY        = 10       # save checkpoint every N episodes
RESUME_TRAINING   = False     # continue from saved checkpoint
MODEL_PATH        = "./models"
LOG_FILE          = "./logs/training_log.csv"

os.makedirs("./models",      exist_ok=True)
os.makedirs("./models/best", exist_ok=True)
os.makedirs("./logs",        exist_ok=True)


def main():
    rclpy.init()
    env    = TD3Env()
    agent  = TD3()
    buffer = ReplayBuffer(STATE_DIM, ACTION_DIM)

    if RESUME_TRAINING and os.path.exists(f"{MODEL_PATH}/TD3_actor.pth"):
        agent.load(MODEL_PATH)
        print("▶  Resumed from existing checkpoint")

    # Log header
    with open(LOG_FILE, "w") as f:
        f.write("episode,steps,reward,result,goal_x,goal_y\n")
    # Log header — only write if starting fresh
    if not RESUME_TRAINING or not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w") as f:
            f.write("episode,steps,reward,result,goal_x,goal_y\n")

    best_reward  = -float("inf")
    total_steps  = 0
    goal_successes = 0

    print("\n" + "="*60)
    print("  TD3 Training Started")
    print(f"  Episodes: {MAX_EPISODES}   Batch: {BATCH_SIZE}")
    print(f"  Arena: 2m×2m   Goal: randomizes every episode")
    print("="*60 + "\n")

    for ep in range(1, MAX_EPISODES + 1):
        # reset() picks a NEW random goal automatically
        obs      = env.reset()
        gx, gy   = env._goal_x, env._goal_y
        ep_reward = 0.0
        ep_steps  = 0
        result    = "timeout"

        for _ in range(500):
            total_steps += 1
            ep_steps    += 1

            # ── Select action ──────────────────────────────────────────
            if total_steps < REPLAY_START:
                action = np.random.uniform(-1, 1, size=ACTION_DIM)
            else:
                action = agent.select_action(obs)
                noise  = np.random.normal(0, EXPLORATION_NOISE, ACTION_DIM)
                action = np.clip(action + noise, -1.0, 1.0)

            # ── Step environment ───────────────────────────────────────
            next_obs, reward, done, info = env.step(action)
            ep_reward += reward
            result     = info.get("result", "timeout")

            buffer.add(obs, action, next_obs, reward, float(done))
            obs = next_obs

            # ── Train ─────────────────────────────────────────────────
            if len(buffer) >= REPLAY_START:
                agent.train(buffer, BATCH_SIZE)

            if done:
                break

        # ── Stats ──────────────────────────────────────────────────────
        if result == "goal":
            goal_successes += 1

        success_rate = goal_successes / ep * 100

        status = "✅" if result == "goal" else ("💥" if result == "collision" else "⏱")
        print(
            f"Ep {ep:4d}/{MAX_EPISODES}  {status}  "
            f"goal=({gx:+.2f},{gy:+.2f})  "
            f"steps={ep_steps:3d}  "
            f"reward={ep_reward:7.1f}  "
            f"success={success_rate:4.1f}%  "
            f"buf={len(buffer):6d}"
        )

        with open(LOG_FILE, "a") as f:
            f.write(f"{ep},{ep_steps},{ep_reward:.2f},{result},{gx:.3f},{gy:.3f}\n")

        # Log to TensorBoard
        agent.log_episode(ep, ep_reward, ep_steps, result, success_rate)

        # ── Save ───────────────────────────────────────────────────────
        if ep % SAVE_EVERY == 0:
            agent.save(MODEL_PATH)

        if ep_reward > best_reward:
            best_reward = ep_reward
            agent.save(f"{MODEL_PATH}/best")
            print(f"  ⭐ New best: {best_reward:.1f}")

    # Final save
    agent.save(MODEL_PATH)
    agent.close()
    print(f"\n✅ Training done! Copy models/best/TD3_actor.pth to your Pi 5.")
    env._pub_cmd(0.0, 0.0)
    env.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
