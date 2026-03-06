import torch
import torch.nn.functional as F
import numpy as np, copy, os
from actor_critic import Actor, Critic, STATE_DIM, ACTION_DIM, MAX_ACTION
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TD3:
    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM,
                 max_action=MAX_ACTION, discount=0.99, tau=0.005,
                 policy_noise=0.2, noise_clip=0.5, policy_freq=2, lr=3e-4,
                 log_dir="./runs"):

        self.actor         = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target  = copy.deepcopy(self.actor)
        self.actor_opt     = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic        = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_opt    = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.max_action   = max_action
        self.discount     = discount
        self.tau          = tau
        self.policy_noise = policy_noise * max_action
        self.noise_clip   = noise_clip   * max_action
        self.policy_freq  = policy_freq
        self.total_it     = 0

        # TensorBoard writer
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)
        print(f"[TD3] TensorBoard logs → {log_dir}")
        print(f"[TD3] Run:  tensorboard --logdir={log_dir} --host=0.0.0.0")
        print(f"[TD3] Open: http://localhost:6006")

        self._last_critic_loss = 0.0
        self._last_actor_loss  = 0.0
        self._last_avg_q       = 0.0

    def select_action(self, state):
        s = torch.FloatTensor(state).unsqueeze(0).to(device)
        return self.actor(s).cpu().detach().numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1
        state, action, next_state, reward, done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(
                -self.max_action, self.max_action)
            tQ1, tQ2   = self.critic_target(next_state, next_action)
            target_Q   = reward + (1 - done) * self.discount * torch.min(tQ1, tQ2)

        cQ1, cQ2    = self.critic(state, action)
        critic_loss = F.mse_loss(cQ1, target_Q) + F.mse_loss(cQ2, target_Q)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        self._last_critic_loss = critic_loss.item()
        self._last_avg_q       = cQ1.mean().item()

        self.writer.add_scalar("Loss/Critic",  self._last_critic_loss, self.total_it)
        self.writer.add_scalar("Q/Average_Q1", self._last_avg_q,       self.total_it)
        self.writer.add_scalar("Q/Target_Q",   target_Q.mean().item(), self.total_it)

        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            self._last_actor_loss = actor_loss.item()
            self.writer.add_scalar("Loss/Actor", self._last_actor_loss, self.total_it)

            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
            for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    def log_episode(self, episode, reward, steps, result, success_rate):
        self.writer.add_scalar("Episode/Reward",       reward,       episode)
        self.writer.add_scalar("Episode/Steps",        steps,        episode)
        self.writer.add_scalar("Episode/Success_Rate", success_rate, episode)
        result_val = 1 if result == "goal" else (-1 if result == "collision" else 0)
        self.writer.add_scalar("Episode/Result", result_val, episode)

    def save(self, path="./models"):
        os.makedirs(path, exist_ok=True)
        torch.save(self.actor.state_dict(),  f"{path}/TD3_actor.pth")
        torch.save(self.critic.state_dict(), f"{path}/TD3_critic.pth")
        print(f"[TD3] Saved → {path}/")

    def load(self, path="./models"):
        self.actor.load_state_dict( torch.load(f"{path}/TD3_actor.pth",  map_location=device))
        self.critic.load_state_dict(torch.load(f"{path}/TD3_critic.pth", map_location=device))
        self.actor_target  = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        print(f"[TD3] Loaded ← {path}/")

    def close(self):
        self.writer.close()
