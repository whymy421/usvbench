import torch
import torch.nn as nn


class LearnedRewardNet(nn.Module):
    def __init__(self, state_dim=7, action_dim=2, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


class LearnedReward:
    def __init__(self, num_envs, state_dim=7, action_dim=2, device='cuda'):
        self.device = device
        self.num_envs = num_envs
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reward_net = LearnedRewardNet(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.reward_net.parameters(), lr=3e-4)

        #  env episode
        self.ep_obs = [[] for _ in range(num_envs)]
        self.ep_act = [[] for _ in range(num_envs)]

        #  trajectory buffer
        self.success_buffer = []  # list of (obs_tensor, act_tensor)
        self.failure_buffer = []
        self.buffer_max = 200  #  200 trajectory

        #
        self.episode_count = 0
        self.episode_successes = []
        self.update_every = 100  #  100 episode
        self.train_epochs = 5

        #  reward
        self.base_reward_weight = 0.7  #  reward
        self.min_base_weight = 0.2

    def compute_reward(self, obs, actions, distance, goal_radius, max_spawn_distance):
        #
        for i in range(self.num_envs):
            self.ep_obs[i].append(obs[i].detach().clone())
            self.ep_act[i].append(actions[i].detach().clone())

        #  reward
        with torch.no_grad():
            learned_r = self.reward_net(obs, actions)

        #  reward
        reached = (distance < goal_radius).float()
        base_r = -distance / max_spawn_distance * 0.3 + reached * 10.0 + 0.05

        #
        w = self.base_reward_weight
        reward = w * base_r + (1 - w) * learned_r

        return reward

    def on_episode_end(self, env_ids, reached):
        for i, env_id in enumerate(env_ids):
            env_id_int = int(env_id) if torch.is_tensor(env_id) else int(env_id)
            success = bool(reached[i].item() if torch.is_tensor(reached[i]) else reached[i])

            #  trajectory buffer
            if len(self.ep_obs[env_id_int]) > 5:
                obs_t = torch.stack(self.ep_obs[env_id_int])
                act_t = torch.stack(self.ep_act[env_id_int])
                if success:
                    self.success_buffer.append((obs_t, act_t))
                    if len(self.success_buffer) > self.buffer_max:
                        self.success_buffer.pop(0)
                else:
                    self.failure_buffer.append((obs_t, act_t))
                    if len(self.failure_buffer) > self.buffer_max:
                        self.failure_buffer.pop(0)

            #  episode
            self.ep_obs[env_id_int] = []
            self.ep_act[env_id_int] = []

            self.episode_successes.append(float(success))
            self.episode_count += 1

        #  reward
        if self.episode_count % self.update_every == 0:
            self._update_reward_net()

    def _update_reward_net(self):
        n_success = len(self.success_buffer)
        n_failure = len(self.failure_buffer)

        recent = self.episode_successes[-self.update_every:]
        success_rate = sum(recent) / len(recent) if recent else 0

        print(f'Reward update | Ep: {self.episode_count} | '
              f'Success rate: {success_rate:.2f} | '
              f'Buffer: {n_success} success, {n_failure} failure | '
              f'Base weight: {self.base_reward_weight:.2f}')

        #
        if n_success < 5 or n_failure < 5:
            # Not enough positive and negative examples; keep the base reward dominant.
            self.base_reward_weight = min(0.9, self.base_reward_weight + 0.02)
            return

        with torch.enable_grad():
            self.reward_net.train()

            for epoch in range(self.train_epochs):
                #  buffer
                s_idx = torch.randint(0, n_success, (16,))
                #  buffer
                f_idx = torch.randint(0, n_failure, (16,))

                loss_total = torch.tensor(0.0, device=self.device)

                for idx in s_idx:
                    obs, act = self.success_buffer[idx]
                    #
                    n_steps = min(32, obs.shape[0])
                    step_idx = torch.randint(0, obs.shape[0], (n_steps,))
                    pred = self.reward_net(obs[step_idx].to(self.device),
                                          act[step_idx].to(self.device))
                    # Encourage a positive prediction on successful trajectories (+1).
                    loss_total = loss_total + ((pred - 1.0) ** 2).mean()

                for idx in f_idx:
                    obs, act = self.failure_buffer[idx]
                    n_steps = min(32, obs.shape[0])
                    step_idx = torch.randint(0, obs.shape[0], (n_steps,))
                    pred = self.reward_net(obs[step_idx].to(self.device),
                                          act[step_idx].to(self.device))
                    # Encourage a negative prediction on failed trajectories (-1).
                    loss_total = loss_total + ((pred + 1.0) ** 2).mean()

                loss_total = loss_total / 32

                self.optimizer.zero_grad()
                loss_total.backward()
                self.optimizer.step()

            self.reward_net.eval()

        #  reward
        if success_rate > 0.3:
            self.base_reward_weight = max(self.min_base_weight,
                                          self.base_reward_weight - 0.03)
        elif success_rate < 0.1:
            self.base_reward_weight = min(0.9, self.base_reward_weight + 0.02)
