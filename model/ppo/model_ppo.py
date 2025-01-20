import PIL
import torch
import time
import random
from PIL import Image
from pathlib import Path
from torch import nn
from loguru import logger
from torch.utils.data import Dataset
from reward import AutoGameEnvironment
from model.model_resnet import ResNetModelConfig, AutoGameForImageClassification
from utils.datasets import ImagePreprocess
from utils.tools import get_available_device


class RolloutBuffer(Dataset):
    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []

    def __getitem__(self, idx):
        return (torch.tensor(self.observations[idx], torch.float32),
                torch.tensor(self.actions[idx], torch.long),
                torch.tensor(self.rewards[idx], torch.float32),
                torch.tensor(self.log_probs[idx], torch.float32),
                torch.tensor(self.values[idx], torch.float32),
                torch.tensor(self.dones[idx], torch.uint8),
                )

    def __len__(self):
        return len(self.observations)

    @staticmethod
    def collate_fn():
        pass

    def reset(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []

    def add(self, obs, action, reward, log_prob, value, done):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)


def load_main_model(model_name_or_path: str = None, requires_grad: bool = False):
    resnet_config = ResNetModelConfig()
    model = AutoGameForImageClassification(config=resnet_config)

    if Path(model_name_or_path).exists():
        logger.info(f"Loading main model from: {model_name_or_path}")
        model.load_state_dict(torch.load(model_name_or_path, map_location=get_available_device()))

    # set weights trainable
    model.requires_grad_(requires_grad)
    return model


class Agent(nn.Module):
    rollout_buffer: RolloutBuffer

    def __init__(self,
                 env: AutoGameEnvironment,
                 learning_rate: float = 3e-4,
                 scratch: bool = False,
                 batch_size: int = 32,
                 deterministic_ratio: float = 0.8,
                 max_grad_norm: float = 0.5,
                 eps_clip: float = 0.2,
                 gamma: float = 0.98,
                 lamda: float = 0.95,
                 k_epochs: int = 3,
                 vf_coef: float = 0.01,
                 ent_coef: float = 0.5,
                 verbose: bool = True,
                 **kwargs,
                 ):
        super(Agent, self).__init__(**kwargs)
        self.env = env
        self.rollout_buffer = RolloutBuffer()
        self.learning_rate = learning_rate
        self.scratch = scratch
        self.batch_size = batch_size
        self.deterministic_ratio = deterministic_ratio
        self.max_grad_norm = max_grad_norm
        self.eps_clip = eps_clip
        self.gamma = gamma
        self.lamda = lamda
        self.k_epochs = k_epochs
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.verbose = verbose
        self.device = get_available_device()
        logger.info(f"Using device {self.device}")

        # model
        if self.scratch:
            config = ResNetModelConfig()
            self.actor_critic_model = AutoGameForImageClassification(config=config)
        else:
            self.actor_critic_model = load_main_model(model_name_or_path=".../weights/main_best.pth", requires_grad=True)
        self.transform = ImagePreprocess()
        self.actor_critic_model = self.actor_critic_model.to(self.device)

        # optimizer
        self.optimizer = torch.optim.Adam(self.actor_critic_model.parameters(), lr=self.learning_rate)
        self.MseLoss = nn.MSELoss()

    def collect_rollouts(self, n_rollout_steps: int):
        _last_obs = self.env.game_control.get_screenshot()

        n_steps = 0
        self.rollout_buffer.reset()
        while n_steps < n_rollout_steps:
            _last_obs = PIL.Image.fromarray(_last_obs[..., ::-1])
            _last_obs = self.transform(_last_obs)
            _last_obs = _last_obs.clone().detach().unsqueeze(0)

            with torch.no_grad():
                # deterministic_ratio比例选择训练好的模型给出的action，剩下的随机，避免在高地不出去
                rand_num = random.random()
                if rand_num < self.deterministic_ratio:
                    action, value, log_prob = self.get_actions(_last_obs, deterministic=True)
                else:
                    action, value, log_prob = self.get_actions(_last_obs, deterministic=False)
                action = action.item()

            new_obs, reward, done = self.env.step(action)
            if done:
                logger.info("game finished.")
                _last_obs = self.env.reset()
                continue

            self.rollout_buffer.add(
                obs=_last_obs,
                action=action,
                reward=reward,
                log_prob=log_prob,
                value=value,
                done=done,
            )
            _last_obs = new_obs
            n_steps += 1

    def get_actions(self, obs, deterministic: bool = False):
        """
        如果deterministic is True，那么根据预训练好的模型选择actions，否则使用sample方式采样选择actions
        """
        latent_pi, latent_vf = self.actor_critic_model(obs)
        latent_pi = nn.Softmax(dim=-1)(latent_pi)

        values = latent_vf

        # get distribution
        distribution = torch.distributions.Categorical(latent_pi)
        if deterministic:
            actions = torch.argmax(latent_pi, dim=-1)
        else:
            actions = distribution.sample()
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return actions, values, log_prob

    def evaluate_actions(self, obs, actions):
        """
        latent_pi代表policy，latent_vf代表value function
        """
        latent_pi, latent_vf = self.actor_critic_model(obs)
        latent_pi = nn.Softmax(dim=-1)(latent_pi)

        # value function
        values = latent_vf

        # get distribution
        distribution = torch.distributions.Categorical(latent_pi)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def predict(self, obs):
        latent_pi, _ = self.actor_critic_model(obs)
        action = torch.argmax(latent_pi, dim=-1)
        return action.item()

    @staticmethod
    def compute_returns_and_advantage(rewards,
                                      values,
                                      dones,
                                      gamma: int = 0.99,
                                      gae_lambda: int = 0.95):
        buffer_size = len(rewards)
        gae = 0
        returns, advantages = [], []
        for step in reversed(range(buffer_size)):
            if step == buffer_size - 1:
                next_non_terminal = 0
                next_values = 0
            else:
                next_non_terminal = 1.0 - dones[step + 1]
                next_values = values[step + 1]
            delta = rewards[step] + gamma * next_values * next_non_terminal - values[step]

            # compute discount reward
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            advantages.insert(0, gae)
        advantages = torch.tensor(advantages, dtype=torch.float32).unsqueeze(0).reshape((buffer_size, -1))
        values = torch.tensor(values, dtype=torch.float32).unsqueeze(0).reshape((buffer_size, -1))
        returns = advantages + values
        return returns, advantages

    def ppo_train(self, normalize_advantage: bool = True):
        returns, advantages = self.compute_returns_and_advantage(rewards=self.rollout_buffer.rewards,
                                                                 values=self.rollout_buffer.values,
                                                                 dones=self.rollout_buffer.dones)

        if normalize_advantage:
            # advantages = torch.tensor(advantages, dtype=torch.float32)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        self.rollout_buffer.observations = torch.cat(self.rollout_buffer.observations, dim=0).to(self.device)
        self.rollout_buffer.actions = torch.tensor(self.rollout_buffer.actions, dtype=torch.long).to(self.device)
        self.rollout_buffer.rewards = torch.tensor(self.rollout_buffer.rewards, dtype=torch.float32).to(self.device)
        self.rollout_buffer.log_probs = torch.tensor(self.rollout_buffer.log_probs, dtype=torch.float32).to(self.device)
        self.rollout_buffer.values = torch.tensor(self.rollout_buffer.values, dtype=torch.float32).to(self.device)
        self.rollout_buffer.dones = torch.tensor(self.rollout_buffer.dones, dtype=torch.int8).to(self.device)

        advantages = advantages.clone().detach().to(self.device)
        # advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = returns.clone().detach().to(self.device)

        self.actor_critic_model.train()
        old_log_probs = self.rollout_buffer.log_probs
        loss = 0
        for _ in range(self.k_epochs):
            values, new_log_probs, dist_entropy = self.evaluate_actions(self.rollout_buffer.observations, self.rollout_buffer.actions)

            # policy loss
            ratio = torch.exp(new_log_probs - old_log_probs)
            policy_loss_1 = ratio * advantages
            policy_loss_2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            policy_loss = -torch.min(policy_loss_1, policy_loss_2)

            # values loss
            value_loss = self.MseLoss(returns, values)

            # entropy loss
            if dist_entropy is None:
                # Approximate entropy when no analytical form
                entropy_loss = -torch.mean(-new_log_probs)
            else:
                entropy_loss = -torch.mean(dist_entropy)

            loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
            loss = loss.mean()

            # Optimization step
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.actor_critic_model.parameters(), self.max_grad_norm)
            self.optimizer.step()

        # logging
        if self.verbose:
            training_values = self.rollout_buffer.values.sum()
            logger.info(f"loss: {loss.item()}, score: {training_values}")

        # clear buffer
        self.rollout_buffer.reset()

    def learn(self, total_timesteps: int = None):
        # start game from main game page
        self.env.reset()

        num_timesteps = 0
        while num_timesteps < total_timesteps:
            self.collect_rollouts(n_rollout_steps=self.batch_size)
            self.ppo_train()
            num_timesteps += 1

            self.save("../weights/main_best.pth")

    def save(self, checkpoint_path):
        torch.save(self.actor_critic_model.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.actor_critic_model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))


if __name__ == "__main__":
    print("-----------------training game----------------------")
    env = AutoGameEnvironment()

    model = Agent(env=env)
    model.learn(total_timesteps=2000)
