import numpy as np
import gym
import torch
from torch import nn
import os
from behavior_cloning.net.model import ResNetImpala
from behavior_cloning.wrappers.action_wrappers import ObtainDiamondActions
from behavior_cloning.wrappers.observation_wrappers import ObtainDiamondObservation


MAX_TEST_EPISODE_LEN = 18000  # 18k is the default for MineRLObtainDiamond.

# !!! Do not change this! This is part of the submission kit !!!
class EpisodeDone(Exception):
    pass


# !!! Do not change this! This is part of the submission kit !!!
class Episode(gym.Env):
    """A class for a single episode."""

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self._done = False

    def reset(self):
        if not self._done:
            return self.env.reset()

    def step(self, action):
        s, r, d, i = self.env.step(action)
        if d:
            self._done = True
            raise EpisodeDone()
        else:
            return s, r, d, i


class MineRLAgent:
    def load_agent(self):
        action_n = 98
        image_shape = (3, 64, 64)
        direct_shape = (172,)

        self.model = ResNetImpala(image_shape, action_n, True, direct_shape).cuda()
        self.model.load_state_dict(
            torch.load(os.path.dirname(__file__) + "/train/model.pt")
        )
        self.model.eval().cuda()

    def run_agent_on_episode(self, single_episode_env: Episode):
        env = single_episode_env

        obs_processor = ObtainDiamondObservation(env.observation_space)
        act_processor = ObtainDiamondActions(env.action_space, always_attack=True)

        obs = env.reset()
        done = False
        total_reward = 0
        steps = 0

        # run
        while not done:
            obs = obs_processor.dict_to_tuple(obs)
            distrib, _ = self.model(
                (
                    torch.unsqueeze(torch.FloatTensor(obs[0]), 0).cuda(),
                    torch.unsqueeze(torch.FloatTensor(obs[1]), 0).cuda(),
                )
            )
            agent_action = act_processor.discrete_to_dict(distrib.sample())
            obs, reward, done, info = env.step(agent_action)
            total_reward += reward
            steps += 1
