import numpy as np
import gym
import torch
from torch import nn
import os
from behavior_cloning.net.model import ResNetImpala
from behavior_cloning.wrappers.action_wrappers import ObtainDiamondActions
from behavior_cloning.wrappers.observation_wrappers import ObtainDiamondObservation


MAX_TEST_EPISODE_LEN = 18000  # 18k is the default for MineRLObtainDiamond.
TREECHOP_STEPS = 2000  # number of steps to run BC lumberjack for in evaluations.

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
    """
    To compete in the competition, you are required to implement the two
    functions in this class:
        - load_agent: a function that loads e.g. network models
        - run_agent_on_episode: a function that plays one game of MineRL

    By default this agent behaves like a random agent: pick random action on
    each step.

    NOTE:
        This class enables the evaluator to run your agent in parallel in Threads,
        which means anything loaded in load_agent will be shared among parallel
        agents. Take care when tracking e.g. hidden state (this should go to run_agent_on_episode).
    """

    def load_agent(self):
        """
        This method is called at the beginning of the evaluation.
        You should load your model and do any preprocessing here.
        THIS METHOD IS ONLY CALLED ONCE AT THE BEGINNING OF THE EVALUATION.
        DO NOT LOAD YOUR MODEL ANYWHERE ELSE.
        """
        # Load up the behavioural cloning model.
        # NOTE: Unlike in the baseline code, we store and load the state dict of the network,
        #       rather than the pickled network itself. This is to avoid importing issues
        #       that rise when code structure changes (which happens if you train the model with
        #       the baseline code and try to import it here).
        # The good thing is that we know exactly what the input shape and output shapes should be
        action_nvec = [2, 3, 3, 5, 8, 2, 2, 8, 3, 7]

        image_shape = (3, 64, 64)
        direct_shape = (172,)

        self.model = ResNetImpala(image_shape, action_nvec, True, direct_shape).cuda()
        self.model.load_state_dict(
            torch.load(
                os.path.dirname(__file__) + "/train/value_policy_v0.pth_steps_2060000"
            )
        )
        self.model.eval().cuda()

    def run_agent_on_episode(self, single_episode_env: Episode):
        """This method runs your agent on a SINGLE episode.

        You should just implement the standard environment interaction loop here:
            obs  = env.reset()
            while not done:
                env.step(self.agent.act(obs))
                ...

        NOTE:
            This method will be called in PARALLEL during evaluation.
            So, only store state in LOCAL variables.
            For example, if using an LSTM, don't store the hidden state in the class
            but as a local variable to the method.

        Args:
            env (gym.Env): The env your agent should interact with.
        """
        env = single_episode_env

        obs_processor = ObtainDiamondObservation(env.observation_space)
        act_processor = ObtainDiamondActions(env.action_space)

        obs = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            obs = preprocess(obs)
            obs = obs_processor.dict_to_tuple(obs)
            value, logits = self.model(
                torch.FloatTensor(obs[0]).cuda(), torch.FloatTensor(obs[1]).cuda()
            )
            # Turn logits into probabilities
            probabilities = []
            for logit in logits:
                probabilities.append(torch.softmax(logit, dim=1))
            # Into numpy
            probabilities = [
                probability.detach().cpu().squeeze().numpy()
                for probability in probabilities
            ]
            agent_action = act_processor.probabilities_to_multidiscrete(probabilities)
            agent_action = act_processor.multidiscrete_to_dict(agent_action)

            obs, reward, done, info = env.step(agent_action)
            total_reward += reward
            steps += 1


def preprocess(obs: dict):
    ret = {}

    def rec_reshape(dic, long_key):
        if isinstance(dic, dict):
            for key, val in dic.items():
                if key == "inventory":
                    ret[key] = preprocess(dic[key])
                else:
                    rec_reshape(
                        dic[key], long_key + ("" if long_key is "" else ".") + key
                    )
        else:
            if isinstance(dic, np.ndarray):
                ret[long_key] = dic[np.newaxis, ...]
            elif isinstance(dic, np.int64):
                ret[long_key] = np.full((1, 1), dic)
            else:
                ret[long_key] = [dic]

    rec_reshape(obs, "")
    return ret
