from gym.wrappers import Monitor
import torch as th
import numpy as np
import os
import gym
import minerl
import torch
from behavior_cloning.net.model import ResNetImpala
from behavior_cloning.wrappers.observation_wrappers import ObtainDiamondObservation
from behavior_cloning.wrappers.action_wrappers import ObtainDiamondActions


def enjoy():
    # Load up the trained network

    env = Monitor(gym.make("MineRLObtainDiamond-v0"), "/root/mnt/videos", force=True)

    obs_processor = ObtainDiamondObservation(env.observation_space)
    act_processor = ObtainDiamondActions(env.action_space, always_attack=True)
    action_n = act_processor.action_space.n

    image_shape = obs_processor.observation_space[0].shape
    direct_shape = obs_processor.observation_space[1].shape

    model = ResNetImpala(image_shape, action_n, True, direct_shape).cuda()
    model.load_state_dict(th.load("/root/mnt/minerl-intro/train/model.pt"))
    model.eval().cuda()
    # Play n games with the model
    for game_i in range(5):
        obs = env.reset()
        done = False
        reward_sum = 0
        while not done:
            obs = obs_processor.dict_to_tuple(obs)
            distrib, _ = model(
                (
                    torch.unsqueeze(torch.FloatTensor(obs[0]), 0).cuda(),
                    torch.unsqueeze(torch.FloatTensor(obs[1]), 0).cuda(),
                )
            )
            agent_action = act_processor.discrete_to_dict(distrib.sample())
            obs, reward, done, info = env.step(agent_action)
            reward_sum += reward
        print("Game {}, total reward {}".format(game_i, reward_sum))
    env.close()


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


if __name__ == "__main__":
    enjoy()
