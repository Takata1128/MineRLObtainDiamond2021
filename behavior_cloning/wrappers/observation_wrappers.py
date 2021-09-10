from collections import OrderedDict
import numpy as np
import random
from copy import deepcopy

import gym
from gym import spaces
from gym import Wrapper

MAX_INT_ONEHOT = 8


def one_hot_encode(value, num_possibilities):
    one_hot = np.zeros([value.shape[0], num_possibilities])
    one_hot[:, value] = 1
    return one_hot


def observation_data_augmentation(obs, do_flipping=False):
    pov_obs = obs[0]
    # gaussian noise
    pov_obs += np.random.normal(scale=0.005, size=pov_obs.shape)

    # contrast
    pov_obs *= random.uniform(0.98, 1.02)

    # brightness
    pov_obs += np.random.uniform(-0.02, 0.02, size=(3, 1, 1))

    if do_flipping:
        if random.random() < 0.5:
            pov_obs = np.fliplr(pov_obs)

    np.clip(pov_obs, 0.0, 1.0, out=obs[0])


class ObtainDiamondObservation:
    def __init__(
        self,
        observation_space,
        max_inventry_count=MAX_INT_ONEHOT,
        augmentation=False,
        augmentation_flip=False,
        gamma_correction=1.0,
        just_pov=False,
        numeric_df=False,
    ):
        self.max_inventry_count = max_inventry_count
        self.inventry_eyes = np.eye(max_inventry_count + 1)
        self.augmentation = augmentation
        self.augmentation_flip = augmentation_flip
        self.inverse_gamma = 1 / gamma_correction
        self.just_pov = just_pov
        self.numeric_df = numeric_df

        old_space = observation_space

        self.direct_features_len = 1

        self.hand_items_map = old_space.spaces["equipped_items"]["mainhand"][
            "type"
        ].value_map
        self.num_hand_items = old_space.spaces["equipped_items"]["mainhand"]["type"].n
        self.direct_features_len += self.num_hand_items

        self.inventry_keys = []
        for key, space in old_space["inventory"].spaces.items():
            self.inventry_keys.append(key)
            if not self.numeric_df:
                self.direct_features_len += max_inventry_count + 1
            else:
                self.direct_features_len += 1

        # (H,W,C) -> (C,H,W)
        pov_shape = (
            old_space["pov"].shape[2],
            old_space["pov"].shape[0],
            old_space["pov"].shape[1],
        )
        if not self.just_pov:
            self.observation_space = spaces.Tuple(
                spaces=(
                    spaces.Box(low=0, high=1, shape=pov_shape, dtype=np.float32),
                    spaces.Box(
                        low=0,
                        high=1,
                        shape=(self.direct_features_len,),
                        dtype=np.float32,
                    ),
                )
            )
        else:
            self.observation_space = spaces.Box(
                low=0, high=1, shape=pov_shape, dtype=np.float32
            )

    def flip_left_right(self, obs):
        obs[0][:] = np.fliplr(obs[0])

    def dict_to_tuple(self, dict_obs):
        if len(dict_obs["pov"].shape) == 3:
            for key, val in dict_obs.items():
                val.reshape(1, -1)

        # normalize
        pov_obs = dict_obs["pov"].astype(np.float32) / 255.0
        pov_obs = pov_obs.transpose(0, 3, 1, 2)
        # apply gamma correction
        if self.inverse_gamma != 1.0:
            pov_obs = pov_obs ** self.inverse_gamma

        direct_features = []
        # Mainhand
        damage = dict_obs["equipped_items.mainhand.damage"]
        max_damage = np.maximum(1, dict_obs["equipped_items.mainhand.maxDamage"])
        damage_ratio = damage / max_damage
        direct_features.append(
            damage_ratio.reshape(damage_ratio.shape[0], 1),
        )
        mainhand_strs = dict_obs["equipped_items.mainhand.type"]
        mainhand_item = np.zeros_like(mainhand_strs, dtype=np.int32)
        for i, hand_item_str in enumerate(mainhand_strs):
            mainhand_item[i] = self.hand_items_map[str(hand_item_str)]

        direct_features.append(one_hot_encode(mainhand_item, self.num_hand_items))

        # Inventry
        for i, (key, count) in enumerate(dict_obs["inventory"].items()):
            count = np.minimum(count, self.max_inventry_count)
            if not self.numeric_df:
                direct_features.append(self.inventry_eyes[count])
            else:
                direct_features.append([count / self.max_inventry_count])

        direct_features = np.concatenate(direct_features, axis=1).astype(np.float32)

        obs = (pov_obs, direct_features)

        if self.augmentation:
            observation_data_augmentation(obs, self.augmentation)

        return obs


class FrameSkipWrapper(Wrapper):
    def __init__(self, env, frame_skip=4):
        super().__init__(env)
        self.env = env
        self.frame_skip = frame_skip

        if not isinstance(self.env.action_space, spaces.Dict):
            raise RuntimeError("FrameSkipWrapper needs dict action space")

    def step(self, action):
        action = deepcopy(action)

        if "camera" in action.keys():
            action["camera"] = action["camera"] // self.frame_skip

        reward_sum = 0
        for i in range(self.frame_skip):
            obs, reward, terminal, info = self.env.step(action)
            reward_sum += reward
            if terminal:
                break
            if i == 0:
                action["craft"] = "none"
                action["nearbyCraft"] = "none"
                action["nearbySmelt"] = "none"
                action["equip"] = "none"
                action["place"] = "none"

        return obs, reward_sum, terminal, info
