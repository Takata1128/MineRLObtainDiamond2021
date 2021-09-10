from typing import OrderedDict
from gym import spaces
from gym.spaces import multi_discrete
import numpy as np
import random

DEFAULT_MOUSE_SPEED = 4
DEFAULT_MOUSE_MARGIN = 4
FORCED_ACTIONS_IN_MINIMAL_ACTIONS = {
    "back": 0,
    "left": 0,
    "right": 0,
    "sneak": 0,
    "sprint": 0,
}

ACTIONS_LIST = [
    "attack",
    "forward",
    "jump",
    "camera_x",
    "camera_y",
    "craft",
    "equip",
    "nearbyCraft",
    "nearbySmelt",
    "place",
]
ACTIONS_DICT = {
    "attack": [0, 1],
    "forward": [0, 1],
    "jump": [0, 1],
    "camera_y": [0, -DEFAULT_MOUSE_SPEED, DEFAULT_MOUSE_SPEED],
    "camera_x": [0, -DEFAULT_MOUSE_SPEED, DEFAULT_MOUSE_SPEED],
    "craft": ["none", "crafting_table", "planks", "stick", "torch"],
    "equip": [
        "none",
        "air",
        "iron_axe",
        "iron_pickaxe",
        "stone_axe",
        "stone_pickaxe",
        "wooden_axe",
        "wooden_pickaxe",
    ],
    "nearbyCraft": [
        "none",
        "furnace",
        "iron_axe",
        "iron_pickaxe",
        "stone_axe",
        "stone_pickaxe",
        "wooden_axe",
        "wooden_pickaxe",
    ],
    "nearbySmelt": ["coal", "iron_ingot", "none"],
    "place": [
        "none",
        "cobblestone",
        "crafting_table",
        "dirt",
        "furnace",
        "stone",
        "torch",
    ],
}


def mouse_action_to_discrete(camera_action):
    if camera_action < -DEFAULT_MOUSE_MARGIN:
        return 1
    elif camera_action > DEFAULT_MOUSE_MARGIN:
        return 2
    else:
        return 0


def discrete_to_mouse_action(discrete_action, mouse_speed):
    if discrete_action == 1:
        return -mouse_speed
    elif discrete_action == 2:
        return mouse_speed
    else:
        return 0


class ObtainDiamondActions:
    """Turn action space of ObtainDiamond to something more managable by networks."""

    def __init__(
        self, action_space, mouse_speed=DEFAULT_MOUSE_SPEED, minimal_actions=True
    ):
        self.mouse_speed = mouse_speed
        self.old_action_space = action_space
        self.ignore_actions = {}

        if minimal_actions:
            self.ignore_actions.update(FORCED_ACTIONS_IN_MINIMAL_ACTIONS)
        self.ignore_action_keys = set(self.ignore_actions.keys())

        self.original_keys = list(self.old_action_space.spaces.keys())

        self.discrete_sizes = []
        self.discrete_names = []

        for key, space in self.old_action_space.spaces.items():
            if key in self.ignore_action_keys:
                continue
            if key == "camera":
                if "camera_x" not in self.ignore_action_keys:
                    self.discrete_sizes.append(3)
                    self.discrete_names.append("camera_x")
                if "camera_y" not in self.ignore_action_keys:
                    self.discrete_sizes.append(3)
                    self.discrete_names.append("camera_y")
            else:
                self.discrete_sizes.append(space.n)
                self.discrete_names.append(key)

        self.num_discrete = len(self.discrete_sizes)
        self.action_space = spaces.MultiDiscrete(self.discrete_sizes)

    def get_no_op(self):
        return [0] * self.num_discrete

    def flip_left_right(self, actions):
        for action in actions:
            # flip camera turning
            if "camera_x" in self.discrete_names:
                camera_x_idx = self.discrete_names.index("camera_x")
                camera_x_action = action[camera_x_idx]
                if camera_x_action == 1:
                    action[camera_x_idx] = 2
                elif camera_x_action == 2:
                    action[camera_x_idx] = 1

            # swap left and right
            if "left" in self.discrete_names:
                left_idx = self.discrete_names.index("left")
                right_idx = self.discrete_names.index("right")

                left_action = action[left_idx]
                action[left_idx] = action[right_idx]
                action[right_idx] = left_action

    # input dict
    # return [seqlen][action_type_index] = action_index
    def dict_to_multidiscrete(self, action_dict):
        seq_len = action_dict["camera"].shape[0]
        discrete_actions = np.zeros((seq_len, len(self.discrete_sizes)), dtype=np.int32)

        for i, key in enumerate(self.discrete_names):
            for j in range(seq_len):
                if key == "camera_x":
                    discrete_actions[j][i] = mouse_action_to_discrete(
                        action_dict["camera"][j][1]
                    )
                elif key == "camera_y":
                    discrete_actions[j][i] = mouse_action_to_discrete(
                        action_dict["camera"][j][0]
                    )
                else:
                    discrete_actions[j][i] = ACTIONS_DICT[key].index(
                        action_dict[key][j]
                    )
        return discrete_actions

    def multidiscrete_to_dict(self, action_multi):
        action_dict = OrderedDict([(key, None) for key in self.original_keys])
        action_dict["camera"] = np.zeros((2,), dtype=np.float32)

        for discrete_i in range(len(self.discrete_names)):
            discrete_name = self.discrete_names[discrete_i]
            discrete_value = action_multi[discrete_i]

            if discrete_name == "camera_x":
                action_dict["camera"][1] = discrete_to_mouse_action(
                    discrete_value, self.mouse_speed
                )
            elif discrete_name == "camera_y":
                action_dict["camera"][0] = discrete_to_mouse_action(
                    discrete_value, self.mouse_speed
                )
            else:
                action_dict[discrete_name] = ACTIONS_DICT[discrete_name][discrete_value]

        for ignored_name, ignored_value in self.ignore_actions.items():
            action_dict[ignored_name] = ignored_value

        return action_dict

    def probabilities_to_multidiscrete(self, predictions, greedy=False, epsilon=0.0):
        multi_discrete_action = []
        for i, discrete_size in enumerate(self.discrete_sizes):
            probabilities = predictions[i]
            assert len(probabilities) == discrete_size
            action = None
            if greedy:
                action = np.argmax(probabilities)
            elif random.random() < epsilon:
                action = random.choice(range(discrete_size))
            else:
                action = np.random.choice(range(discrete_size), p=probabilities)
            multi_discrete_action.append(action)
        return multi_discrete_action
