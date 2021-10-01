from typing import OrderedDict
from gym import spaces
from gym.spaces import multi_discrete
import numpy as np
import random

DEFAULT_MOUSE_SPEED = 4.0
DEFAULT_MOUSE_MARGIN = 4.0
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
        self,
        action_space,
        always_attack=False,
        mouse_speed=DEFAULT_MOUSE_SPEED,
    ):
        self.mouse_speed = mouse_speed
        self.old_action_space = action_space
        self.always_attack = always_attack
        self.special_actions_dict = {
            "craft": ["crafting_table", "planks", "stick", "torch"],
            "equip": [
                "air",
                "iron_axe",
                "iron_pickaxe",
                "stone_axe",
                "stone_pickaxe",
                "wooden_axe",
                "wooden_pickaxe",
            ],
            "nearbyCraft": [
                "furnace",
                "iron_axe",
                "iron_pickaxe",
                "stone_axe",
                "stone_pickaxe",
                "wooden_axe",
                "wooden_pickaxe",
            ],
            "nearbySmelt": ["coal", "iron_ingot"],
            "place": [
                "cobblestone",
                "crafting_table",
                "dirt",
                "furnace",
                "stone",
                "torch",
            ],
        }
        self.special_actions_list = [
            (key, item)
            for key, lst in self.special_actions_dict.items()
            for item in lst
        ]
        self.original_keys = list(self.old_action_space.spaces.keys())

        self.action_space = spaces.Discrete(72 + len(self.special_actions_list))

    def dict_to_discrete(self, action_dict):
        discrete_action = 0

        def special_action_index(key, item):
            return self.special_actions_list.index((key, item))

        discrete_action += 1 * action_dict["attack"]
        discrete_action += 2 * action_dict["forward"]
        discrete_action += 4 * action_dict["jump"]
        discrete_action += mouse_action_to_discrete(action_dict["camera"][1]) * 8
        discrete_action += mouse_action_to_discrete(action_dict["camera"][0]) * 24

        offset = 2 * 2 * 2 * 3 * 3
        for key, item in action_dict.items():
            if key not in self.special_actions_dict.keys():
                continue
            if item == "none":
                continue
            discrete_action = offset + special_action_index(key, item)
        return discrete_action

    def discrete_to_dict(self, action_index):
        action_index = int(action_index)
        action_dict = OrderedDict(
            [
                (key, 0 if key not in self.special_actions_dict.keys() else "none")
                for key in self.original_keys
            ]
        )
        action_dict["camera"] = np.zeros((2,), dtype=np.float32)

        if action_index < 72:
            action_dict["attack"] = ACTIONS_DICT["attack"][action_index % 2]
            action_index //= 2
            action_dict["forward"] = ACTIONS_DICT["forward"][action_index % 2]
            action_index //= 2
            action_dict["jump"] = ACTIONS_DICT["jump"][action_index % 2]
            action_index //= 2
            action_dict["camera"][1] = ACTIONS_DICT["camera_x"][action_index % 3]
            action_index //= 3
            action_dict["camera"][0] = ACTIONS_DICT["camera_y"][action_index % 3]
            action_index //= 3
        else:
            key, item = self.special_actions_list[action_index - 72]
            action_dict[key] = item

        if self.always_attack:
            action_dict["attack"] = 1

        return action_dict
