ANGLE = 3
ACTIONS_LIST = [
    "attack",
    "forward",
    "jump",
    "camera",
    "camera",
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
    "camera1": [[-ANGLE, 0], [0, 0], [ANGLE, 0]],
    "camera2": [[0, -ANGLE], [0, 0], [0, ANGLE]],
    "craft": ["crafting_table", "none", "planks", "stick", "torch"],
    "equip": [
        "air",
        "iron_axe",
        "iron_pickaxe",
        "none",
        "stone_axe",
        "stone_pickaxe",
        "wooden_axe",
        "wooden_pickaxe",
    ],
    "nearbyCraft": [
        "furnace",
        "iron_axe",
        "iron_pickaxe",
        "none",
        "stone_axe",
        "stone_pickaxe",
        "wooden_axe",
        "wooden_pickaxe",
    ],
    "nearbySmelt": ["coal", "iron_ingot", "none"],
    "place": [
        "cobblestone",
        "crafting_table",
        "dirt",
        "furnace",
        "none",
        "stone",
        "torch",
    ],
}
COMPONENT_NUMS = [(16, 2), (32, 2), (32, 2)]
ACTION_NVEC = [2, 2, 2, 3, 3, 5, 8, 8, 3, 7]
RESIDUAL_CHANNEL_LIST = [
    [[16, 16], [16, 16]],
    [[32, 32], [32, 32]],
    [[32, 32], [32, 32]],
]

DN_FILTERS = 32
RESIDUAL_NUM = 3
# TODO COMPONENT_LIST と RESIDUAL_CHANNEL_LIST　依存関係
