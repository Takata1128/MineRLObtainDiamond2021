import os
from behavior_cloning.training import replay_train, sequential_train

# All the evaluations will be evaluated on MineRLObtainDiamond-v0 environment
MINERL_GYM_ENV = os.getenv("MINERL_GYM_ENV", "MineRLObtainDiamond-v0")
# You need to ensure that your submission is trained in under MINERL_TRAINING_MAX_STEPS steps
MINERL_TRAINING_MAX_STEPS = int(os.getenv("MINERL_TRAINING_MAX_STEPS", 8000000))
# You need to ensure that your submission is trained by launching less than MINERL_TRAINING_MAX_INSTANCES instances
MINERL_TRAINING_MAX_INSTANCES = int(os.getenv("MINERL_TRAINING_MAX_INSTANCES", 5))
# You need to ensure that your submission is trained within allowed training time.
# Round 1: Training timeout is 15 minutes
# Round 2: Training timeout is 4 days
MINERL_TRAINING_TIMEOUT = int(os.getenv("MINERL_TRAINING_TIMEOUT_MINUTES", 4 * 24 * 60))
# The dataset is available in data/ directory from repository root.
MINERL_DATA_ROOT = os.getenv("MINERL_DATA_ROOT", os.path.dirname(__file__) + "/data")


def main():
    imitation_arguments = [
        "--epochs",
        "100",
        "--save-every-updates",
        "20000",
        MINERL_DATA_ROOT,
        os.path.dirname(__file__) + "/train/discrete_v0.pth_steps_250000",
        "MineRLObtainDiamond-v0",
        "MineRLObtainIronPickaxe-v0",
        "--no-flipping",
    ]

    os.makedirs(os.path.dirname(__file__) + "/train", exist_ok=True)
    train_args = replay_train.parser.parse_args(imitation_arguments)
    sequential_train.main(train_args)


if __name__ == "__main__":
    main()
