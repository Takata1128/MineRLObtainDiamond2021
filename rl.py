"""An example of training PPO against OpenAI Gym Atari Envs.
This script is an example of training a PPO agent on Atari envs.
To train PPO for 10M timesteps on Breakout, run:
    python train_ppo_ale.py
To train PPO using a recurrent model on a flickering Atari env, run:
    python train_ppo_ale.py --recurrent --flicker --no-frame-stack
"""
import argparse
import functools
import os
import numpy as np
import torch
from torch import nn

import minerl
import gym

import pfrl
from pfrl import experiments, utils
from pfrl.agents import PPO
from pfrl.policies import SoftmaxCategoricalHead
from pfrl.wrappers import atari_wrappers

from behavior_cloning.wrappers.action_wrappers import ObtainDiamondActions
from behavior_cloning.wrappers.observation_wrappers import ObtainDiamondObservation
from behavior_cloning.net.model import ResNetImpala


class ActionShaping(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_processor = ObtainDiamondActions(env.action_space)
        self.action_space = self.action_processor.action_space

    def action(self, action):
        return self.action_processor.discrete_to_dict(action)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env", type=str, default="MineRLObtainDiamond-v0", help="Gym Env ID."
    )
    parser.add_argument(
        "--gpu", type=int, default=0, help="GPU device ID. Set to -1 to use CPUs only."
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=6,
        help="Number of env instances run in parallel.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed [0, 2 ** 32)")
    parser.add_argument(
        "--outdir",
        type=str,
        default="results",
        help=(
            "Directory path to save output files."
            " If it does not exist, it will be created."
        ),
    )
    parser.add_argument(
        "--steps", type=int, default=10 ** 7, help="Total time steps for training."
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=2 * 60 * 60,  # 30 minutes with 60 fps
        help="Maximum number of frames for each episode.",
    )
    parser.add_argument("--lr", type=float, default=2.5e-4, help="Learning rate.")
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=300000,
        help="Interval (in timesteps) between evaluation phases.",
    )
    parser.add_argument(
        "--eval-n-runs",
        type=int,
        default=5,
        help="Number of episodes ran in an evaluation phase.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        default=False,
        help="Run demo episodes, not training.",
    )
    parser.add_argument(
        "--load",
        type=str,
        default="",
        help=(
            "Directory path to load a saved agent data from"
            " if it is a non-empty string."
        ),
    )
    parser.add_argument(
        "--log-level",
        type=int,
        default=20,
        help="Logging level. 10:DEBUG, 20:INFO etc.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=False,
        help="Render env states in a GUI window.",
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        default=False,
        help=(
            "Monitor env. Videos and additional information are saved as output files."
        ),
    )
    parser.add_argument(
        "--update-interval",
        type=int,
        default=128 * 8,
        help="Interval (in timesteps) between PPO iterations.",
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        default=32 * 8,
        help="Size of minibatch (in timesteps).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of epochs used for each PPO iteration.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10000,
        help="Interval (in timesteps) of printing logs.",
    )
    parser.add_argument(
        "--recurrent",
        action="store_true",
        default=False,
        help="Use a recurrent model. See the code for the model definition.",
    )
    parser.add_argument(
        "--flicker",
        action="store_true",
        default=False,
        help=(
            "Use so-called flickering Atari, where each"
            " screen is blacked out with probability 0.5."
        ),
    )
    parser.add_argument(
        "--no-frame-stack",
        action="store_true",
        default=False,
        help=(
            "Disable frame stacking so that the agent can only see the current screen."
        ),
    )
    parser.add_argument(
        "--checkpoint-frequency",
        type=int,
        default=None,
        help="Frequency at which agents are stored.",
    )
    args = parser.parse_args()

    import logging

    logging.basicConfig(level=args.log_level)

    # Set a random seed used in PFRL.
    utils.set_random_seed(args.seed)
    # Set different random seeds for different subprocesses.
    # If seed=0 and processes=4, subprocess seeds are [0, 1, 2, 3].
    # If seed=1 and processes=4, subprocess seeds are [4, 5, 6, 7].
    process_seeds = np.arange(args.num_envs) + args.seed * args.num_envs
    assert process_seeds.max() < 2 ** 32

    args.outdir = experiments.prepare_output_dir(args, args.outdir)
    print("Output files are saved in {}".format(args.outdir))

    def make_env(idx, test):
        # Use different random seeds for train and test envs
        process_seed = int(process_seeds[idx])
        env_seed = 2 ** 32 - 1 - process_seed if test else process_seed
        env = ActionShaping(gym.make(args.env))
        env.seed(env_seed)
        if args.monitor:
            env = pfrl.wrappers.Monitor(
                env, args.outdir, mode="evaluation" if test else "training"
            )
        if args.render:
            env = pfrl.wrappers.Render(env)
        return env

    def make_batch_env(test):
        vec_env = pfrl.envs.MultiprocessVectorEnv(
            [
                functools.partial(make_env, idx, test)
                for idx, env in enumerate(range(args.num_envs))
            ]
        )
        # if not args.no_frame_stack:
        #     vec_env = pfrl.wrappers.VectorFrameStack(vec_env, 4)
        return vec_env

    # sample_env = make_batch_env(test=False)
    # print("Observation space", sample_env.observation_space)
    # print("Action space", sample_env.action_space)
    # n_actions = sample_env.action_space.n
    # obs_n_channels = sample_env.observation_space.low.shape[0]
    # del sample_env

    action_n = 98
    image_shape = (3, 64, 64)
    direct_shape = (172,)

    model = ResNetImpala(image_shape, action_n, True, direct_shape).cuda()

    model.load_state_dict(
        torch.load(os.path.dirname(__file__) + "/train/before_ppo.pth")
    )
    model.eval().cuda()

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    test_env = ActionShaping(gym.make(args.env))
    obs_processor = ObtainDiamondObservation(test_env.observation_space)
    del test_env

    def phi(x):
        x = obs_processor.dict_to_tuple(x)
        return x

    agent = PPO(
        model,
        opt,
        gpu=args.gpu,
        phi=phi,
        update_interval=args.update_interval,
        minibatch_size=args.batchsize,
        epochs=args.epochs,
        clip_eps=0.1,
        clip_eps_vf=None,
        standardize_advantages=True,
        entropy_coef=1e-2,
        recurrent=args.recurrent,
        max_grad_norm=0.5,
    )
    if args.load:
        agent.load(args.load)

    if args.demo:
        eval_stats = experiments.eval_performance(
            env=make_batch_env(test=True),
            agent=agent,
            n_steps=None,
            n_episodes=args.eval_n_runs,
        )
        print(
            "n_runs: {} mean: {} median: {} stdev: {}".format(
                args.eval_n_runs,
                eval_stats["mean"],
                eval_stats["median"],
                eval_stats["stdev"],
            )
        )
    else:
        step_hooks = []

        # Linearly decay the learning rate to zero
        def lr_setter(env, agent, value):
            for param_group in agent.optimizer.param_groups:
                param_group["lr"] = value

        step_hooks.append(
            experiments.LinearInterpolationHook(args.steps, args.lr, 0, lr_setter)
        )

        experiments.train_agent_batch_with_evaluation(
            agent=agent,
            env=make_batch_env(False),
            eval_env=make_batch_env(True),
            outdir=args.outdir,
            steps=args.steps,
            eval_n_steps=None,
            eval_n_episodes=args.eval_n_runs,
            checkpoint_freq=args.checkpoint_frequency,
            eval_interval=args.eval_interval,
            log_interval=args.log_interval,
            save_best_so_far_agent=False,
            step_hooks=step_hooks,
        )

    torch.save(model.state_dict(), os.path.dirname(__file__) + "/train/after_ppo.pth")


if __name__ == "__main__":
    main()
