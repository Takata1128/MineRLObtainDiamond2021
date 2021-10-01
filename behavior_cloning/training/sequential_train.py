from argparse import ArgumentParser
from typing import Dict
import minerl
from itertools import cycle
import os

import torch

from behavior_cloning.net.model import ResNetImpala
from behavior_cloning.wrappers.action_wrappers import ObtainDiamondActions
from behavior_cloning.wrappers.observation_wrappers import ObtainDiamondObservation
from behavior_cloning.utils.replay_memory import ArbitraryReplayMemory
from behavior_cloning.utils.util import unzip_states_or_actions
from multiprocessing import Process, Queue
import time
from collections import deque
import numpy as np
import random

parser = ArgumentParser("Train models to do imitation learning")

parser.add_argument("data_dir", type=str, help="Path to MineRL dataset.")
parser.add_argument("model", type=str, help="Path to model.")
parser.add_argument(
    "datasets",
    type=str,
    nargs="+",
    help="List of datasets to use for the training. First one should include biggest action space",
)
parser.add_argument("--workers", type=int, default=2, help="Number of dataset workers")
parser.add_argument("--max-seqlen", type=int, default=1, help="Max length per loader")
parser.add_argument(
    "--seqs-per-update",
    type=int,
    default=1,
    help="How many sequences are loaded per one update (mini-batch) train",
)
parser.add_argument(
    "--replay-size",
    type=int,
    default=100000,
    help="Maximum number of individual training samples to store in replay memory.",
)
parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train.")
parser.add_argument(
    "--save-every-updates",
    type=int,
    default=50000,
    help="How many iterations between saving a snapshot of the model",
)
parser.add_argument("--batch-size", type=int, default=32, help="Ye' olde batch size.")
parser.add_argument("--lr", type=float, default=0.00005, help="Adam learning rate.")
parser.add_argument(
    "--lr-decay", type=float, default=0.0, help="Decay for learning rate."
)
parser.add_argument(
    "--target-value",
    type=float,
    default=0.995,
    help="Target value where cross-entropy aims.",
)
parser.add_argument("--l2", type=float, default=0.001, help="L2 regularizer weight.")
parser.add_argument(
    "--gamma",
    type=float,
    default=1.0,
    help="Additional gamma correction (on top of the regular correction).",
)
parser.add_argument(
    "--numeric-df",
    action="store_true",
    help="Use scalars for representing inventory rather than one-hot encoding.",
)
parser.add_argument(
    "--no-augmentation", action="store_true", help="Do not use augmentation for images."
)
parser.add_argument(
    "--no-flipping",
    action="store_true",
    help="Do not do horizontal flipping for augmentation.",
)


# trajectories : [states/action/rewards][batch][seq_len]
def trajectories_to_replay_memory(trajectories, replay_memory, args):
    for trajectory in trajectories:
        states, actions, rewards = trajectory
        for i in range(len(states) - 1, -1, -1):
            # skip no-ops
            if actions[i] == 0:
                continue
            replay_memory.add((states[i], actions[i]))


def get_training_batch(replay_memory, batch_size):
    raw_batch = replay_memory.get_batch(batch_size)
    inputs = None
    outputs = None

    if isinstance(raw_batch[0][0], tuple) or isinstance(raw_batch[0][0], list):
        inputs = []
        for i in range(len(raw_batch[0][0])):
            inputs.append(np.array([raw_batch[b][0][i] for b in range(batch_size)]))
    else:
        inputs = np.array([raw_batch[b][0] for b in range(batch_size)])

    outputs = np.array([raw_batch[b][1] for b in range(batch_size)], dtype=np.int32)

    return inputs, outputs


def process_data(in_sample, obs_processor, act_processor, do_flipping):
    states = in_sample[0]
    actions = in_sample[1]
    rewards = in_sample[2]

    states = unzip_states_or_actions(states)
    actions = unzip_states_or_actions(actions)

    states = list(map(lambda state: obs_processor.dict_to_tuple(state), states))
    actions = list(map(lambda action: act_processor.dict_to_discrete(action), actions))

    return [states, actions, rewards]


def calc_loss(logits, out_y, parameters):
    ce_loss = 0
    l2_loss = torch.tensor(0.0, requires_grad=True)
    ce_loss += torch.nn.CrossEntropyLoss()(
        logits, torch.squeeze(torch.from_numpy(out_y).long().cuda())
    )
    for w in parameters:
        l2_loss = l2_loss + torch.norm(w) ** 2
    return ce_loss, l2_loss


def train_model(model, optimizer, train_inputs, train_outputs, l2_weight=0.0):
    pov = torch.tensor(train_inputs[0], device="cuda", dtype=torch.float32).squeeze()
    direct_input = torch.tensor(
        train_inputs[1], device="cuda", dtype=torch.float32
    ).squeeze()
    distribs, _ = model((pov, direct_input))
    ce_loss, l2_loss = calc_loss(distribs.logits, train_outputs, model.parameters())
    loss = ce_loss + l2_weight * l2_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def main(args):
    workers_per_loader = args.workers // len(args.datasets)

    data_loaders = [
        minerl.data.make(
            dataset, data_dir=args.data_dir, num_workers=workers_per_loader
        )
        for dataset in args.datasets
    ]

    obs_processor = ObtainDiamondObservation(
        data_loaders[0].observation_space,
        augmentation=not args.no_augmentation,
        gamma_correction=args.gamma,
        numeric_df=args.numeric_df,
    )

    act_processor = ObtainDiamondActions(data_loaders[0].action_space)
    action_n = act_processor.action_space.n
    image_shape = obs_processor.observation_space[0].shape
    direct_shape = obs_processor.observation_space[1].shape

    model = ResNetImpala(image_shape, action_n, True, direct_shape).cuda()
    if os.path.exists(args.model):
        model.load_state_dict(torch.load(args.model))
        model.eval()
    else:
        torch.save(model.state_dict(), args.model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    data_iterators = cycle(
        [
            minerl.data.BufferedBatchIter(data).buffered_batch_iter(
                batch_size=args.batch_size, num_epochs=args.epochs
            )
            for data in data_loaders
        ]
    )

    replay_memory = ArbitraryReplayMemory(args.replay_size)

    num_updates = 0
    start_time = time.time()
    average_losses = deque(maxlen=1000)
    last_save_updates = 0

    states = None
    acts = None
    rewards = None
    state_primes = None
    dones = None

    trajectories = []

    for data_iterator in data_iterators:
        try:
            states, acts, rewards, state_primes, dones = next(data_iterator)
        except StopIteration:
            break
        trajectories.append(
            process_data(
                [states, acts, rewards],
                obs_processor,
                act_processor,
                not args.no_flipping,
            )
        )
        if len(trajectories) >= args.seqs_per_update:
            trajectories_to_replay_memory(trajectories, replay_memory, args)
            trajectories.clear()

            if len(replay_memory) > args.batch_size:
                train_inputs, train_outputs = get_training_batch(
                    replay_memory, args.batch_size
                )
                # train_model
                loss_value = train_model(
                    model, optimizer, train_inputs, train_outputs, args.l2
                )
                average_losses.append(loss_value)
                num_updates += 1

            if (num_updates % 1000) == 0:
                time_passed = int(time.time() - start_time)
                print(
                    "Time: {:<8} Updates: {:<8} AvrgLoss: {:.4f}".format(
                        time_passed, num_updates, np.mean(average_losses)
                    )
                )
            if (num_updates - last_save_updates) >= args.save_every_updates:
                torch.save(
                    model.state_dict(), args.model + "_steps_{}".format(num_updates)
                )
                last_save_updates = num_updates
    torch.save(model.state_dict(), args.model + "_steps_{}".format(num_updates))


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
