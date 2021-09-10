from tqdm import tqdm
import numpy as np
import os
import torch as th
from torch import nn
import gym
import minerl
import minerl.data
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from net.model import ConvNet, ResNetImpala
from config import ACTIONS_LIST, ACTIONS_DICT, ACTION_NVEC


def dataset_action_batch_to_actions(
    dataset_actions, actions_dict, actions_list, camera_margin=3
):
    camera_actions = dataset_actions["camera"].squeeze()
    batch_size = len(camera_actions)
    res_actions = np.zeros(
        (
            batch_size,
            len(actions_list),
        ),
        dtype=np.int64,
    )
    for type, actions in dataset_actions.items():
        actions = actions.squeeze()
        if type in ["back", "left", "right", "sneak", "sprint"]:
            continue
        if type == "camera":
            for i in range(len(actions)):
                if actions[i][0] < -camera_margin:
                    res_actions[i][3] = 0
                elif actions[i][0] > camera_margin:
                    res_actions[i][3] = 2
                if actions[i][1] < -camera_margin:
                    res_actions[i][4] = 0
                elif actions[i][1] > camera_margin:
                    res_actions[i][4] = 2
        else:
            for i in range(len(actions)):
                res_actions[i][actions_list.index(type)] = actions_dict[type].index(
                    actions[i]
                )
    return res_actions


def train(env: str, model: nn.Module, epochs: int):
    # Path to where MineRL dataset resides (should contain "MineRLTreechop-v0" directory)
    DATA_DIR = "/root/minerl-intro/data"
    # How many times we train over dataset and how large batches we use.
    # Larger batch size takes more memory but generally provides stabler learning.
    BATCH_SIZE = 32
    LEARNING_RATE = 0.00005

    data = minerl.data.make(env, data_dir=DATA_DIR, num_workers=4)
    iterator = minerl.data.BufferedBatchIter(data)

    optimizer = th.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    iter_count = 0
    losses = []
    for dataset_obs, dataset_actions, _, _, _ in tqdm(
        iterator.buffered_batch_iter(num_epochs=epochs, batch_size=BATCH_SIZE)
    ):
        # We only use camera observations here
        obs = dataset_obs["pov"].squeeze().astype(np.float32)
        # Transpose observations to be channel-first (BCHW instead of BHWC)
        obs = obs.transpose(0, 3, 1, 2)
        # Normalize observations, otherwise the neural network will get spooked
        obs /= 255.0

        # Turn dataset actions into agent actions
        actions = dataset_action_batch_to_actions(
            dataset_actions, ACTIONS_DICT, ACTIONS_LIST
        )

        probs_list = model(th.from_numpy(obs).float().cuda())
        loss = 0
        actions = actions.transpose(1, 0)

        for probs, action in zip(probs_list, actions):
            loss += nn.CrossEntropyLoss()(probs, th.from_numpy(action).long().cuda())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iter_count += 1
        losses.append(loss.item())
        if (iter_count % 1000) == 0:
            mean_loss = sum(losses) / len(losses)
            tqdm.write("Iteration {}. Loss {:<10.3f}".format(iter_count, mean_loss))
            losses.clear()
            th.save(
                model.state_dict(),
                os.path.dirname(__file__) + "/train/behavioural_cloning_state_dict.pth",
            )

    # Store the network
    del data


from gym.wrappers import Monitor


def enjoy(env: str, model):
    # Load up the trained network
    env = Monitor(gym.make(env), os.path.dirname(__file__) + "/videos", force=True)
    num_actions = 7
    action_list = np.arange(num_actions)
    # Play 10 games with the model
    for game_i in range(10):
        obs = env.reset()
        done = False
        reward_sum = 0
        while not done:
            obs = th.from_numpy(
                obs["pov"].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
            ).cuda()
            logits = model(obs)
            # Turn logits into probabilities
            probabilities = th.softmax(logits, dim=1)[0]
            # Into numpy
            probabilities = probabilities.detach().cpu().numpy()
            agent_action = np.random.choice(action_list, p=probabilities)

            noop_action = env.action_space.noop()
            environment_action = agent_action_to_environment(
                noop_action, agent_action, False
            )

            obs, reward, done, info = env.step(environment_action)
            reward_sum += reward
        print("Game {}, total reward {}".format(game_i, reward_sum))

    env.close()


if __name__ == "__main__":
    # First train the model...
    network = ResNetImpala((3, 64, 64), ACTION_NVEC, False).cuda()
    # network = th.load_state_dict(
    #     th.load(os.path.dirname(__file__) + "/train/behavioural_cloning_state_dict.pth")
    # ).cuda()

    # train("MineRLTreechop-v0", network, 3)
    train("MineRLObtainDiamond-v0", network, 20)
    # ... then play it on the environment to see how it does
    # enjoy("MineRLObtainDiamond-v0", network)
