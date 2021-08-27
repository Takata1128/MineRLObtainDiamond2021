from tqdm import tqdm
import numpy as np
import torch as th
from torch import nn
import gym
import minerl
import minerl.data
from model import ConvNet, ResNet


"""
Your task: Implement behavioural cloning for MineRLTreechop-v0.

Behavioural cloning is perhaps the simplest way of using a dataset of demonstrations to train an agent:
learn to predict what actions they would take, and take those actions.
In other machine learning terms, this is almost like building a classifier to classify observations to
different actions, and taking those actions.

For simplicity, we build a limited set of actions ("agent actions"), map dataset actions to these actions
and train on the agent actions. During evaluation, we transform these agent actions (integerse) back into
MineRL actions (dictionaries).

To do this task, fill in the "TODO"s and remove `raise NotImplementedError`s.

Note: For this task you need to download the "MineRLTreechop-v0" dataset. See here:
https://minerl.readthedocs.io/en/latest/tutorials/data_sampling.html#downloading-the-minerl-dataset-with-minerl-data-download
"""


class ActionShaping(gym.ActionWrapper):
    def __init__(self, env, camera_angle=10, always_attack=False):
        super().__init__(env)

        self.camera_angle = camera_angle
        self.always_attack = always_attack
        self._actions = [
            [("attack", 1)],
            [("forward", 1)],
            [("forward", 1), ("jump", 1)],
            [("camera", [-self.camera_angle, 0])],
            [("camera", [self.camera_angle, 0])],
            [("camera", [0, -self.camera_angle])],
            [("camera", [0, self.camera_angle])],
        ]

        self.actions = []
        for actions in self._actions:
            act = self.env.action_space.noop()
            for a, v in actions:
                act[a] = v
                if self.always_attack:
                    act["attack"] = 1
                self.actions.append(act)

        self.action_space = gym.spaces.Discrete(len(self.actions))

    def action(self, action):
        return self.actions[action]


def agent_action_to_environment(noop_action, agent_action, always_attack):
    camera_angle = 5
    _actions = [
        [("attack", 1)],
        [("forward", 1)],
        [("forward", 1), ("jump", 1)],
        [("camera", [-camera_angle, 0])],
        [("camera", [camera_angle, 0])],
        [("camera", [0, -camera_angle])],
        [("camera", [0, camera_angle])],
    ]

    res = []
    for actions in _actions:
        act = noop_action.copy()
        for a, v in actions:
            act[a] = v
            if always_attack:
                act["attack"] = 1
            res.append(act)
    return res[agent_action]


def environment_action_batch_to_agent_actions(dataset_actions, camera_margin=5):
    # There are dummy dimensions of shape one
    camera_actions = dataset_actions["camera"].squeeze()
    attack_actions = dataset_actions["attack"].squeeze()
    forward_actions = dataset_actions["forward"].squeeze()
    jump_actions = dataset_actions["jump"].squeeze()
    batch_size = len(camera_actions)
    actions = np.zeros((batch_size,), dtype=np.int)

    for i in range(batch_size):
        if camera_actions[i][0] < -camera_margin:
            actions[i] = 3
        elif camera_actions[i][0] > camera_margin:
            actions[i] = 4
        elif camera_actions[i][1] < -camera_margin:
            actions[i] = 5
        elif camera_actions[i][1] > camera_margin:
            actions[i] = 6
        elif forward_actions[i] == 1:
            if jump_actions[i] == 1:
                actions[i] = 2
            else:
                actions[i] = 1
        elif attack_actions[i] == 1:
            actions[i] = 0
        else:
            actions[i] = -1

    return actions


def train(env: str, model: nn.Module, epochs: int):
    # Path to where MineRL dataset resides (should contain "MineRLTreechop-v0" directory)
    DATA_DIR = "./data"
    # How many times we train over dataset and how large batches we use.
    # Larger batch size takes more memory but generally provides stabler learning.
    BATCH_SIZE = 64
    LEARNING_RATE = 0.00005

    data = minerl.data.make(env, data_dir=DATA_DIR, num_workers=4)
    iterator = minerl.data.BufferedBatchIter(data)

    optimizer = th.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_function = nn.CrossEntropyLoss()

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
        actions = environment_action_batch_to_agent_actions(dataset_actions)
        assert actions.shape == (
            obs.shape[0],
        ), "Array from environment_action_batch_to_agent_actions should be of shape {}".format(
            (obs.shape[0],)
        )

        # Remove samples that had no corresponding action
        mask = actions != -1
        obs = obs[mask]
        actions = actions[mask]

        logits = model(th.from_numpy(obs).float().cuda())
        loss = loss_function(logits, th.from_numpy(actions).long().cuda())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iter_count += 1
        losses.append(loss.item())
        if (iter_count % 1000) == 0:
            mean_loss = sum(losses) / len(losses)
            tqdm.write("Iteration {}. Loss {:<10.3f}".format(iter_count, mean_loss))
            losses.clear()
            th.save(model, "./train/behavioural_cloning_state_dict.pth")

    # Store the network
    del data


from gym.wrappers import Monitor


def enjoy():
    # Load up the trained network
    network = th.load("./train/behavioural_cloning_state_dict.pth").cuda()
    env = Monitor(gym.make("MineRLTreechop-v0"), "./videos", force=True)
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
            logits = network(obs)
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
    number_of_actions = 7
    network = th.load("./train/behavioural_cloning_state_dict.pth").cuda()
    # network = ResNet((3, 64, 64), number_of_actions).cuda()
    # train("MineRLTreechop-v0", network, 3)
    train("MineRLObtainDiamond-v0", network, 3)
    # ... then play it on the environment to see how it does
    enjoy()
