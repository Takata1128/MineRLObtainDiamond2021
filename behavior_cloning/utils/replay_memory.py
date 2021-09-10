import random
import numpy as np


class ArbitraryReplayMemory:
    def __init__(self, max_size):
        self.capacity = max_size
        self.replay_memory = [None for i in range(max_size)]

        # index for the next sample
        self.ctr = 0

        # How many items are in the replay memory
        self.size = 0

    def __len__(self):
        return self.size

    def add(self, item):
        self.replay_memory[self.ctr] = item
        self.ctr += 1
        self.size = max(self.size, self.ctr)

        if self.ctr == self.capacity:
            self.ctr = 0

    def get_batch(self, batch_size):
        random_idxs = random.sample(range(self.size), batch_size)
        data = [self.replay_memory[idx] for idx in random_idxs]
        return data
