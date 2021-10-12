from .prs import PRS, SubStream_Container
import random
import torch
import numpy as np
from collections import deque


class DelayBuffer(PRS):
    """
    Delayed Buffer for new data samples that need to be learned in chunks.
    and used to made the decision later whether to enter the buffer or not.
    """
    def reset(self):
        """reset the buffer.
        """
        self.rsvr = dict()
        self.rsvr_available_idx = deque(range(self.rsvr_total_size))
        self.substreams = SubStream_Container(self.rsvr_total_size)
        self.n = 0
        np.random.seed(self.config['random_seed'])
        random.seed(self.config['random_seed'])
        torch.manual_seed(self.config['random_seed'])
        return
