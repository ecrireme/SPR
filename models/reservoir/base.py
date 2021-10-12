from abc import ABC, abstractmethod
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

class rsvrBase(ABC):
    def __init__(self, config, size):
        super().__init__()
        self.config = config
        self.rsvr = {}
        self.rsvr_total_size = size
        self.n = 0


    @abstractmethod
    def update(self, **args):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def sample(self, num):
        pass

    @abstractmethod
    def __str__(self):
        pass

    def is_full(self):
        return len(self) == self.rsvr_total_size

    def write(self, writer:SummaryWriter, step):
        writer.add_text(
            'train/reservoir_summary',
            str(self), step
        )

    def get(self, data_type):
        return self.rsvr[data_type][:len(self)]

    def get_dataloader(self, batch_size, drop_last, shuffle):
        indices = torch.arange(len(self))
        class DataSet():
            def __len__(self):
                return len(indices)
            def __getitem__(self, idx):
                return idx

        def collate_fn(idxs):
            return {data_type: self.get(data_type)[indices[idxs]] for data_type in self.rsvr.keys()}

        return DataLoader(DataSet(), batch_size, drop_last=drop_last, shuffle=shuffle, collate_fn=collate_fn)
