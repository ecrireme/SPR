from argparse import ArgumentParser
import os
import random
import datetime
import yaml
import colorful
import tqdm
import torch
import numpy as np
from torch.utils.data import Subset, DataLoader
from tensorboardX import SummaryWriter
from data import DataScheduler
from components import Net
from utils import override_config

colors = [colorful.bold_gold, colorful.bold_green, colorful.bold_orchid, colorful.bold_beige, colorful.bold_coral, colorful.bold_cyan, colorful.bold_khaki]

parser = ArgumentParser()
parser.add_argument(
    '--idx', type=int, required=True)
parser.add_argument(
    '--random_seed', type=int, default=0)
parser.add_argument(
    '--progress_bar', action='store_true', default=False)
parser.add_argument(
    '--config', '-c', required=True)
parser.add_argument(
    '--episode', '-e', required=True)
parser.add_argument('--log-dir', '-l', required=True)
parser.add_argument('--override', default='')
args = parser.parse_args()

config = yaml.load(open(args.config), Loader=yaml.FullLoader)
episode = yaml.load(open(args.episode), Loader=yaml.FullLoader)

config['data_schedule'] = episode
config = override_config(config, args.override)

if 'corruption_percent' not in config:
    config['corruption_percent'] = 0
config['log_dir'] = os.path.join(os.path.dirname(args.log_dir),
                                    'noiserate_{}'.format(config['corruption_percent']),
                                    'expt_{}'.format(config['expert_train_epochs']),
                                    'randomseed_{}'.format(args.random_seed))

writer = SummaryWriter(config['log_dir'])

class CustomDataset():
    """This Dataset puts data on gpu for fast loading"""
    def __init__(self, dataset, device="cuda:0"):
        self.dataset = dataset
        self.datas = self.load_data(dataset, device)
        self.device = device

        self.i = 0
        self.indices = torch.arange(len(dataset))

    def load_data(self, dataset, device):
        # load data into memory by type of tensor
        sample_data = dataset[0]
        datas = [torch.empty((len(dataset), *sample_data[i].shape), dtype=sample_data[i].dtype, device=device) if isinstance(sample_data[i], torch.Tensor) \
                 else torch.empty((len(dataset), ), dtype=torch.long, device=device) for i in range(len(sample_data))]

        for i in tqdm.trange(len(dataset), desc='prepare dataset'):
            data = dataset[i]
            for j in range(len(data)):
                if isinstance(data[j], torch.Tensor):
                    datas[j][i] = data[j].to(device)
                else:
                    datas[j][i] = data[j]
        return datas

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return idx

    def collate_fn(self, idxs):
        batch = [self.datas[j][self.indices[idxs]] for j in range(len(self.datas))]
        return batch

def get_dataset():
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    data_scheduler = DataScheduler(config)
    with open(os.path.join(config['log_dir'], 'idx_sets.npy'), 'rb') as f:
        idxs = np.load(f, allow_pickle=True)[args.idx]

    dataset_name = list(data_scheduler.datasets.keys())[0]
    dataset = data_scheduler.datasets[dataset_name]
    dataset = CustomDataset(Subset(dataset, idxs))
    return dataset


def train(dataset):
    batch_size = min(config['expert_batch_size'], len(dataset))
    expert = Net[config['net']](config)
    expert.setup_optimizer(config['optimizer'])
    lr_config = config['lr_scheduler']
    if lr_config['type'] == 'CosineAnnealingLR':
        lr_config['options'].update({'T_max': config['expert_train_epochs']})
    expert.setup_lr_scheduler(lr_config)
    expert.train()
    expert.init_ntxent(config, batch_size=batch_size)

    dataloader = DataLoader(dataset, batch_size=batch_size,
                            num_workers=0, drop_last=True, shuffle=True, collate_fn=dataset.collate_fn)
    step = 0
    if args.idx == 0 and args.progress_bar:
        epoch_range = tqdm.trange(config['expert_train_epochs'], desc='progress')
    else:
        epoch_range = range(config['expert_train_epochs'])
    for epoch in epoch_range:
        if epoch % 200 == 0:
            print(colors[args.idx % len(colors)]("IDX {}:".format(args.idx)).styled_string + " {}% done".format(round(100 * epoch / config['expert_train_epochs'])))
        for x, *_ in dataloader:
            expert.zero_grad()
            loss = expert.get_selfsup_loss(x)
            loss.backward()
            expert.optimizer.step()

            writer.add_scalar(
                'expert_train_loss/t{}'.format(args.idx), loss, step)
            step += 1
        if epoch >= 10:
            expert.lr_scheduler.step()

    writer.flush()
    torch.save(expert.state_dict(), os.path.join(config['log_dir'], 'model{}.ckpt'.format(args.idx)))


if __name__ == '__main__':
    print(colors[args.idx % len(colors)]("IDX {} has been started. \njob id: {}\t pid: {}".format(args.idx, os.getenv("SLURM_JOB_ID"), os.getpid())).styled_string)
    begin_time = datetime.datetime.now()
    if not os.path.exists(os.path.join(config['log_dir'], 'model{}.ckpt'.format(args.idx))):
        dataset = get_dataset()
        train(dataset)
    print(colors[args.idx % len(colors)]("IDX {} has been ended. \t execution time: {}".format(args.idx, datetime.datetime.now() - begin_time)).styled_string)
