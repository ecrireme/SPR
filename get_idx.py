from argparse import ArgumentParser
import os
import random
import yaml
import torch
import colorful
import tqdm
import numpy as np
from tensorboardX import SummaryWriter
from data import DataScheduler
from models.SPR import SPR
from utils import override_config

parser = ArgumentParser()
parser.add_argument(
    '--random_seed', type=int, default=0)
parser.add_argument(
    '--ssl_slarge', default='False')
parser.add_argument(
    '--ssl_nepoch', default=3000)
parser.add_argument(
    '--ngpu', default=4)
parser.add_argument(
    '--config', '-c', required=True)
parser.add_argument(
    '--episode', '-e', required=True)
parser.add_argument('--log-dir', '-l', required=True)
parser.add_argument('--override', default='')
args = parser.parse_args()

def save_idx():
    config = yaml.load(open(args.config), Loader=yaml.FullLoader)
    episode = yaml.load(open(args.episode), Loader=yaml.FullLoader)

    config['data_schedule'] = episode
    config['random_seed'] = args.random_seed
    config = override_config(config, args.override)
    if 'corruption_percent' not in config:
        config['corruption_percent'] = 0
    config['log_dir'] = os.path.join(os.path.dirname(args.log_dir),
                                     'noiserate_{}'.format(config['corruption_percent']),
                                     'expt_{}'.format(config['expert_train_epochs']),
                                     'randomseed_{}'.format(args.random_seed))
    if not os.path.exists(config['log_dir']):
        os.makedirs(config['log_dir'])

    done = True
    for i in range(len(episode)):
        if not os.path.exists(os.path.join(config['log_dir'], 'task{}_idxs.npy'.format(i))):
            done = False
    if done:
        return

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    data_scheduler = DataScheduler(config)

    config['device'] = "cpu"
    writer = SummaryWriter(config['log_dir'])
    model = SPR(config, writer)

    print("Index gathering")
    idxs = []
    for _, ((x, y, corrupt, idx), _) in tqdm.tqdm(enumerate(data_scheduler), desc='idx gathering'):
        for _ in range(config['batch_iter']):
            for i in range(len(x)):
                model.delay_buffer.update(imgs=x[i: i + 1], cats=y[i: i + 1], corrupts=corrupt[i: i + 1], idxs=idx[i: i + 1])

                if model.delay_buffer.is_full():
                    idxs.append(model.delay_buffer.get('idxs').tolist())
                    model.delay_buffer.reset()


    # save index
    idx = np.asarray(idxs)
    with open(os.path.join(config['log_dir'], 'idx_sets.npy'), 'wb') as f:
        np.save(f, idx, allow_pickle=True)
    print(colorful.bold_gold("Total number of Expert training: {}".format(len(idxs))))

if __name__ == '__main__':
    save_idx()
