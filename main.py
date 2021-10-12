#!/usr/bin/env python3
from argparse import ArgumentParser
from pprint import pprint
import os
import resource
import random
import yaml
import torch
import colorful
import numpy as np
from tensorboardX import SummaryWriter
from data import DataScheduler
from models.SPR import SPR
from train import train_model
from utils import setup_logger, override_config

# Increase maximum number of open files from 1024 to 4096
# as suggested in https://github.com/pytorch/pytorch/issues/973
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

parser = ArgumentParser()
parser.add_argument(
    '--random_seed', type=int, default=0)
parser.add_argument(
    '--config', '-c', default='configs/ccmodel-coco.yaml'
)
parser.add_argument(
    '--episode', '-e', default='episodes/coco-split.yaml'
)
parser.add_argument('--log-dir', '-l')
parser.add_argument('--resume-ckpt', default=None)
parser.add_argument('--override', default='')


def main():
    args = parser.parse_args()
    logger = setup_logger()

    ## Use below for slurm setting.
    slurm_job_id = os.getenv('SLURM_JOB_ID', 'nojobid')
    slurm_proc_id = os.getenv('SLURM_PROC_ID', None)

    unique_identifier = str(slurm_job_id)
    if slurm_proc_id is not None:
        unique_identifier = unique_identifier + "_" + str(slurm_proc_id)

    # Load config
    config_path = args.config
    episode_path = args.episode

    if args.resume_ckpt and not args.config:
        base_dir = os.path.dirname(os.path.dirname(args.resume_ckpt))
        config_path = os.path.join(base_dir, 'config.yaml')
        episode_path = os.path.join(base_dir, 'episode.yaml')
    config = yaml.load(open(config_path), Loader=yaml.FullLoader)
    episode = yaml.load(open(episode_path), Loader=yaml.FullLoader)
    config['data_schedule'] = episode
    config['random_seed'] = args.random_seed
    if 'corruption_percent' not in config:
        config['corruption_percent'] = 0
    config = override_config(config, args.override)

    # Set log directory
    config['log_dir'] = os.path.join(args.log_dir, unique_identifier)
    if not args.resume_ckpt and os.path.exists(config['log_dir']):
        logger.warning('%s already exists' % config['log_dir'])
        input('Press enter to continue')

    # print the configuration
    print(colorful.bold_white("configuration:").styled_string)
    pprint(config)
    print(colorful.bold_white("configuration end").styled_string)

    if args.resume_ckpt and not args.log_dir:
        config['log_dir'] = os.path.dirname(
            os.path.dirname(args.resume_ckpt)
        )

    # Save config
    os.makedirs(config['log_dir'], mode=0o755, exist_ok=True)
    if not args.resume_ckpt or args.config:
        config_save_path = os.path.join(config['log_dir'], 'config.yaml')
        episode_save_path = os.path.join(config['log_dir'], 'episode.yaml')
        yaml.dump(config, open(config_save_path, 'w'))
        yaml.dump(episode, open(episode_save_path, 'w'))
        print(colorful.bold_yellow('config & episode saved to {}'.format(config['log_dir'])).styled_string)

    # Build components
    if args.random_seed != 0:
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
    data_scheduler = DataScheduler(config)

    writer = SummaryWriter(config['log_dir'])
    model = SPR(config, writer)

    if args.resume_ckpt:
        model.load_state_dict(torch.load(args.resume_ckpt))
    model.to(config['device'])
    train_model(config, model, data_scheduler, writer)

    print(colorful.bold_pink("\nThank you and Good Job Computer").styled_string)


if __name__ == '__main__':
    main()
