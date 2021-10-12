from argparse import ArgumentParser
import os
import time
import warnings
import multiprocessing as mp
import yaml
import numpy as np
from utils import override_config

parser = ArgumentParser()
parser.add_argument(
    '--random_seed', type=int, default=0)  # if random_seed == 0, random random seed will be used
parser.add_argument(
    '--progress_bar', action='store_true', default=False)
parser.add_argument(
    '--jobs_per_gpu', type=int, default=1)
parser.add_argument(
    '--ngpu', type=int, default=4)
parser.add_argument(
    '--config', '-c', required=True)
parser.add_argument(
    '--episode', '-e', required=True)
parser.add_argument(
    '--expert_train_only', action='store_true', default=False)
parser.add_argument('--log-dir', '-l', required=True)
parser.add_argument('--override', default='')
args = parser.parse_args()

if args.random_seed == 0:
    args.random_seed = int(time.time()) % 1000

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


def run_getidx():
    command = 'srun0 python get_idx.py'
    command += ' --random_seed ' + str(args.random_seed)
    command += ' -l ' + args.log_dir
    command += ' -c ' + args.config
    command += ' -e ' + args.episode
    command += ' --override "' + args.override + '"'
    return os.system(command)


def run_ssl_sbatch(idxs):
    command = 'sbatch '
    command += '-W --gres=gpu:1 -n {num_job} ./run_script/run_general.sh '.format(num_job=len(idxs))

    for idx in idxs:
        command += "'python early-ssl.py --idx {}".format(idx)
        command += " --random_seed " + str(args.random_seed)
        if args.progress_bar:
            command += " --progress_bar"
        command += " -l " + args.log_dir
        command += " -c " + args.config
        command += " -e " + args.episode
        command += ' --override "' + args.override + '"'
        command += "' "

    command += ";"
    return os.system(command)


def run_ssl(idx):
    command = 'srun1'
    command += ' python early-ssl.py --idx {}'.format(idx)
    command += ' --random_seed ' + str(args.random_seed)
    if args.progress_bar:
        command += " --progress_bar"
    command += ' -l ' + args.log_dir
    command += ' -c ' + args.config
    command += ' -e ' + args.episode
    command += ' --override "' + args.override + '"'
    return os.system(command)


def run_main():
    command = 'srun1'
    command += ' python main.py '
    command += ' --random_seed ' + str(args.random_seed)
    command += ' -l ' + args.log_dir
    command += ' -c ' + args.config
    command += ' -e ' + args.episode
    command += ' --override "' + args.override + '"'
    return os.system(command)


if __name__ == '__main__':
    print("random seed: {}".format(args.random_seed))
    if args.jobs_per_gpu > 3:
        warnings.warn("Warning! {} jobs_per_gpu may be too many for gpu:normal.".format(args.jobs_per_gpu))

    # make sure logdir exsits
    if not os.path.exists(os.path.join(config['log_dir'], str(os.getpid()))):
        os.makedirs(os.path.join(config['log_dir'], str(os.getpid())))

    run_getidx()
    nidxs = 0
    with open(os.path.join(config['log_dir'], 'idx_sets.npy'), 'rb') as f:
        nidxs = len(np.load(f, allow_pickle=True))

    # parallel self-supervised training
    if args.jobs_per_gpu == 1:
        run_func = run_ssl
        inputs = range(nidxs)
    else:
        run_func = run_ssl_sbatch
        idxs = []
        nchunk = (nidxs // args.jobs_per_gpu)
        if nidxs % args.jobs_per_gpu != 0:
            nchunk += 1
        for i in range(nchunk):
            st = i * args.jobs_per_gpu
            en = ((i + 1) * args.jobs_per_gpu) if i < nchunk - 1 else nidxs
            idxs.append(list(range(st, en)))
        inputs = idxs

    with mp.Pool(args.ngpu) as p:
        p.map(run_func, inputs)

    if not args.expert_train_only:
        print("Main job started!!")
        run_main()

