import argparse
import os
import logging
import torch
import numpy as np
import time
import torch
from utils import *
import agents

def fix_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def set_logger(args):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fileHandler = logging.FileHandler(args.exp_dir+f'/{args.time_str}.txt', mode='w')
    fileHandler.setLevel(logging.INFO)
    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    consoleHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)
    logger.addHandler(fileHandler)
    return logger

CONFIG = {
    'fmnist': {'milestones': [30, 60, 90],  'model_name': 'alexnet'},
    'cifar10': {'milestones': [30, 60, 90], 'model_name': 'allcnn'},
    'cifar100': {'milestones': [60, 120, 160], 'model_name': 'resnet18'},
    'svhn': {'milestones': [30, 60, 90], 'model_name': 'vgg11'},
    }
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds',          type=int,       default=[2023, 2024, 2025])
    parser.add_argument('--repeat',         type=int,       default=3)
    parser.add_argument('--num_epochs',     type=int,       default=2)
    parser.add_argument('--milestones',     type=list,      default=[60, 120, 160])
    parser.add_argument('--model_name',     type=str,       default='vgg11')

    parser.add_argument('--lr',             type=float,     default=0.1)
    parser.add_argument('--wd',             type=float,     default=5e-4)
    parser.add_argument('--momentum',       type=float,     default=0.9)
    parser.add_argument('--gamma',          type=float,     default=0.2)
    parser.add_argument('--warm',           type=int,       default=1)

    parser.add_argument('--dataset',        type=str,       default='svhn')
    parser.add_argument('--batch_size',     type=int,       default=512)
    parser.add_argument('--num_workers',    type=int,       default=8)
    parser.add_argument('--patience',       type=int,       default=30)
    parser.add_argument('--agent',          type=str,       default='Base')
    parser.add_argument('--device',         type=str,       default='cuda')
    parser.add_argument('--retrain',        action='store_true')
    parser.add_argument('--unlearn_class',  nargs='+',      type=int)

    args = parser.parse_args()
    args.time_str = time.strftime("%m-%d-%H-%M", time.localtime())
    args.exp_dir = f'./save/{args.dataset.lower()}/{args.model_name}'
    os.makedirs(args.exp_dir, exist_ok=True)

    if args.dataset.lower() == 'fmnist':
        args.n_channels = 1
    else:
        args.n_channels = 3
    
    if args.dataset.lower() == 'cifar100':
        args.num_classes = 100
    else:
        args.num_classes = 10
    return args

def run(args):
    logger = set_logger(args)
    for seed in args.seeds:
        args.seed = seed
        fix_seed(seed)
        logger.info(f'Running seed {seed}')
        logger.info(f'Experiment dir: {args.exp_dir}')
        logger.info(f'Unlearn class: {args.unlearn_class}')
        logger.info(vars(args))
        agent = getattr(agents, args.agent)(args, logger)
        agent.train()

if __name__ == '__main__':
    args = get_args()
    run(args)

