import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import argparse
from normalization import Normalization, RewardScaling
from replay_buffer import ReplayBuffer
from mappo_model import MAPPO
from environment import IRS_env
from arguments import parse_args
from mappo_runner import Runner_MAPPO

if __name__ == '__main__':
    args = parse_args()
    runner = Runner_MAPPO(args, env_name="UAV network", number=1, seed=0)
    runner.run()