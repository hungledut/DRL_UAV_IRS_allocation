import torch
from mappo_model import MAPPO
from environment import IRS_env
from mappo_runner import Runner_MAPPO
from arguments import parse_args
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':

    args = parse_args()
    runner = Runner_MAPPO(args, env_name="UAV network", number=1, seed=30, testing = True) # default seed = 300
    runner.agent_n.load_model("UAV network","1",args.consider_cloud)
    runner.run_episode_mpe()
