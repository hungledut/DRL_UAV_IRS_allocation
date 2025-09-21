import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, args):
        self.N = args.N
        self.obs_dim = args.obs_dim
        self.obs_dim_n = args.obs_dim_n
        self.state_dim = args.state_dim
        self.episode_limit = args.episode_limit
        self.batch_size = args.batch_size
        self.episode_num = 0
        self.buffer = None
        self.reset_buffer()
        # create a buffer (dictionary)

    def reset_buffer(self):
        self.buffer = {'obs_n': np.empty([self.batch_size, self.episode_limit, self.N - 1, self.obs_dim]),
                       'obs_n_IRS': np.empty([self.batch_size, self.episode_limit, 1, self.obs_dim_n[-1]]),
                       's': np.empty([self.batch_size, self.episode_limit, self.state_dim]),
                       'v_n': np.empty([self.batch_size, self.episode_limit + 1, self.N]),
                       'a_n': np.empty([self.batch_size, self.episode_limit, self.N]),
                       'a_logprob_n': np.empty([self.batch_size, self.episode_limit, self.N]),
                       'r_n': np.empty([self.batch_size, self.episode_limit, self.N]),
                       'done_n': np.empty([self.batch_size, self.episode_limit, self.N])
                       }
        self.episode_num = 0

    def store_transition(self, episode_step, obs_n, s, v_n, a_n, a_logprob_n, r_n, done_n):
        self.buffer['obs_n'][self.episode_num][episode_step] = obs_n[0:-1]
        self.buffer['obs_n_IRS'][self.episode_num][episode_step] = obs_n[-1]
        self.buffer['s'][self.episode_num][episode_step] = s
        self.buffer['v_n'][self.episode_num][episode_step] = v_n
        self.buffer['a_n'][self.episode_num][episode_step] = a_n
        self.buffer['a_logprob_n'][self.episode_num][episode_step] = a_logprob_n
        self.buffer['r_n'][self.episode_num][episode_step] = r_n
        self.buffer['done_n'][self.episode_num][episode_step] = done_n

    def store_last_value(self, episode_step, v_n):
        self.buffer['v_n'][self.episode_num][episode_step] = v_n
        self.episode_num += 1

    def get_training_data(self):
        batch = {}
        for key in self.buffer.keys():
            if key == 'a_n':
                batch[key] = torch.tensor(self.buffer[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(self.buffer[key], dtype=torch.float32)
        return batch
