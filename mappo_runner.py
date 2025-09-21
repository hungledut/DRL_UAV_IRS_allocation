import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import argparse
from normalization import Normalization, RewardScaling
from replay_buffer import ReplayBuffer
from mappo_model import MAPPO
from environment import IRS_env


class Runner_MAPPO:
    def __init__(self, args, env_name , number, seed, testing = False):
        self.args = args
        self.env_name = "UAV network"
        self.number = number
        self.seed = seed
        self.testing = testing
        # Set random seed
        # np.random.seed(self.seed)
        # torch.manual_seed(self.seed)
        # Create env
        self.env = IRS_env() # Discrete action space

        self.env.max_step = self.args.episode_limit # Set the max length of an episode
        print("max_step in testing:", self.env.max_step) 
        self.args.N = 4  # The number of agents
        self.args.obs_dim_n = [309, 309, 309, 9]  # obs dimensions of N agents
        self.args.action_dim_n = [5, 5, 5, 6]  # actions dimensions of N agents
        # Only for homogenous agents environments like Spread in MPE,all agents have the same dimension of observation space and action space
        self.args.obs_dim = self.args.obs_dim_n[0]  # The dimensions of an agent's observation space
        self.args.action_dim = self.args.action_dim_n[0]  # The dimensions of an agent's action space
        self.args.state_dim = np.sum(self.args.obs_dim_n) # The dimensions of global state space（Sum of the dimensions of the local observation space of all agents）
        # print("observation_space=", self.env.observation_space)
        # print("obs_dim_n={}".format(self.args.obs_dim_n))
        # print("action_space=", self.env.action_space)
        # print("action_dim_n={}".format(self.args.action_dim_n))

        # Create N agents
        self.agent_n = MAPPO(self.args)
        self.replay_buffer = ReplayBuffer(self.args)

        # Create a tensorboard
        self.writer = SummaryWriter(log_dir='runs/MAPPO/MAPPO_env_{}_number_{}_seed_{}'.format("UAV network", self.number, self.seed))

        self.evaluate_rewards = []  # Record the rewards during the evaluating
        self.total_steps = 0
        if self.args.use_reward_norm:
            print("------use reward norm------")
            self.reward_norm = Normalization(shape=self.args.N)
        elif self.args.use_reward_scaling:
            print("------use reward scaling------")
            self.reward_scaling = RewardScaling(shape=self.args.N, gamma=self.args.gamma)

    def run(self, ):
        evaluate_num = -1  # Record the number of evaluations
        while self.total_steps < self.args.max_train_steps:
            if self.total_steps // self.args.evaluate_freq > evaluate_num:
                self.evaluate_policy()  # Evaluate the policy every 'evaluate_freq' steps
                evaluate_num += 1
                if self.total_steps%5000 == 0:
                    self.env.plot()

            _, episode_steps = self.run_episode_mpe(evaluate=False)  # Run an episode
            self.total_steps += episode_steps

            if self.replay_buffer.episode_num == self.args.batch_size:
                self.agent_n.train(self.replay_buffer, self.total_steps)  # Training
                self.replay_buffer.reset_buffer()

        self.evaluate_policy()
        # self.env.close()

    def evaluate_policy(self, ):
        evaluate_reward = 0
        for _ in range(self.args.evaluate_times):
            episode_reward, _ = self.run_episode_mpe(evaluate=True)
            evaluate_reward += episode_reward

        evaluate_reward = evaluate_reward / self.args.evaluate_times
        self.evaluate_rewards.append(evaluate_reward)
        print("total_steps:{} \t evaluate_reward:{}".format(self.total_steps, evaluate_reward))
        self.writer.add_scalar('evaluate_step_rewards_{}'.format(self.env_name), evaluate_reward, global_step=self.total_steps)
        # Save the rewards and models
        np.save('./data_train/MAPPO_env_{}_number_{}_consider_cloud_{}.npy'.format(self.env_name, self.number, self.args.consider_cloud), np.array(self.evaluate_rewards))
        self.agent_n.save_model(self.env_name, self.number, self.args.consider_cloud, self.total_steps)

    def run_episode_mpe(self, evaluate=False):
        episode_reward = 0
        obs_n = self.env.reset()

        S_t, N_UAV0_t, N_UAV1_t, N_UAV2_t = 0,0,0,0

        g_t = 0
        l_t_0 = 0
        l_t_1 = 0
        l_t_2 = 0
        w = 0.6

        percentage_users = []

        if self.args.use_reward_scaling:
            self.reward_scaling.reset()
        if self.args.use_rnn:  # If use RNN, before the beginning of each episode，reset the rnn_hidden of the Q network.
            self.agent_n.actor.rnn_hidden = None
            self.agent_n.critic.rnn_hidden = None
        for episode_step in range(self.args.episode_limit):
            a_n, a_logprob_n = self.agent_n.choose_action(obs_n, evaluate=evaluate)  # Get actions and the corresponding log probabilities of N agents
            s = np.concatenate(obs_n).flatten()  # In MPE, global state is the concatenation of all agents' local obs.
            v_n = self.agent_n.get_value(s)  # Get the state values (V(s)) of N agents

            obs_next_n, S, N_UAV0, N_UAV1, N_UAV2, done_n, _ = self.env.step(a_n)

            if S > S_t:
                g_t = 1
            elif S < S_t:
                g_t = -1
            else:
                g_t = 0

            if N_UAV0 > N_UAV0_t:
                l_t_0 = 1
            elif N_UAV0 < N_UAV0_t:
                l_t_0 = -1
            else:
                l_t_0 = 0

            if N_UAV1 > N_UAV1_t:
                l_t_1 = 1
            elif N_UAV1 < N_UAV1_t:
                l_t_1 = -1
            else:
                l_t_1 = 0

            if N_UAV2 > N_UAV2_t:
                l_t_2 = 1
            elif N_UAV2 < N_UAV2_t:
                l_t_2 = -1
            else:
                l_t_2 = 0

            S_t, N_UAV0_t, N_UAV1_t, N_UAV2_t = S, N_UAV0, N_UAV1, N_UAV2

            # print(S_t, N_UAV0_t, N_UAV1_t, N_UAV2_t)

            reward0 = w*l_t_0 + (1-w)*g_t
            reward1 = w*l_t_1 + (1-w)*g_t
            reward2 = w*l_t_2 + (1-w)*g_t
            reward_irs = g_t

            episode_reward += reward0+reward1+reward2+reward_irs

            r_n = [reward0, reward1, reward2, reward_irs]

            if not evaluate:
                if self.args.use_reward_norm:
                    r_n = self.reward_norm(r_n)
                elif args.use_reward_scaling:
                    r_n = self.reward_scaling(r_n)

                # Store the transition
                self.replay_buffer.store_transition(episode_step, obs_n, s, v_n, a_n, a_logprob_n, r_n, done_n)

            obs_n = obs_next_n
            if self.testing == True:
                if episode_step%2 == 0:
                    self.env.plot()
                    print("IRS allocation of ", np.sum(self.env.irs) ,"=", self.env.irs)
                percentage_users.append(S)  # Total number of users = 400
                
            if all(done_n):
                break

        if not evaluate:
            # An episode is over, store v_n in the last step
            s = np.concatenate(obs_n).flatten()
            v_n = self.agent_n.get_value(s)
            self.replay_buffer.store_last_value(episode_step + 1, v_n)

        if self.testing == True:
            print("The number of users served in this episode: {}".format(percentage_users[-1]))
            np.save('percentage_users.npy', np.array(percentage_users))


        return episode_reward, episode_step + 1

