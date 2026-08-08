from three_phase_BFS import three_phase_BFS_ENV
from matd3 import MATD3
import torch
import matplotlib.pyplot as plt

from env_sample import GYMEnv

class Arguments():
    def __init__(self):
        # Environment setting
        self.max_train_steps = 10000  # Maximum training steps
        self.max_episodes=3000
        self.episode_limit = 100  # Maximum steps for each episode
        self.evaluate_freq = 5000  # The number of evaluating episodes
        self.evaluate_times = 3  # Evaluate times
        self.max_action = 1.0  # Max action
        self.algorithm = "MATD3"  # MADDPG
        self.buffer_size = 50000  # The capacity of the replay buffer
        self.batch_size = 256  # Batch size
        self.hidden_dim = 1024  # Hidden dimension of the network
        self.noise_std_init = 0.2  # The std of Gaussian noise for exploration
        self.noise_std_min = 0.05  # The minimum std of Gaussian noise for exploration
        self.noise_decay_steps = 72000  # How many steps before the noise_std decays to the minimum
        self.use_noise_decay = True  # Whether to decay the noise_std
        self.lr_a = 0.0001  # Learning rate of the actor
        self.lr_c = 0.0001  # Learning rate of the critic
        self.gamma = 0.98  # Discount factor
        self.tau = 0.001  # Softly update the target network
        self.use_orthogonal_init = True  # Whether to use orthogonal initialization
        self.use_grad_clip = True  # Whether to use gradient clip

        # TD3MORE(暂未使用)
        self.policy_noise = 0.2  # Noise added to target policy during critic update
        self.noise_clip = 0.5  # Range to clip target policy noise
        self.policy_freq = 2  # Frequency of delayed policy updates


# 模拟运行训练好的模型
def sim():
    args = Arguments()
    args.noise_std_decay = (args.noise_std_init - args.noise_std_min) / args.noise_decay_steps
    args.N = 1  # 环境中智能体的数量
    args.obs_dim_total = 4
    args.obs_dim_n = [4]  # observation dimensions of N agents
    args.action_dim_n = [2]  # action dimensions of N agents
    noise_std = args.noise_std_min  # 记录噪声标准差

    agent_good_n = [MATD3(args, i, torch.device('cuda:0')) for i in range(args.N)]
    for s_agent_id in range(args.N):
        agent_good_n[s_agent_id].load_model('BFS', 'MATD3', 2, 1000, s_agent_id)

    s_env = GYMEnv()
    s_obs_n = s_env.reset()
    for step in range(24):
        s_a_n = [agent.choose_action(s_obs, noise_std) for agent, s_obs in zip(agent_good_n, s_obs_n)]
        s_obs_next_n, s_r_n, s_done_n,info = s_env.step(s_a_n)  # 步进
        s_obs_n = s_obs_next_n #182 167

        #print s_obs_n
        print(s_obs_n)





sim()