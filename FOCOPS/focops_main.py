import argparse
import gym
import torch.nn as nn
import torch
import time
from data_generator import DataGenerator
from models import GaussianPolicy, Value
from environment import get_threshold
from utils.logger import Logger
from utils.math import gaussian_kl
from utils.pytorch import graph_detach, to_dytype_device, to_device, torch_to_numpy
from utils.misc import println
from utils.running_stats import RunningStats
from collections import deque
import numpy as np
from gym_torcs import TorcsEnv
from torch.utils.data import DataLoader


class FOCOPS:
    """
    Implement FOCOPS algorithm
    """
    def __init__(self,
                 env,
                 policy_net,
                 value_net,
                 cvalue_net,
                 pi_optimizer,
                 vf_optimizer,
                 cvf_optimizer,
                 num_epochs,
                 mb_size,
                 c_gamma,
                 lam,
                 delta,
                 eta,
                 nu,
                 nu_lr,
                 nu_max,
                 cost_lim,
                 l2_reg,
                 score_queue,
                 cscore_queue,
                 logger):


        self.env = env

        self.policy = policy_net
        self.value_net = value_net
        self.cvalue_net = cvalue_net

        self.pi_optimizer = pi_optimizer
        self.vf_optimizer = vf_optimizer
        self.cvf_optimizer = cvf_optimizer

        self.pi_loss = None
        self.vf_loss = None
        self.cvf_loss = None

        self.num_epochs = num_epochs
        self.mb_size = mb_size

        self.c_gamma = c_gamma
        self.lam = lam
        self.delta = delta
        self.eta = eta
        self.cost_lim = cost_lim

        self.nu = nu
        self.nu_lr = nu_lr
        self.nu_max = nu_max

        self.l2_reg = l2_reg

        self.logger = logger
        self.score_queue = score_queue
        self.cscore_queue = cscore_queue


    def update_params(self, rollout, dtype, device):

        # Convert data to tensor
        obs = torch.Tensor(rollout['states'][:, None, :]).to(dtype).to(device)
        act = torch.Tensor(rollout['actions'][:, None, :]).to(dtype).to(device)
        vtarg = torch.Tensor(rollout['v_targets'][:, None, :]).to(dtype).to(device).detach()
        adv = torch.Tensor(rollout['advantages'][:, None, :]).to(dtype).to(device).detach()
        cvtarg = torch.Tensor(rollout['cv_targets'][:, None, :]).to(dtype).to(device).detach()
        cadv = torch.Tensor(rollout['c_advantages'][:, None, :]).to(dtype).to(device).detach()

        h = torch.tensor(rollout['h'][None, ...], requires_grad=True, dtype=torch.float).to(device)
        c = torch.tensor(rollout['c'][None, ...], requires_grad=True, dtype=torch.float).to(device)
        nh = torch.tensor(rollout['nh'][None, ...], requires_grad=True, dtype=torch.float).to(device)
        nc = torch.tensor(rollout['nc'][None, ...], requires_grad=True, dtype=torch.float).to(device)

        original_hidden = (h, c)
        next_hidden = (nh, nc)

        # Get log likelihood, mean, and std of current policy
        old_logprob, old_mean, old_std, _ = self.policy.logprob(obs, act, original_hidden)
        old_logprob, old_mean, old_std = to_dytype_device(dtype, device, old_logprob, old_mean, old_std)
        old_logprob, old_mean, old_std = graph_detach(old_logprob, old_mean, old_std)

        h = torch.tensor(rollout['h'][:, None, :], requires_grad=True, dtype=torch.float).to(device)
        c = torch.tensor(rollout['c'][:, None, :], requires_grad=True, dtype=torch.float).to(device)
        nh = torch.tensor(rollout['nh'][:, None, :], requires_grad=True, dtype=torch.float).to(device)
        nc = torch.tensor(rollout['nc'][:, None, :], requires_grad=True, dtype=torch.float).to(device)

        # Store in TensorDataset for minibatch updates
        dataset = torch.utils.data.TensorDataset(obs, act, vtarg, adv, cvtarg, cadv,
                                                 old_logprob, old_mean, old_std, h, c, nh, nc)
        loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.mb_size, shuffle=True)
        avg_cost = rollout['avg_cost']

        # print(avg_cost)
        # print(self.cost_lim)
        # Update nu
        self.nu += self.nu_lr * (avg_cost - self.cost_lim)
        if self.nu < 0:
            self.nu = 0
        elif self.nu > self.nu_max:
            self.nu = self.nu_max

        for epoch in range(self.num_epochs):

            for _, (obs_b, act_b, vtarg_b, adv_b, cvtarg_b, cadv_b,
                    old_logprob_b, old_mean_b, old_std_b, h_b, c_b, nh_b, nc_b) in enumerate(loader):
                # print(h_b.size())
                # print(h_b)

                h_b = h_b.permute(1, 0, 2)
                c_b = c_b.permute(1, 0, 2)
                nh_b = nh_b.permute(1, 0, 2)
                nc_b = nc_b.permute(1, 0, 2)
                # print(h_b.size())
                # print(h_b)
                hidden = (h_b, c_b)
                next_hidden = (nh_b, nc_b)
                # Update reward critic
                mse_loss = nn.MSELoss()
                vf_pred = self.value_net(obs_b, hidden)
                # print("vf_pred", vf_pred)
                # print("vtarg_b", vtarg_b)
                self.vf_loss = mse_loss(vf_pred, vtarg_b)
                # print(self.vf_loss)
                # weight decay
                for param in self.value_net.parameters():
                    self.vf_loss += param.pow(2).sum() * self.l2_reg
                self.vf_optimizer.zero_grad()
                self.vf_loss.backward()
                self.vf_optimizer.step()

                # Update cost critic
                cvf_pred = self.cvalue_net(obs_b, hidden)
                # print("cvf_pred", cvf_pred)
                # print("cvtarg_b", cvtarg_b)
                self.cvf_loss = mse_loss(cvf_pred, cvtarg_b)
                # print(self.cvf_loss)
                # weight decay
                for param in self.cvalue_net.parameters():
                    self.cvf_loss += param.pow(2).sum() * self.l2_reg
                self.cvf_optimizer.zero_grad()
                self.cvf_loss.backward()
                self.cvf_optimizer.step()


                # Update policy
                logprob, mean, std, new_hidden = self.policy.logprob(obs_b, act_b, hidden)
                kl_new_old = gaussian_kl(mean, std, old_mean_b, old_std_b)
                ratio = torch.exp(logprob - old_logprob_b)
                self.pi_loss = (kl_new_old - (1 / self.lam) * ratio * (adv_b - self.nu * cadv_b)) \
                          * (kl_new_old.detach() <= self.eta).type(dtype)
                self.pi_loss = self.pi_loss.mean()
                print(self.pi_loss)
                self.pi_optimizer.zero_grad()
                self.pi_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 40)
                self.pi_optimizer.step()


            # Early stopping
            logprob, mean, std, _ = self.policy.logprob(obs, act, original_hidden)
            kl_val = gaussian_kl(mean, std, old_mean, old_std).mean().item()
            if kl_val > self.delta:
                println('Break at epoch {} because KL value {:.4f} larger than {:.4f}'.format(epoch + 1, kl_val, self.delta))
                break



        # Store everything in log
        self.logger.update('MinR', np.min(self.score_queue))
        self.logger.update('MaxR', np.max(self.score_queue))
        self.logger.update('AvgR', np.mean(self.score_queue))
        self.logger.update('MinC', np.min(self.cscore_queue))
        self.logger.update('MaxC', np.max(self.cscore_queue))
        self.logger.update('AvgC', np.mean(self.cscore_queue))
        self.logger.update('nu', self.nu)


        # Save models
        self.logger.save_model('policy_params', self.policy.state_dict())
        self.logger.save_model('value_params', self.value_net.state_dict())
        self.logger.save_model('cvalue_params', self.cvalue_net.state_dict())
        self.logger.save_model('pi_optimizer', self.pi_optimizer.state_dict())
        self.logger.save_model('vf_optimizer', self.vf_optimizer.state_dict())
        self.logger.save_model('cvf_optimizer', self.cvf_optimizer.state_dict())
        self.logger.save_model('pi_loss', self.pi_loss)
        self.logger.save_model('vf_loss', self.vf_loss)
        self.logger.save_model('cvf_loss', self.cvf_loss)


def train(args):

    # Initialize data type
    dtype = torch.float32
    torch.set_default_dtype(dtype)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Initialize environment
    env = TorcsEnv()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Initialize neural nets
    policy = GaussianPolicy(obs_dim, act_dim, args.hidden_size, args.activation, args.logstd)
    value_net = Value(obs_dim, args.hidden_size, args.activation)
    cvalue_net = Value(obs_dim, args.hidden_size, args.activation)
    policy.to(device)
    value_net.to(device)
    cvalue_net.to(device)

    # Initialize optimizer
    pi_optimizer = torch.optim.Adam(policy.parameters(), args.pi_lr)
    vf_optimizer = torch.optim.Adam(value_net.parameters(), args.vf_lr)
    cvf_optimizer = torch.optim.Adam(cvalue_net.parameters(), args.cvf_lr)

    # Initialize learning rate scheduler
    lr_lambda = lambda it: max(1.0 - it / args.max_iter_num, 0)
    pi_scheduler = torch.optim.lr_scheduler.LambdaLR(pi_optimizer, lr_lambda=lr_lambda)
    vf_scheduler = torch.optim.lr_scheduler.LambdaLR(vf_optimizer, lr_lambda=lr_lambda)
    cvf_scheduler = torch.optim.lr_scheduler.LambdaLR(cvf_optimizer, lr_lambda=lr_lambda)

    # Store hyperparameters for log
    hyperparams = vars(args)

    # Initialize RunningStat for state normalization, score queue, logger
    running_stat = RunningStats(clip=5)
    score_queue = deque(maxlen=100)
    cscore_queue = deque(maxlen=100)
    logger = Logger(hyperparams)

    # Get constraint bounds
    cost_lim = get_threshold('Torcs', constraint=args.constraint)

    # Initialize and train FOCOPS agent
    agent = FOCOPS(env, policy, value_net, cvalue_net,
                   pi_optimizer, vf_optimizer, cvf_optimizer,
                   args.num_epochs, args.mb_size,
                   args.c_gamma, args.lam, args.delta, args.eta,
                   args.nu, args.nu_lr, args.nu_max, cost_lim,
                   args.l2_reg, score_queue, cscore_queue, logger)

    start_time = time.time()
    episode_num = 0
    t = 0
    episode_timesteps = 0
    # results_episode = open('results.txt', 'w')
    while t < 150000:

        # Update iteration for model
        agent.logger.save_model('iter', iter)

        # Collect trajectories
        data_generator = DataGenerator(obs_dim, act_dim, args.batch_size, args.max_eps_len)
        rollout, episode_timesteps, total_abs_center, travelled, episode_reward, total_steer_agent, total_steer_driver = data_generator.run_traj(env, agent.policy, agent.value_net, agent.cvalue_net,
                                          running_stat, agent.score_queue, agent.cscore_queue,
                                          args.gamma, args.c_gamma, args.gae_lam, args.c_gae_lam,
                                          dtype, device, args.constraint, episode_num, episode_timesteps)

        # Update FOCOPS parameters
        episode_num += 1
        t += episode_timesteps
        agent.update_params(rollout, dtype, device)

        # results_episode.write(
        #     "{0} {1} {2} {3:.2f} {4:.2f} {5:.2f} {6:.2f} {7:.2f}".format(episode_num, episode_timesteps, t,
        #                                                                  total_abs_center,
        #                                                                  travelled, episode_reward, total_steer_agent,
        #                                                                  total_steer_driver))
        # results_episode.write("\n")

        # Update learning rates
        pi_scheduler.step()
        vf_scheduler.step()
        cvf_scheduler.step()

        # Update time and running stat
        agent.logger.update('time', time.time() - start_time)
        agent.logger.update('running_stat', running_stat)

        # Save and print values
        agent.logger.dump()

    # results_episode.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch FOCOPS Implementation')
    parser.add_argument('--env-id', default='Torchs-v3',
                        help='Name of Environment (default: Humanoid-v3')
    parser.add_argument('--constraint', default='distance',
                        help='Constraint setting (default: velocity')
    parser.add_argument('--activation', default="tanh",
                        help='Activation function for policy/critic network (Default: tanh)')
    parser.add_argument('--hidden_size', type=float, default=(64, 64),
                        help='Tuple of size of hidden layers for policy/critic network (Default: (64, 64))')
    parser.add_argument('--logstd', type=float, default=-0.5,
                        help='Log std of Policy (Default: -0.5)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor for reward (Default: 0.99)')
    parser.add_argument('--c-gamma', type=float, default=0.99,
                        help='Discount factor for cost (Default: 0.99)')
    parser.add_argument('--gae-lam', type=float, default=0.95,
                        help='Lambda value for GAE for reward (Default: 0.95)')
    parser.add_argument('--c-gae-lam', type=float, default=0.95,
                        help='Lambda value for GAE for cost (Default: 0.95)')
    parser.add_argument('--l2-reg', type=float, default=1e-3,
                        help='L2 Regularization Rate (default: 1e-3)')
    parser.add_argument('--pi-lr', type=float, default=3e-4,
                        help='Learning Rate for policy (default: 3e-4)')
    parser.add_argument('--vf-lr', type=float, default=3e-4,
                        help='Learning Rate for value function (default: 3e-4)')
    parser.add_argument('--cvf-lr', type=float, default=3e-4,
                        help='Learning Rate for c-value function (default: 3e-4)')
    parser.add_argument('--lam', type=float, default=1.5,
                        help='Inverse temperature lambda (default: 1.5)')
    parser.add_argument('--delta', type=float, default=0.02,
                        help='KL bound (default: 0.02)')
    parser.add_argument('--eta', type=float, default=0.02,
                        help='KL bound for indicator function (default: 0.02)')
    parser.add_argument('--nu', type=float, default=1,
                        help='Cost coefficient (default: 0)')
    parser.add_argument('--nu_lr', type=float, default=0.01,
                        help='Cost coefficient learning rate (default: 0.01)')
    parser.add_argument('--nu_max', type=float, default=2.0,
                        help='Maximum cost coefficient (default: 2.0)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random Seed (default: 0)')
    parser.add_argument('--max-eps-len', type=int, default=512,
                        help='Maximum length of episode (default: 400)')
    parser.add_argument('--mb-size', type=int, default=64,
                        help='Minibatch size per update (default: 64)')
    parser.add_argument('--batch-size', type=int, default=2048,
                        help='Batch Size per Update (default: 2048)')
    parser.add_argument('--num-epochs', type=int, default=10,
                        help='Number of passes through each minibatch per update (default: 10)')
    parser.add_argument('--max-iter-num', type=int, default=500,
                        help='Number of Main Iterations (default: 500)')
    args = parser.parse_args()

    train(args)