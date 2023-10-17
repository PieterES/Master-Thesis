import numpy as np
import torch
from utils.pytorch import torch_to_numpy
import threading
import time
from random import randint



def change_variable():
    # Function to change the distracted state of the driver.
    global DISTRACTED
    global terminate
    if DISTRACTED == False:
        # If the driver is attentive, it remains attentive for 4-5 seconds until becoming distracted
        for i in range(randint(15,20)):
            time.sleep(0.5)
            if terminate:
                break
        DISTRACTED = True
        # print(DISTRACTED)

    else:
        # If the driver is distracted, it remains distracted for 2-3 seconds until becoming attentive again.
        for i in range(randint(4,16)):
            time.sleep(0.5)
            if terminate:
                break
        DISTRACTED = False



class DataGenerator:
    """
    A data generator used to collect trajectories for on-policy RL with GAE
    References:
        https://github.com/Khrylx/PyTorch-RL
        https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
        https://github.com/ikostrikov/pytorch-trpo
    """
    def __init__(self, obs_dim, act_dim, batch_size, max_eps_len):

        # Hyperparameters
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.batch_size = batch_size
        self.max_eps_len = max_eps_len
        hidden_size = 128

        # Batch buffer
        self.obs_buf = np.zeros((batch_size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((batch_size, act_dim),  dtype=np.float32)
        self.vtarg_buf = np.zeros((batch_size, 1), dtype=np.float32)
        self.adv_buf = np.zeros((batch_size, 1), dtype=np.float32)
        self.cvtarg_buf = np.zeros((batch_size, 1), dtype=np.float32)
        self.cadv_buf = np.zeros((batch_size, 1), dtype=np.float32)

        self.h_buff = np.zeros((batch_size, hidden_size), dtype=np.float32)
        self.nh_buff = np.zeros((batch_size, hidden_size), dtype=np.float32)

        self.c_buff = np.zeros((batch_size, hidden_size), dtype=np.float32)
        self.nc_buff = np.zeros((batch_size, hidden_size), dtype=np.float32)

        # Episode buffer
        self.obs_eps = np.zeros((max_eps_len, obs_dim),  dtype=np.float32)
        self.next_obs_eps = np.zeros((max_eps_len, obs_dim),  dtype=np.float32)
        self.act_eps = np.zeros((max_eps_len, act_dim),  dtype=np.float32)
        self.rew_eps = np.zeros((max_eps_len, 1),  dtype=np.float32)
        self.cost_eps = np.zeros((max_eps_len, 1), dtype=np.float32)
        self.eps_len = 0
        self.not_terminal = 1

        self.h_eps = np.zeros((max_eps_len, hidden_size), dtype=np.float32)
        self.nh_eps = np.zeros((max_eps_len, hidden_size), dtype=np.float32)

        self.c_eps = np.zeros((max_eps_len, hidden_size), dtype=np.float32)
        self.nc_eps = np.zeros((max_eps_len, hidden_size), dtype=np.float32)

        self.device = torch.device("cuda")

        # Pointer
        self.ptr = 0

    def run_traj(self, env, policy, value_net, cvalue_net, running_stat,
                 score_queue, cscore_queue, gamma, c_gamma, gae_lam, c_gae_lam,
                 dtype, device, constraint, episode_num, episode_timesteps):

        batch_idx = 0

        cost_ret_hist = []

        avg_eps_len = 0
        num_eps = 0

        canBecomeDistracted = True


        results_step = open('stepresults.txt', 'a')
        results_episode = open('results.txt', 'a')
        global DISTRACTED
        global terminate
        canBecomeDistracted = True
        terminate = False
        DISTRACTED = False
        noise = 1.05
        obs, done = env.reset(), False

        while batch_idx < self.batch_size:
            obs, done = env.reset(), False
            if running_stat is not None:
                obs = running_stat.normalize(obs)
            ret_eps = 0
            cost_ret_eps = 0
            total_abs_center = 0
            episode_reward = 0
            total_steer_agent = 0
            total_steer_driver = 0
            travelled = 0
            PREVSTEER = 0
            PREVACCEL = 0
            PREV_AGENT_STEER = 0
            PREV_AGENT_ACCEL = 0
            TOTAL_COST = 0
            episode_timesteps = 0
            hidden = policy.get_initial_states()
            for t in range(self.max_eps_len):
                act, next_hidden = policy.get_act(torch.Tensor(obs).to(dtype).to(device), hidden)
                act = torch_to_numpy(act).squeeze()

                if canBecomeDistracted and episode_timesteps > 10 and threading.active_count() < 2:
                    thread = threading.Thread(target=change_variable)
                    thread.start()

                if randint(0, 25) == 10:
                    if noise == 0.95:
                        noise = 1.05
                    elif noise == 1.05:
                        noise = 0.95

                next_obs, rew, done, travelled, trackPos, new_distracted, agentaccel, agentsteer, driveraccel, driversteer, savedaction, PREVSTEER, PREVACCEL, PREV_AGENT_STEER, PREV_AGENT_ACCEL, info = env.step(
                    act, DISTRACTED=DISTRACTED, noise=noise, episode_timesteps=episode_timesteps, PREVSTEER=PREVSTEER,
                    PREVACCEL=PREVACCEL, PREV_AGENT_STEER=PREV_AGENT_STEER, PREV_AGENT_ACCEL=PREV_AGENT_ACCEL)

                cost = info['cost']
                TOTAL_COST += cost
                episode_reward += rew
                total_abs_center += np.abs(trackPos)
                total_steer_agent += np.abs(agentsteer)
                total_steer_driver += np.abs(driversteer)

                ret_eps += rew
                cost_ret_eps += (c_gamma ** t) * cost

                episode_timesteps += 1

                if running_stat is not None:
                    next_obs = running_stat.normalize(next_obs)

                # Store in episode buffer
                self.obs_eps[t] = obs
                self.act_eps[t] = act
                self.next_obs_eps[t] = next_obs
                self.rew_eps[t] = rew
                self.cost_eps[t] = cost
                h, c = hidden
                nh, nc = next_hidden
                self.h_eps[t] = h.detach().cpu()
                self.c_eps[t] = c.detach().cpu()
                self.nh_eps[t] = nh.detach().cpu()
                self.nc_eps[t] = nc.detach().cpu()

                obs = next_obs
                self.eps_len += 1
                batch_idx += 1
                hidden = next_hidden
                results_step.write(
                    "{0} {1} {2} {3} {4:.2f} {5:.2f} {6:.2f} {7:.2f} {8:.2f} {9:.2f} {10:.2f} {11}".format(num_eps,
                                                                                                      episode_timesteps, t,
                                                                                                      new_distracted,
                                                                                                      travelled,
                                                                                                      episode_reward,
                                                                                                      trackPos, agentaccel,
                                                                                                      agentsteer,
                                                                                                      driveraccel,
                                                                                                      driversteer,
                                                                                                      cost))
                results_step.write("\n")
                if travelled > 2000:
                    done = True
                # Store return for score if only episode is terminal
                if done or t == self.max_eps_len - 1:
                    if done:
                        self.not_terminal = 0
                    results_episode.write(
                        "{0} {1} {2} {3:.2f} {4:.2f} {5:.2f} {6:.2f} {7:.2f} {8}".format(episode_num, episode_timesteps,
                        t, total_abs_center, travelled, episode_reward, total_steer_agent, total_steer_driver, TOTAL_COST))
                    results_episode.write("\n")
                    hidden = policy.get_initial_states()
                    score_queue.append(ret_eps)
                    cscore_queue.append(cost_ret_eps)
                    cost_ret_hist.append(cost_ret_eps)
                    print(episode_num, episode_timesteps, travelled, episode_reward)
                    num_eps += 1
                    avg_eps_len += (self.eps_len - avg_eps_len) / num_eps

                    episode_timesteps = 0
                    total_abs_center = 0
                    episode_reward = 0
                    total_steer_agent = 0
                    total_steer_driver = 0
                    travelled = 0
                    PREVSTEER = 0
                    PREVACCEL = 0
                    TOTAL_COST = 0
                    PREV_AGENT_STEER = 0
                    PREV_AGENT_ACCEL = 0
                if batch_idx == self.batch_size or done:
                    break

            # Store episode buffer
            self.obs_eps, self.next_obs_eps = self.obs_eps[:self.eps_len], self.next_obs_eps[:self.eps_len]
            self.act_eps, self.rew_eps = self.act_eps[:self.eps_len], self.rew_eps[:self.eps_len]
            self.cost_eps = self.cost_eps[:self.eps_len]
            self.h_eps, self.c_eps = self.h_eps[:self.eps_len], self.c_eps[:self.eps_len]
            self.nh_eps, self.nc_eps = self.nh_eps[:self.eps_len], self.nc_eps[:self.eps_len]

            # Calculate advantage
            adv_eps, vtarg_eps = self.get_advantage(value_net, gamma, gae_lam, dtype, device, mode='reward')
            cadv_eps, cvtarg_eps = self.get_advantage(cvalue_net, c_gamma, c_gae_lam, dtype, device, mode='cost')

            # Update batch buffer
            start_idx, end_idx = self.ptr, self.ptr + self.eps_len
            self.obs_buf[start_idx: end_idx], self.act_buf[start_idx: end_idx] = self.obs_eps, self.act_eps
            self.vtarg_buf[start_idx: end_idx], self.adv_buf[start_idx: end_idx] = vtarg_eps[0], adv_eps[0]
            self.cvtarg_buf[start_idx: end_idx], self.cadv_buf[start_idx: end_idx] = cvtarg_eps[0], cadv_eps[0]
            self.h_buff[start_idx:end_idx], self.c_buff[start_idx:end_idx] = self.h_eps, self.c_eps
            self.nh_buff[start_idx:end_idx], self.nc_buff[start_idx:end_idx] = self.nh_eps, self.nc_eps

            # Update pointer
            self.ptr = end_idx

            # Reset episode buffer and update pointer
            self.obs_eps = np.zeros((self.max_eps_len, self.obs_dim), dtype=np.float32)
            self.next_obs_eps = np.zeros((self.max_eps_len, self.obs_dim), dtype=np.float32)
            self.act_eps = np.zeros((self.max_eps_len, self.act_dim), dtype=np.float32)
            self.rew_eps = np.zeros((self.max_eps_len, 1), dtype=np.float32)
            self.cost_eps = np.zeros((self.max_eps_len, 1), dtype=np.float32)
            self.h_eps = np.zeros((self.max_eps_len, 128), dtype=np.float32)
            self.nh_eps = np.zeros((self.max_eps_len, 128), dtype=np.float32)

            self.c_eps = np.zeros((self.max_eps_len, 128), dtype=np.float32)
            self.nc_eps = np.zeros((self.max_eps_len, 128), dtype=np.float32)

            self.eps_len = 0
            self.not_terminal = 1

        avg_cost = np.mean(cost_ret_hist)
        std_cost = np.std(cost_ret_hist)

        # Normalize advantage functions
        self.adv_buf = (self.adv_buf - self.adv_buf.mean()) / (self.adv_buf.std() + 1e-6)
        self.cadv_buf = (self.cadv_buf - self.cadv_buf.mean()) / (self.cadv_buf.std() + 1e-6)

        return {'states':self.obs_buf, 'actions':self.act_buf,
                'v_targets': self.vtarg_buf,'advantages': self.adv_buf,
                'cv_targets': self.cvtarg_buf, 'c_advantages': self.cadv_buf,
                'avg_cost': avg_cost, 'std_cost': std_cost, 'avg_eps_len': avg_eps_len, 'h':self.h_buff,
                'c':self.c_buff, 'nh':self.nh_buff, 'nc':self.nc_buff}, \
            episode_timesteps, total_abs_center, travelled, episode_reward, total_steer_agent, total_steer_driver


    def get_advantage(self, value_net, gamma, gae_lam, dtype, device, mode):
        gae_delta = np.zeros((self.eps_len, 1))
        adv_eps = np.zeros((self.eps_len, 1))
        # Check if terminal state, if terminal V(S_T) = 0, else V(S_T)
        status = np.ones((self.eps_len, 1))
        status[-1] = self.not_terminal
        prev_adv = 0

        for t in reversed(range(self.eps_len)):
            # Get value for current and next state
            obs_tensor = torch.FloatTensor(self.obs_eps[t][None, :]).to(self.device)
            next_obs_tensor = torch.FloatTensor(self.next_obs_eps[t][None, :]).to(self.device)
            h = torch.tensor(self.h_eps[t][None, ...], requires_grad=True, dtype=torch.float).to(self.device)
            c = torch.tensor(self.c_eps[t][None, ...], requires_grad=True, dtype=torch.float).to(self.device)
            nh = torch.tensor(self.nh_eps[t][None, ...], requires_grad=True, dtype=torch.float).to(self.device)
            nc = torch.tensor(self.nc_eps[t][None, ...], requires_grad=True, dtype=torch.float).to(self.device)
            hidden = (h,c)
            next_hidden = (nh, nc)
            current_val, next_val = torch_to_numpy(value_net(obs_tensor, hidden), value_net(next_obs_tensor, next_hidden))

            # Calculate delta and advantage
            if mode == 'reward':
                gae_delta[t] = self.rew_eps[t] + gamma * next_val * status[t] - current_val
            elif mode =='cost':
                gae_delta[t] = self.cost_eps[t] + gamma * next_val * status[t] - current_val
            adv_eps[t] = gae_delta[t] + gamma * gae_lam * prev_adv
            # Update previous advantage
            prev_adv = adv_eps[t]

        # Get target for value function
        obs_eps_tensor = torch.FloatTensor(self.obs_eps[:, None, :]).to(self.device)
        h = torch.tensor(self.h_eps[None, ...], requires_grad=True, dtype=torch.float).to(self.device)
        c = torch.tensor(self.c_eps[None, ...], requires_grad=True, dtype=torch.float).to(self.device)
        hidden = (h,c)
        vtarg_eps = torch_to_numpy(value_net(obs_eps_tensor, hidden, batch=True)) + adv_eps



        return adv_eps, vtarg_eps
