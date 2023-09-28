import argparse
import gym
import numpy as np
import os
# import pybullet_envs  # noqa F401
# import pybulletgym  # noqa F401 register PyBullet enviroments with open ai gym
import torch
from gym_torcs import TorcsEnv

from algos import DDPG, PPO, TD3
from utils import memory
from random import randint
import time
import threading

LSTM = True


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, episode_num, DISTRACTED, noise, episode_timesteps, eval_episodes=1):
    results_eval_step = open('eval_step.txt', 'a')
    policy.eval_mode()
    total_reward = 0.
    t=0
    env = TorcsEnv()
    # env.seed(seed + 100)

    for _ in range(eval_episodes):
        total_pos = 0
        state, done = env.reset(), False
        hidden = None
        while not done:
            t+=1
            if t > 10 and threading.active_count() < 2:
                thread = threading.Thread(target=change_variable)
                thread.start()

            if randint(0, 25) == 10:
                # print('change noise:', noise)
                if noise == 0.95:
                    noise = 1.05
                elif noise == 1.05:
                    noise = 0.95
            action, hidden = policy.select_action(np.array(state), hidden)
            # env.render(mode='human', close=False)
            next_state, reward, done, travelled, trackPos, new_distracted, agentaccel, agentsteer, driveraccel, driversteer, savedaction, info = env.step(
                action, DISTRACTED=DISTRACTED, noise=noise, episode_timesteps=t)
            total_pos += trackPos
            total_reward += reward
            results_eval_step.write(
                "{0} {1} {2} {3:2f} {4:.2f} {5:.2f} {6:.2f} {7:.2f} {8:.2f} {9:.2f}".format(episode_num,
                                                                                                  t,
                                                                                                  new_distracted,
                                                                                                  travelled,
                                                                                                  reward,
                                                                                                  trackPos, agentaccel,
                                                                                                  agentsteer,
                                                                                                  driveraccel,
                                                                                                  driversteer))
            if travelled > 2000 or done == True or t == 512:
                break
    policy.train_mode()
    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {total_reward:.3f}")
    print("---------------------------------------")
    return total_reward, travelled, total_pos, t

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
def main():

    parser = argparse.ArgumentParser()
    # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--policy", default="TD3")
    # OpenAI gym environment name
    parser.add_argument("--env", default="Torcs")
    # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--seed", default=0, type=int)
    # Time steps initial random policy is used
    parser.add_argument("--start_timesteps", default=500, type=int)
    # How often (episodes) we evaluate
    parser.add_argument("--eval_freq", default=1, type=int)
    # Max time steps to run environment
    parser.add_argument("--max_timesteps", default=150000, type=int)
    # Std of Gaussian exploration noise
    parser.add_argument("--expl_noise", default=0.25)
    # Batch size for both actor and critic
    parser.add_argument("--batch_size", default=100, type=int)
    # Memory size
    parser.add_argument("--memory_size", default=1e6, type=int)
    # Learning rate
    parser.add_argument("--lr", default=3e-4, type=float)
    # Discount factor
    parser.add_argument("--discount", default=0.99)
    # Target network update rate
    parser.add_argument("--tau", default=0.005)
    # Noise added to target policy during critic update
    parser.add_argument("--policy_noise", default=0.25)
    # Range to clip target policy noise
    parser.add_argument("--noise_clip", default=0.5)
    # Frequency of delayed policy updates
    parser.add_argument("--policy_freq", default=2, type=int)
    # Model width
    parser.add_argument("--hidden_size", default=128, type=int)
    # Use recurrent policies or not
    parser.add_argument("--recurrent", action="store_false")
    # Save model and optimizer parameters
    parser.add_argument("--save_model", action="store_false")
    # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--load_model", default="")
    # Don't train and just run the model
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    file_name = f"{args.policy}_{LSTM}_{args.hidden_size}_{args.env}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, RNN: {LSTM}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if not os.path.exists("./models"):
        os.makedirs("./models")

    env = TorcsEnv()

    # Set seeds
    # env.seed(args.seed)
    # torch.manual_seed(args.seed)
    # np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # TODO: Add this to parameters
    print("args.recurrent", LSTM)
    recurrent_actor = LSTM
    recurrent_critic = LSTM

    print(state_dim)
    print(action_dim)
    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "hidden_dim": args.hidden_size,
        "discount": args.discount,
        "tau": args.tau,
        "recurrent_actor": LSTM,
        "recurrent_critic": LSTM,
    }

    load = False
    save = True

    # Initialize policy
    if args.policy == "TD3":
        # Target policy smoothing is scaled wrt the action scale
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        policy = TD3.TD3(**kwargs)

    elif args.policy == "DDPG":
        policy = DDPG.DDPG(**kwargs)

    elif args.policy == "PPO":
        # TODO: Add kwargs for PPO
        kwargs["K_epochs"] = 10
        kwargs["eps_clip"] = 0.1
        policy = PPO.PPO(**kwargs)
        args.start_timesteps = 0
        n_update = 256
    if load:
        try:
            policy.load(f"./models/{file_name}")
        except:
            print('no policy to load')
    #
    # if args.test:
    #     eval_policy(policy, args.env, args.seed, eval_episodes=10, test=True)
    #     return

    replay_buffer = memory.ReplayBuffer(
        state_dim, action_dim, args.hidden_size,
        args.memory_size, recurrent=LSTM)
    global DISTRACTED
    global terminate
    canBecomeDistracted = True
    terminate = False
    DISTRACTED = False
    # Evaluate untrained policy
    # total_reward, _, _, _ = eval_policy(policy, 0, DISTRACTED=DISTRACTED, noise=0.95, episode_timesteps=0)
    #
    # best_reward = total_reward

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    PREVSTEER = 0
    PREVACCEL = 0
    PREV_AGENT_STEER = 0
    PREV_AGENT_ACCEL = 0
    hidden = policy.get_initial_states()
    print(hidden)
    total_abs_center = 0
    results_episode = open('results.txt', 'w')
    # eval_ep = open('eval_ep.txt', 'w')
    results_step = open('stepresults.txt', 'w')
    inputs = open('inputs.txt', 'w')
    noise = 1.05
    evalep = 0
    total_steer_agent = 0
    total_steer_driver = 0
    TOTAL_COST = 0
    for t in range(1, int(args.max_timesteps)):
        episode_timesteps += 1
        if canBecomeDistracted and episode_timesteps > 20 and threading.active_count() < 2:
            thread = threading.Thread(target=change_variable)
            thread.start()

        if randint(0, 25) == 10:
            # print('change noise:', noise)
            if noise == 0.95:
                noise = 1.05
            elif noise == 1.05:
                noise = 0.95

        # Select action randomly or according to policy
        # if t < args.start_timesteps:
        #     action = env.action_space.sample()
        #     _, next_hidden = policy.select_action(np.array(state), hidden)
        # else:
        a, next_hidden = policy.select_action(np.array(state), hidden)
        action = (
            a + np.random.normal(
                0, max_action * args.expl_noise, size=action_dim)
        ).clip(-max_action, max_action)


        # Perform action
        next_state, reward, done, travelled, trackPos, new_distracted, agentaccel, agentsteer, driveraccel, driversteer, savedaction, PREVSTEER, PREVACCEL, PREV_AGENT_STEER, PREV_AGENT_ACCEL, info = env.step(action, DISTRACTED=DISTRACTED, noise=noise, episode_timesteps=episode_timesteps, PREVSTEER=PREVSTEER, PREVACCEL=PREVACCEL, PREV_AGENT_STEER=PREV_AGENT_STEER, PREV_AGENT_ACCEL=PREV_AGENT_ACCEL, prevstate=state)
        done_bool = float(
            done) if episode_timesteps < 1024 or travelled < 2000 else 0
        # print(next_state)
        if episode_timesteps > 1024 or travelled > 2000:
            done = True
        # Store data in replay buffer
        cost = info['cost']
        replay_buffer.add(
            state, savedaction, next_state, reward, cost, done_bool, hidden, next_hidden)
        state = next_state
        hidden = next_hidden
        episode_reward += reward
        total_abs_center += np.abs(trackPos)
        total_steer_agent += np.abs(agentsteer)
        total_steer_driver += np.abs(driversteer)
        TOTAL_COST += cost

        # Train agent after collecting sufficient data
        if (not policy.on_policy) and t >= args.start_timesteps:
            policy.train(replay_buffer, args.batch_size)
        elif policy.on_policy and t % n_update == 0:
            policy.train(replay_buffer)
            replay_buffer.clear_memory()
        results_step.write(
            "{0} {1} {2} {3} {4:.2f} {5:.2f} {6:.2f} {7:.2f} {8:.2f} {9:.2f} {10:.2f} {11}".format(episode_num, episode_timesteps, t,
                                                                             new_distracted, travelled, episode_reward,
                                                                             trackPos, agentaccel, agentsteer, driveraccel, driversteer, cost))
        results_step.write("\n")
        if new_distracted == False:
            input_distracted = 0
        if new_distracted == True:
            input_distracted = 1
        # inputs.write("{0:.2f} {1:.2f} {2:.2f} {3:.2f} {4}".format(state[0], state[1], state[2], state[3], input_distracted))
        # inputs.write("\n")
        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it
            #  will increment +1 even if done=True
            print(
                f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f} Distance: {travelled:.3f}")
            results_episode.write(
                "{0} {1} {2} {3:.2f} {4:.2f} {5:.2f} {6:.2f} {7:.2f} {8}".format(episode_num, episode_timesteps, t, total_abs_center,
                                                             travelled, episode_reward, total_steer_agent, total_steer_driver, TOTAL_COST))
            results_episode.write("\n")
            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
            total_abs_center = 0
            total_steer_agent = 0
            total_steer_driver = 0
            PREVSTEER = 0
            PREVACCEL = 0
            PREV_AGENT_STEER = 0
            PREV_AGENT_ACCEL = 0
            TOTAL_COST = 0
            hidden = policy.get_initial_states()
            if save == True:
                policy.save(f"./models/{file_name}")
        # Evaluate episode
        # if (episode_num + 1) % args.eval_freq == 0:
        #     total_reward, travelled, total_abs_center, t = eval_policy(policy, episode_num, DISTRACTED=DISTRACTED, noise=noise, episode_timesteps=episode_timesteps)
        #     eval_ep.write(
        #         "{0} {1} {2:.2f} {3:.2f} {4:.2f}".format(evalep, t, total_abs_center,
        #                                                      travelled, total_reward))
    if save == True:
        policy.save(f"./models/{file_name}")



if __name__ == "__main__":
    main()
