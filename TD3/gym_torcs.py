import gym
from gym import spaces
import numpy as np
import snakeoil3_gym as snakeoil3
import numpy as np
import copy
import collections as col
import os
import time
import xml.etree.ElementTree as ET
import random
import threading
from random import randint
from collections import deque

def clamp(num, min_value, max_value):
    return max(min(num, max_value), min_value)


def drive_example(c):
    '''This is only an example. It will get around the track but the
    correct thing to do is write your own `drive()` function.'''
    S, R = c.S.d, c.R.d
    target_speed = 100

    # Steer To Corner
    R['steer'] = S['angle'] * 10 / 3.14
    # Steer To Center
    R['steer'] -= S['trackPos']

    # Throttle Control
    if S['speedX'] < target_speed - (R['steer'] * 50):
        R['accel'] += .01
    else:
        R['accel'] -= .01
    if S['speedX'] < 10:
        R['accel'] += 1 / (S['speedX'] + .1)

    # Traction Control System
    if ((S['wheelSpinVel'][2] + S['wheelSpinVel'][3]) -
            (S['wheelSpinVel'][0] + S['wheelSpinVel'][1]) > 5):
        R['accel'] -= .2

    return R['steer'], R['accel']


class TorcsEnv:
    """
    Gym torcs environment.

    Start a Torcs process which wait for client(s) to
    connect through a socket connection. Environment sets
    the connection and communicates with the game whenever
    step, reset or constructor is called.

    Note: In order to change the track randomly
    at reset, you need to feed the path of the
    game's quickrace.xml file. If you installed the
    game as a root user it can be found at:
    "/usr/local/share/games/torcs/config/raceman"
    Also you may need to change the permissions
    for the file in order to modify it through
    the environment's reset method.


    Arguments:
        port: port the game will wait for connection
        path: path of the "quickrace.xml" file

    Methods:
        step: send the given action container(list, tuple
            or any other container). Action needs to have
            3 elements.
        reset: send a message to reset the game.

    """
    terminal_judge_start = 50  # Speed limit is applied after this step
    termination_limit_progress = 5  # [km/h], episode terminates if car is running slower than this limit
    initial_reset = True

    def __init__(self, port=3101, path=None):
        self.port = port
        self.initial_run = True
        # self.reset_torcs()

        if path:
            self.tree = ET.parse(path)
            self.root = self.tree.getroot()
            self.path = path

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
        high = np.concatenate([
            # np.array([1.0]),
            # np.ones(19),
            # np.array([1.0]),
            # np.array([1.0]),
            np.array([1.0]),
            np.array([1.0]),
            np.array([1.0]),
            np.array([1.0]),
            # np.array([1.0]),
            # np.array([1.0]),
            # np.ones(4),
            # np.array([1.0]),
            # np.array([1.0]),
        ])
        low = np.concatenate([
            # np.array([-1.0]),
            # np.ones(19) * -1 / 200,
            # np.array([-1.0]),
            # np.array([-1.0]),
            np.array([-1.0]),
            np.array([-1.0]),
            np.array([-1.0]),
            np.array([-1.0]),
            # np.array([-1.0]),
            # np.array([-1.0]),
            # np.zeros(4),
            # np.array([0.0]),
            # np.array([0.0]),
        ])
        self.observation_space = spaces.Box(low=low, high=high)

    def step(self, u, DISTRACTED, noise, episode_timesteps, PREVSTEER, PREVACCEL, PREV_AGENT_STEER, PREV_AGENT_ACCEL, prevstate):
        AgentInteract = True
        if episode_timesteps < 20:
            DISTRACTED = False
        # print('Distracted: ', DISTRACTED)
        # multiplier = np.clip(0.0125*episodenumber, 0, 0.5)
        # print('Multiplier: ', multiplier)
        # if randint(0, 40) < episodenumber:
        #     print('AgentActive')
        #     AgentInteract = True
        Driver = True
        Agent = True
        TeacherActive = False

        assert self.initial_reset == False, "Call the reset() function before step() function!"
        client = self.client
        prev_steer = PREVSTEER
        prev_accel = PREVACCEL

        obs = client.S.d
        obs['prev_driverSteer'] = prev_steer
        obs['prev_driverAccel'] = prev_accel

        obs['prev_agentSteer'] = PREV_AGENT_STEER
        obs['prev_agentAccel'] = PREV_AGENT_ACCEL

        this_action = self.agent_to_torcs(u)
        # Apply Action
        action_torcs = client.R.d
        # print('angle', client.S.d['angle'])
        # print('trackpos', client.S.d['trackPos'])
        if TeacherActive:
            if client.S.d['trackPos'] > 0.5 and client.S.d['angle'] < -0.2:
                print('Too low, Teacher Active')
                # np.clip(action_torcs['steer'], -1, 0)
                this_action['steer']  = -0.5
                print(this_action['steer'])
            if client.S.d['trackPos'] < -0.5 and client.S.d['angle'] > 0.2:
                print('Too high, Teacher Active')
                # np.clip(action_torcs['steer'], 0, 1)
                this_action['steer'] = 0.5
                print(this_action['steer'])

        # Combining the steering actions from the driver and the agent
        # this_action are the agent and steer and accel are the driver.
        # if Agent and not Driver:
        #     action_torcs["steer"] = (this_action["steer"])
        #     # action_torcs["brake"] = this_action["brake"]
        #     action_torcs["accel"] = (abs(this_action["accel"]))
        #
        # # Steering and acceleration input from the driver
        # if Driver and not Agent:
        #     if not DISTRACTED:
        #         steer, accel = drive_example(client)
        #     else:
        #         steer, accel = prev_steer * noise, prev_accel * 0.95
        #     action_torcs["steer"] = steer
        #     action_torcs["accel"] = accel
        #     savedaction = [steer, accel]
        # range noise randomly from 0.9 - 0.95
        # range noise randomly from 1.05 - 1.10
        # EXTRA NOISE
        if noise == 0.95:
            noise = random.uniform(0.85, 0.95)
        if noise == 1.05:
            noise = random.uniform(1.05, 1.15)
        # print('noise: ', noise)
        if Agent and Driver:
            if not DISTRACTED:
                steer, accel = drive_example(client)
            else:
                steer, accel = prev_steer * noise, prev_accel * 0.95

            action_torcs["steer"] = np.clip((this_action["steer"] + steer) / 2, -1, 1)
            action_torcs["accel"] = np.clip((abs(this_action["accel"]) + accel) / 2, 0, 1)

            # action_torcs["steer"] = np.clip((this_action["steer"]*multiplier + steer*(1-multiplier)), -1, 1)
            # action_torcs["brake"] = this_action["brake"]/2
            # action_torcs["accel"] = np.clip((abs(this_action["accel"]*multiplier) + accel*(1-multiplier)), 0, 1)

        savedaction = [this_action["steer"], this_action["accel"]]
        # print(savedaction)

        if client.S.d['speedX'] > 100:
            action_torcs["accel"] = 0
        action_torcs['gear'] = 1
        if client.S.d['speedX'] > 50:
            action_torcs['gear'] = 2
        if client.S.d['speedX'] > 80:
            action_torcs['gear'] = 3
        if client.S.d['speedX'] > 110:
            action_torcs['gear'] = 4
        if client.S.d['speedX'] > 140:
            action_torcs['gear'] = 5
        if client.S.d['speedX'] > 170:
            action_torcs['gear'] = 6

        # Save the privious full-obs from torcs for the reward calculation
        obs_pre = copy.deepcopy(client.S.d)

        # action_torcs['steer'] = clamp(action_torcs['steer'], -1, 1)
        # action_torcs['accel'] = clamp(action_torcs['accel'], 0, 1)
        # print(action_torcs)
        # One-Step Dynamics Update #################################
        # Apply the Agent's action into torcs
        client.respond_to_server()
        # Get the response of TORCS
        client.get_servers_input()

        # Get the current full-observation from torcs
        obs = client.S.d
        if Driver:
            obs['driverSteer'] = steer
            obs['driverAccel'] = accel

        obs['agentSteer'] = this_action["steer"]
        obs['agentAccel'] = this_action['accel']
        if DISTRACTED == True:
            obs['distracted'] = 1
        else:
            obs['distracted'] = 0

        # Make an obsevation from a raw observation vector from TORCS
        self.observation = self.make_observaton(obs)

        # print(self.observation)
        episode_terminate = False
        info = {}
        # Reward setting Here #######################################
        # direction-dependent positive reward
        track = np.array(obs['track'])
        sp = np.array(obs['speedX'])
        progress = sp * np.cos(obs['angle']) - np.abs(sp * np.sin(obs['angle'])) - sp * np.abs(
            obs['trackPos']) - 0.1 * sp * np.abs(obs['agentSteer'])
        reward = progress
        # REWARD_COMPARE = sp * np.cos(obs['angle'])

        # print('previous steer: ', obs['prev_driverSteer'])
        # print('current steer: ', obs['driverSteer'])
        # collision detection
        info['cost'] = 0
        if np.abs(obs['trackPos']) > 0.7:
            info['cost'] = 100
        # Termination settings here ##############################
        if obs['damage'] - obs_pre['damage'] > 0:
            info["collision"] = True
            reward -= 10
            episode_terminate = True

        # if self.terminal_judge_start < self.time_step: # Episode terminates if the progress of agent is small
        #    if abs(progress) < self.termination_limit_progress:
        #        if self.time_step > 20:
        #             reward -= 10
        #             # print("--- No progress restart : reward: {},x:{},angle:{},trackPos:{}".format(reward,sp,obs['angle'],obs['trackPos']))
        #             # print(self.time_step)
        #             episode_terminate = True
        #             info["no progress"] = True
        #             # client.R.d['meta'] = True

        if np.cos(obs['angle']) < 0:  # Episode is terminated if the agent runs backward
            if self.time_step > 20:
                reward -= 10
                # print(self.time_step)
                episode_terminate = True
                info["moving back"] = True
                # client.R.d['meta'] = True
        if self.terminal_judge_start < self.time_step:  # Episode terminates if the progress of agent is small
            if obs['speedX'] < self.termination_limit_progress:
                print("No progress")
                reward -= 10
                episode_terminate = True
                client.R.d['meta'] = True

        info["place"] = int(obs["racePos"])
        if episode_terminate is True:  # Send a reset signal
            # reward += (obs["racePos"] == 1)*20 # If terminated and first place
            self.initial_run = False
            # client.respond_to_server()
        PREVSTEER, PREVACCEL = steer, accel
        self.time_step += 1
        PREV_AGENT_STEER, PREV_AGENT_ACCEL = this_action['steer'], this_action['accel']
        # newstate = self.get_obs()
        # if randint(1, 2) == 1:
            # newstate = [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)]
            # newstate = self.get_obs()
            # newstate = prevstate
        # else:
        #     newstate = self.get_obs()
            # newstate = [0., 0., 0., 0.]
        newstate = self.get_obs()
        return newstate, reward, episode_terminate, obs['distRaced'], obs['trackPos'], DISTRACTED, this_action["accel"],this_action["steer"], accel, steer, savedaction, PREVSTEER, PREVACCEL, PREV_AGENT_STEER, PREV_AGENT_ACCEL, info

    def reset(self, relaunch=False, sampletrack=False, render=False):
        """ Reset the environment
            Arguments:
                - relaunch: Relaunch the game. Necessary to call with
                    from time to time because of the memory leak
                sampletrack: Sample a random track and load the game
                    with it at the relaunch. Relaunch needs to be
                    true in order to modify the track!
                render: Change the mode. If true, game will be launch
                    in "render" mode else with "results only" mode.
                    Relaunch needs to be true in order to modify the track!
        """
        self.time_step = 0
        #
        # if relaunch:
        #     if sampletrack:
        #         try:
        #             sample_track(self.root)
        #         except AttributeError:
        #             pass
        #     try:
        #         set_render_mode(self.root, render=render)
        #     except AttributeError:
        #             pass
        #     self.tree.write(self.path)
        #     time.sleep(0.5)

        if self.initial_reset is not True:
            self.client.R.d['meta'] = True
            self.client.respond_to_server()

            ## TENTATIVE. Restarting TORCS every episode suffers the memory leak bug!
            if relaunch is True:
                self.reset_torcs()
                # print("### TORCS is RELAUNCHED ###")

        # Modify here if you use multiple tracks in the environment
        self.client = snakeoil3.Client(p=self.port, vision=False)  # Open new UDP in vtorcs
        self.client.MAX_STEPS = np.inf

        client = self.client
        client.get_servers_input()  # Get the initial input from torcs
        self.client.MAX_STEPS = np.inf

        obs = client.S.d  # Get the current full-observation from torcs
        obs['driverSteer'] = 0
        obs['driverAccel'] = 0
        obs['prev_driverSteer'] = 0
        obs['prev_driverAccel'] = 0
        obs['agentSteer'] = 0
        obs['agentAccel'] = 0
        obs['prev_agentSteer'] = 0
        obs['prev_agentAccel'] = 0
        obs['distracted'] = 0
        self.observation = self.make_observaton(obs)

        self.initial_reset = False

        return self.get_obs()

    def kill(self):
        os.system('pkill torcs')

    def close(self):
        self.client.R.d['meta'] = True
        self.client.respond_to_server()

    def get_obs(self):
        return self.observation

    def reset_torcs(self, port=3101):
        # print("relaunch torcs")
        os.system('pkill torcs')
        time.sleep(0.5)
        os.system('torcs -nofuel -nodamage -nolaptime -p 3101 &')
        time.sleep(2)
        os.system('sh autostart.sh')
        time.sleep(2)

    def agent_to_torcs(self, u):
        torcs_action = {'steer': u[0]}
        torcs_action.update({'accel': u[1]})
        # torcs_action.update({'brake': (u[2])/2})
        return torcs_action

    def make_observaton(self, raw_obs):
        return np.concatenate(
            [
             np.array(raw_obs["angle"], dtype=np.float32).reshape(1) / np.pi * 2,
             # np.array(raw_obs["track"], dtype=np.float32) / 100,
             np.array(raw_obs["trackPos"], dtype=np.float32).reshape(1) / 2,
             # np.array(raw_obs["speedX"], dtype=np.float32).reshape(1) / 200,
             # np.array(raw_obs["prev_agentSteer"], dtype=np.float32).reshape(1),
             # np.array(raw_obs["prev_agentAccel"], dtype=np.float32).reshape(1),
             np.array(raw_obs["prev_driverSteer"], dtype=np.float32).reshape(1),
             np.array(raw_obs["prev_driverAccel"], dtype=np.float32).reshape(1),
             # np.array(raw_obs["driverSteer"], dtype=np.float32).reshape(1),
             # np.array(raw_obs["driverAccel"], dtype=np.float32).reshape(1),
             # np.array(raw_obs["agentSteer"], dtype=np.float32).reshape(1),
             # np.array(raw_obs["agentAccel"], dtype=np.float32).reshape(1),
             # np.array(raw_obs["speedZ"], dtype=np.float32).reshape(1) / 200,
             # np.array(raw_obs["speedY"], dtype=np.float32).reshape(1) / 200,
             # np.array(raw_obs["wheelSpinVel"], dtype=np.float32) / 200,
             # np.array(raw_obs["rpm"], dtype=np.float32).reshape(1) / 5000,
             # np.array(raw_obs['distracted'], dtype=np.float32).reshape(1)
            ]
        )

    def __del__(self):
        self.kill()
