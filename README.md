# Master-Thesis
Code used for Master Thesis: Distracted Driver Detection: A Safer Reinforcement Learning Approach.

____________________________
                    
README
____________________________

STRUCTURE

1. Contents
2. How to install TORCS
3. Component overview
4. How to run experiment

____________________________

1. Contents

".thesis"                            - The thesis describes the key idea of the provided algorithm. No need to read the technical design of the algorithm.
"./TD3"                              - The environment and experimental setup used for the TD3 agent.
"./FOCOPS"                           - The environment and experimental setup used for the FOCOPS agent.
"./FOCOPS_WORKS_LSTM"                - The environment and experimental setup used for the FOCOPS agent with a LSTM layer.
"./FLICKER"                          - The environment and experimental setup for TD3 with flicker conditions.

None of the contained source files may be publicly published or provided to anyone not affiliated with Utrecht University and this project. They are intended to remain private.

____________________________

2. How to install TORCS

This installation guide only works on an Ubuntu Linux system. Our agent has not been tested on a Windows system, although it technically should be compatible. See here for a Windows guide: "http://torcs.sourceforge.net/index.php?name=sections&op=viewarticle&artid=3".

We have to install dependencies and set environment variables before we can install and run TORCS itself.

> Dependencies

"sudo apt-get update"
"sudo apt-get install screen libglib2.0-dev  libgl1-mesa-dev libglu1-mesa-dev  freeglut3-dev  libplib-dev  libopenal-dev libalut-dev libxi-dev libxmu-dev libxrender-dev  libxrandr-dev libpng-dev libvorbis-dev"

For other linux distributions than Ubuntu, the dependencies may have to be compiled from source. See here for a guide: "http://www.berniw.org/tutorials/robot/torcs/install/software-requirements.html".

Download and install boost 1.76.0 from "https://www.boost.org/users/history/version_1_76_0.html" and do a text search for "/Somewhere/boost_1_76_0". Replace the "/Somewhere" with your boost install path.

Do a text search for "/Somewhere/Torcs". You need to replace the "/Somewhere/" with the path of this directory on your system.

> Environment Variables

"export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib"
"export TORCS_BASE=***BASE DIRECTORY***/Torcs"
"export MAKE_DEFAULT=$TORCS_BASE/Make-default.mk"

Or put them in ~/.bashrc for convenience and reload the terminal.

> Installation

1. "cd ./Torcs"
2. "./configure --prefix=***BASE DIRECTORY***/Torcs/build/"
3. "make"
4. "make install"
5. "make datainstall"

If everything completes without errors, the installation is complete.

The python wrapper of TORCS for RL experiments Gym-TORCS was used which can be downloaded from: https://github.com/ugo-nama-kun/gym_torcs

____________________________


Main files:

TD3 and FOCOPS:

> "snakeoil3_gym.py"
Code for interfacing with a TORCS environment. Does not need to be changed.

> "autostart.sh"
File used to automatically restart TORCS in case of memory leakage.

> "gym_torcs.py"
Settings for the environment that can be adapted depending on experiment.
>   Size and values of action  space and observation space can be changed in the TorcsEnv class.
>   The step function is the most important. Here the reward function is set, steering and acceleration actions are set and we check if the experiment should reset. At the end we return the values we want for data.
>   In the make_observation function we can select the observations we want our agent to gather.

TD3:

> "main.py"
Main script that needs to be ran for the experiments.
>   In the change_variable function we can change when and how long the driver is distracted.
>   In the main function we can change which algorithm we use and other hyperparmaters as well as noise values
>   If we want to use a LSTM layer, change LSTM to True at the top

> "./utils/memory"
Replay buffer used for TD3. If we want to use a LSTM layer, change LSTM to True at top.

> ".algos/TD3"
Main TD3 algortihm script. If we want to use a LSTM layer, change LSTM to True at the top. Other hyperparameters can be changed in the TD3 class and if other activation functions want to be used, change these in the critic and actor forward functions. PPO and DDPG can be used similarly.

> Flicker:
The only change to the flicker condition is the percentage change of receiving 0 inputs instead of normal data. This value cn be changed in the gym_torcs.py file at the end of the step function. Either 'newstate' is the actual observations, or an array of 0's. 

FOCOPS: 
For FOCOPS the recurrent algorithm and the non-recurrent one are separated.
> "models.py"
Main FOCOPS algorithm script. Here the neural network used for the agent is created as well as action selection. Hidden dimensions can be changed as well as activation functions.

> "focops_main.py"
The main script for experiments. In the main function, the hyperparameters can be changed and in the train function we get the result of the data generators, which enables us to select the data we want to save. The script handles all the updates to the agent.

> "environment.py"
Here we can set the cost threshold for our experiments. We can change it beneath the 'constraint = distance' line.

> "data_generator.py"
Similar to the main.py for TD3, we can change the distractions of the driver, noise and the returns for each step.
____________________________
5. How to run experiment

1. Start Ubuntu and navigate to the directory where either main.py or focops_main.py are.
2. Type in 'python3 main.py' or 'python3 focops_main.py' for either TD3 or FOCOPS respectively.
3. TORCS should start up automatically.
4. Navigate toward Practise Mode and select the track.
5. Experiment should run until completion.
____________________________
