import gym
from pysc2_envs.envs import DZBEnv
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from absl import flags
import os
import matplotlib.pyplot as plt
import numpy as np

# Make sure that the mini-game environment is working properly
from stable_baselines.common.env_checker import check_env

FLAGS = flags.FLAGS
FLAGS([''])

# create vectorized environment
env = gym.make('defeat-zerglings-banelings-v0')
eng = DZBEnv()
#env = DummyVecEnv([lambda: DZBEnv()])

# use ppo2 to learn and save the model when finished
model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log="log/")
model.learn(total_timesteps=int(10), tb_log_name="first_run", reset_num_timesteps=False)

# Save model
if not os.path.exists("./model"):
    os.makedirs("./model")
model.save("model/dbz_ppo")

# Test model
env = gym.make('defeat-zerglings-banelings-v0')
obs, _, _, _ = env.reset()

done = False
while not done:
    action, _state = model.predict(np.array(obs).flatten().reshape(1, 7056))
    obs, reward, done, info = env.step(action)
    env.render()
    print(action, reward)

# How to view tensorboard
# tensorboard --logdir=log/first_run_1/

