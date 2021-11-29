import gym
from pysc2_envs.envs import DZBEnv
#from stable_baselines3.common.policies import MlpPolicy, CnnPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import A2C
from absl import flags
import os
import matplotlib.pyplot as plt
import numpy as np
import torch as th
import torch.nn as nn

# Make sure that the mini-game environment is working properly
from stable_baselines.common.env_checker import check_env

FLAGS = flags.FLAGS
FLAGS([''])

# create vectorized environment
env = gym.make('defeat-zerglings-banelings-v0')
check_env(env)
eng = DZBEnv()
#env = DummyVecEnv([lambda: DZBEnv()])

# use ppo2 to learn and save the model when finished
# https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 1, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=32),
)

model = A2C('CnnPolicy', env, verbose=0, tensorboard_log="log/", n_steps=50, learning_rate=0.001, policy_kwargs=policy_kwargs)
model.learn(total_timesteps=int(1000), tb_log_name="first_run")
print("Training finished")


# Save model
if not os.path.exists("./model"):
    os.makedirs("./model")
model.save("model/dbz_ppo")

# Test model for 1 episode
model.load("model/dbz_ppo")
env = gym.make('defeat-zerglings-banelings-v0')
obs = env.reset()
done = False

total_reward = 0
while not done:
    action, _state = model.predict(np.array(obs))
    obs, reward, done, info = env.step(action)
    total_reward += reward
    env.render()
    print(action, reward)
print(total_reward)

# How to view tensorboard
# tensorboard --logdir=log/first_run_1/

