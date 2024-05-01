from policy_network import PolicyNetwork
import gym
import torch
import numpy as np

env = gym.make("LunarLander-v2", continuous=True)

# Define the dimensions of the input and output
input_dim = 8  # Dimensionality of the state space in LunarLanderContinuous
hidden_dim = 128  # Number of neurons in the hidden layer
output_dim = 4  # Dimensionality of the action space in LunarLanderContinuous

# Create an instance of the PolicyNetwork
policy_net = PolicyNetwork(input_dim, hidden_dim, output_dim)

observation = env.reset()
observation_numpy = observation[0]
observation_tensor = torch.from_numpy(observation_numpy)
action_probs = policy_net(observation_tensor)
print(action_probs)
