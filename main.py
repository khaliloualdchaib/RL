from policy_network import PolicyNetwork
import gym
import torch
from rl_algo import zeroth_order_method
env = gym.make("LunarLander-v2", continuous=True)
action_space = env.action_space

# Define the dimensions of the input and output
input_dim = 8  # Dimensionality of the state space in LunarLanderContinuous
hidden_dim = 128  # Number of neurons in the hidden layer
output_dim = 2  # Dimensionality of the action space in LunarLanderContinuous

# Create an instance of the PolicyNetwork
policy_net = PolicyNetwork(input_dim, hidden_dim, output_dim)
trained_policy = zeroth_order_method(env, policy_net)

