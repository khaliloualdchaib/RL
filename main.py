from policy_network import PolicyNetwork
import gym
from zeroth_order_method import zeroth_order_method
#from population_method import population_method

env = gym.make("LunarLander-v2", continuous=True)

# Define the dimensions of the input and output
input_dim = 8  # Dimensionality of the state space in LunarLanderContinuous
hidden_dim = 128  # Number of neurons in the hidden layer
output_dim = 2  # Dimensionality of the action space in LunarLanderContinuous

# Create an instance of the PolicyNetwork
policy_net = PolicyNetwork(input_dim, hidden_dim, output_dim)

# Run the zeroth-order method with logging
zeroth_order_method(env, policy_net, num_iterations=200, initial_learning_rate=0.0001, log_file="zeroth_order_log.txt")
