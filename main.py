from policy_network import PolicyNetwork
import gym
from zeroth_order_method import zeroth_order_method
#from population_method import population_method

env = gym.make("LunarLander-v2", continuous=True)

input_dim = 8  # Dimensionality of the state space in LunarLanderContinuous
hidden_dim = 128  # Number of neurons in the hidden layer
output_dim = 2  # Dimensionality of the action space in LunarLanderContinuous

policy_net = PolicyNetwork(input_dim, hidden_dim, output_dim)



zeroth_order_method(env, 
                    policy_net, 
                    num_iterations=1000,  
                    learning_rate=0.001, 
                    log_file="zeroth_order_log.txt",
                    decay_rate=0.999,
                    num_episodes_per_eval=10,
                    discount_factor=0.999)