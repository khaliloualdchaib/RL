from policy_network import PolicyNetwork
import gym

#https://elegantrl.readthedocs.io/en/latest/tutorial/LunarLanderContinuous-v2.html

env = gym.make("LunarLander-v2", render_mode="human")
env.action_space.seed(42)

observation, info = env.reset(seed=42)

for _ in range(1000):
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())

    if terminated or truncated:
        observation, info = env.reset()

env.close()


# # Define the dimensions of the input and output
# input_dim = 8  # Dimensionality of the state space in LunarLanderContinuous
# hidden_dim = 128  # Number of neurons in the hidden layer
# output_dim = 4  # Dimensionality of the action space in LunarLanderContinuous

# # Create an instance of the PolicyNetwork
# policy_net = PolicyNetwork(input_dim, hidden_dim, output_dim)





# # Create a dummy input tensor (you can replace this with actual data)
# dummy_input = torch.randn(1, input_dim)  # Dummy input with shape (1, input_dim)

# # Pass the dummy input through the network
# output = policy_net(dummy_input)

# # Print the output
# print("Output:", output)
