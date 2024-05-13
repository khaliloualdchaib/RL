import gym
import torch
import numpy as np

def evaluate_policy(env, policy_net, num_episodes=1):
    total_reward = 0
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            state_tensor = torch.FloatTensor(state)
            action_probs = policy_net(state_tensor)
            action = action_probs.detach().numpy()
            state, reward, done, _ = env.step(action)
            total_reward += reward
    return total_reward / num_episodes

def population_method(env, policy_net, num_episodes=100, num_perturbations=10, learning_rate=0.001):
    pass

def zeroth_order_method(env, policy_net, num_episodes=100, num_perturbations=10, learning_rate=0.001):
    pass

