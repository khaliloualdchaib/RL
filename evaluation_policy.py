import torch
import numpy as np
import sys

def evaluate_policy(env, policy_net, params, num_episodes=10):
    total_return = 0.0
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        step = 0
        current_state = state[0]
        while not done:
            step += 1
            observation = torch.tensor(current_state, dtype=torch.float32)
            observation = torch.unsqueeze(observation, 0)  # Add a batch dimension
            with torch.no_grad():
                # Set policy network parameters to perturbed parameters
                for param, perturbed_param in zip(policy_net.parameters(), params):
                    param.copy_(perturbed_param)
                    
                # Get action from the policy network
                action_tensor = policy_net(observation)
                action = action_tensor.detach().numpy().squeeze()
            
            # Take action in the environment
            next_state, reward, done, _, _ = env.step(action)
            total_return += reward
            current_state = next_state
            if step == 1000:
                break
    return total_return / num_episodes
