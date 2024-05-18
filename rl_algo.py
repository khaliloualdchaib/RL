import gym
import torch
import numpy as np
from policy_network import PolicyNetwork

def evaluate_policy(env, policy_net, parameters, episodes=5):
    total_return = 0
    for _ in range(episodes):
        state = env.reset()
        observation = state[0]
        observation = torch.from_numpy(observation).float()
        done = False
        while not done:
            with torch.no_grad():
                # Manually set the parameters for the policy network
                for param, perturbed_param in zip(policy_net.parameters(), parameters):
                    param.copy_(perturbed_param)
                action = policy_net(observation)
                action = action.detach().numpy()
                next_state, reward, done, _, _ = env.step(action)
                total_return += reward
                observation = torch.from_numpy(next_state).float()
    avg_return = total_return / episodes
    return avg_return

def zeroth_order_method(env, policy_net, num_iterations=100, learning_rate=0.01, episodes_per_eval=5):
    best_score = float("-inf")
    
    for iteration in range(num_iterations):
        # Generate perturbation
        perturbation = [torch.randn_like(param) for param in policy_net.parameters()]
        perturbed_params_plus = [param + perturbation[i] for i, param in enumerate(policy_net.parameters())]
        perturbed_params_minus = [param - perturbation[i] for i, param in enumerate(policy_net.parameters())]
        
        score_plus = evaluate_policy(env, policy_net, perturbed_params_plus, episodes_per_eval)
        score_minus = evaluate_policy(env, policy_net, perturbed_params_minus, episodes_per_eval)
        
        gradient_estimate = [0.5 * (score_plus - score_minus) * perturbation[i] for i in range(len(perturbed_params_plus))]
        
        with torch.no_grad():
            for param, grad in zip(policy_net.parameters(), gradient_estimate):
                param += learning_rate * grad


        current_score = max(score_plus, score_minus)
        if current_score > best_score:
            best_score = current_score
        print(f"Iteration {iteration}, Best Score: {best_score}")
