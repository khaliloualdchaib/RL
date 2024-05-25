import torch
from evaluation_policy import evaluate_policy

def population_method(env, policy_net, N, num_iterations=100, episodes_per_eval=5):
    best_score = float("-inf")
    best_params = None
    
    for iteration in range(num_iterations):
        perturbed_params_list = []
        for i in range(N):
            perturbed_params = [torch.randn_like(param) for param in policy_net.parameters()]
            perturbed_params_list.append(perturbed_params)
        
        for perturbed_params in perturbed_params_list:
            score = evaluate_policy(env, policy_net, perturbed_params, episodes_per_eval)
            if score > best_score:
                best_score = score
                best_params = perturbed_params

        with torch.no_grad():
            for param, new_param in zip(policy_net.parameters(), best_params):
                param.copy_(new_param)
        print(f"Iteration {iteration}, Best Score: {best_score}")
    return best_score
        

