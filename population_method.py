import torch
from evaluation_params import evaluate_params
from evaluate_policy import evaluate_policy

def population_method(env, policy_net, number_pertubations, num_iterations=100, episodes_per_eval=5, log_file="zeroth_order_log.txt", discount_factor=0.99):
    best_score = float("-inf")
    best_params = None
    with open(log_file, "w") as f:
        episode = 0
        for iteration in range(num_iterations):
            perturbed_params_list = []
            for i in range(number_pertubations):
                perturbed_params = [param + 0.1 * torch.randn_like(param) for param in policy_net.parameters()]
                perturbed_params_list.append(perturbed_params)
            
            for perturbed_params in perturbed_params_list:
                score = sum(evaluate_params(env=env, policy_net=policy_net, params=perturbed_params, num_episodes=episodes_per_eval, discount_factor=discount_factor))/episodes_per_eval
                if score > best_score:
                    best_score = score
                    best_params = perturbed_params

            with torch.no_grad():
                for param, new_param in zip(policy_net.parameters(), best_params):
                    param.copy_(new_param)
            evaluation_rewards = evaluate_policy(env=env, policy_net=policy_net, num_episodes=1)
            for reward in evaluation_rewards:
                episode += 1
                f.write(f"RETURN {episode} {reward}\n")
                print(f"RETURN {episode}: {reward}")
        