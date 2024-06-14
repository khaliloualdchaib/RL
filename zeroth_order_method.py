from evaluation_policy import evaluate_policy
import torch
from tqdm import tqdm

def zeroth_order_method(env, policy_net, num_iterations=1000, learning_rate=0.00001, log_file="zeroth_order_log.txt", decay_rate=0.99, num_episodes_per_eval=20):
    with open(log_file, "w") as f:
        episode = 0
        for i in range(num_iterations):
            perturbation = [torch.randn_like(param) for param in policy_net.parameters()]
            theta_plus = [param + pert for param, pert in zip(policy_net.parameters(), perturbation)]
            theta_min = [param - pert for param, pert in zip(policy_net.parameters(), perturbation)]
            theta_plus_score = sum(evaluate_policy(env=env, policy_net=policy_net, params=theta_plus, num_episodes=num_episodes_per_eval))/num_episodes_per_eval
            theta_min_score = sum(evaluate_policy(env=env, policy_net=policy_net, params=theta_min, num_episodes=num_episodes_per_eval))/num_episodes_per_eval
            #gradient = [(0.5 * (theta_plus_score - theta_min_score) * pert)/torch.norm(pert) for pert in theta_plus]
            gradient = [0.5*(theta_plus_score - theta_min_score) * plus for plus in theta_plus]
            for param, grad in zip(policy_net.parameters(), gradient):
                param.data += (learning_rate * grad)
            learning_rate *= decay_rate
            current_params = None
            if theta_min_score > theta_plus_score:
                current_params = theta_min
            else:
                current_params = theta_plus
            evaluation_rewards = evaluate_policy(env=env, policy_net=policy_net, params=current_params, num_episodes=num_episodes_per_eval)
            for reward in evaluation_rewards:
                episode += 1
                f.write(f"RETURN {episode} {reward}\n")
                print(f"RETURN {episode}: {reward}")