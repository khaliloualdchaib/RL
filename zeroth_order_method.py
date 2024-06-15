from evaluation_params import evaluate_params
from evaluate_policy import evaluate_policy
import torch

def zeroth_order_method(env, policy_net, num_iterations=1000, learning_rate=0.00001, log_file="zeroth_order_log.txt", decay_rate=0.99, num_episodes_per_eval=20, discount_factor=0.99):
    with open(log_file, "w") as f:
        episode = 0
        for i in range(num_iterations):
            perturbation = [torch.randn_like(param) for param in policy_net.parameters()]
            theta_plus = [param + pert for param, pert in zip(policy_net.parameters(), perturbation)]
            theta_min = [param - pert for param, pert in zip(policy_net.parameters(), perturbation)]
            theta_plus_score = sum(evaluate_params(env=env, policy_net=policy_net, params=theta_plus, num_episodes=num_episodes_per_eval, discount_factor=discount_factor))/num_episodes_per_eval
            theta_min_score = sum(evaluate_params(env=env, policy_net=policy_net, params=theta_min, num_episodes=num_episodes_per_eval, discount_factor=discount_factor))/num_episodes_per_eval
            
            gradient = [0.5*(theta_plus_score - theta_min_score) * plus for plus in theta_plus]
            with torch.no_grad():
                for param, grad in zip(policy_net.parameters(), gradient):
                    param.data += (learning_rate * grad)
            learning_rate *= decay_rate
            evaluation_rewards = evaluate_policy(env=env, policy_net=policy_net, num_episodes=num_episodes_per_eval)
            for reward in evaluation_rewards:
                episode += 1
                f.write(f"RETURN {episode} {reward}\n")
                print(f"RETURN {episode}: {reward}")