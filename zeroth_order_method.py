from evaluation_policy import evaluate_policy
import torch

def zeroth_order_method(env, policy_net, num_iterations=100, initial_learning_rate=0.01, log_file="zeroth_order_log.txt"):
    best_score = float("-inf")
    best_params = [param.clone() for param in policy_net.parameters()]
    learning_rate = initial_learning_rate
    decay_rate = 0.99
    # Open a file for logging returns
    with open(log_file, "w") as f:
        episode_count = 0  # Track the total number of episodes

        for iteration in range(num_iterations):
            # Generate perturbation
            perturbation = [torch.randn_like(param) for param in policy_net.parameters()]
            perturbed_params_plus = perturbation
            perturbed_params_minus = [param - perturbation[i] for i, param in enumerate(policy_net.parameters())]

            score_plus = evaluate_policy(env, policy_net, perturbed_params_plus)
            score_minus = evaluate_policy(env, policy_net, perturbed_params_minus)

            perturbation_norm = sum((p ** 2).sum() for p in perturbation).sqrt()
            gradient_estimate = [0.5 * (score_plus - score_minus) * perturbation[i] / perturbation_norm for i in range(len(perturbed_params_plus))]
            
            with torch.no_grad():
                for param, grad in zip(policy_net.parameters(), gradient_estimate):
                    param += learning_rate * grad

            current_score = max(score_plus, score_minus)
            if current_score > best_score:
                best_score = current_score
                best_params = [param.clone() for param in policy_net.parameters()]
            
            learning_rate *= decay_rate

            # Evaluate and log the return using the best parameters found so far
            episode_return = evaluate_policy(env, policy_net, best_params)
            f.write(f"RETURN {episode_count} {episode_return}\n")
            episode_count += 1
            print(f"Iteration {iteration}, Best Score: {best_score}, Episode Return: {episode_return}")