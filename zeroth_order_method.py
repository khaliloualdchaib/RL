from evaluation_policy import evaluate_policy
import torch

def zeroth_order_method(env, policy_net, num_iterations=100, learning_rate=0.01, log_file="zeroth_order_log.txt"):
    best_score = float("-inf")
    best_params = [param.clone() for param in policy_net.parameters()]

    # Open a file for logging returns
    with open(log_file, "w") as f:
        episode_count = 0  # Track the total number of episodes

        for iteration in range(num_iterations):
            # Generate perturbation
            perturbation = [torch.randn_like(param) for param in policy_net.parameters()]
            perturbed_params_plus = [param + perturbation[i] for i, param in enumerate(policy_net.parameters())]
            perturbed_params_minus = [param - perturbation[i] for i, param in enumerate(policy_net.parameters())]

            score_plus = evaluate_policy(env, policy_net, perturbed_params_plus)
            score_minus = evaluate_policy(env, policy_net, perturbed_params_minus)

            gradient_estimate = [0.5 * (score_plus - score_minus) * perturbation[i] for i in range(len(perturbed_params_plus))]

            with torch.no_grad():
                for param, grad in zip(policy_net.parameters(), gradient_estimate):
                    param += learning_rate * grad

            current_score = max(score_plus, score_minus)
            if current_score > best_score:
                best_score = current_score
                best_params = [param.clone() for param in policy_net.parameters()]

            # Evaluate and log the return using the best parameters found so far
            episode_return = evaluate_policy(env, policy_net, best_params)
            f.write(f"RETURN {episode_count} {episode_return}\n")
            episode_count += 1

            print(f"Iteration {iteration}, Best Score: {best_score}, Episode Return: {episode_return}")