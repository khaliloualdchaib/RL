from evaluation_policy import evaluate_policy
import torch


def zeroth_order_method(env, policy_net, num_iterations=100, learning_rate=0.01, episodes_per_eval=10):
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