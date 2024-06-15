import torch


def evaluate_params(env, policy_net, params, num_episodes=10, discount_factor=0.99):
    episode_rewards = []
    original_params = [param.clone() for param in policy_net.parameters()]
    with torch.no_grad():
        for param, tensor in zip(policy_net.parameters(), params):
            param.data = tensor
    for _ in range(num_episodes):
        begin_state = env.reset()
        current_state = begin_state[0]
        done = False
        total_rewards = 0
        step = 0
        discount = 1.0

        while not done:
            step += 1
            observation = torch.tensor(current_state, dtype=torch.float32)
            observation = torch.unsqueeze(observation, 0)  # Add a batch dimension
            action_tensor = policy_net(observation)
            action = action_tensor.detach().numpy().squeeze()
            next_state, reward, done, _, _ = env.step(action)
            current_state = next_state
            total_rewards+=reward * discount
            discount *= discount_factor
            if step == 1000:
                break
        episode_rewards.append(total_rewards)

    with torch.no_grad():
        for param, tensor in zip(policy_net.parameters(), original_params):
            param.data = tensor
    return episode_rewards

