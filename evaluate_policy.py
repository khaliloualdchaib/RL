import torch

def evaluate_policy(env, policy_net, num_episodes):
    episode_rewards = []
    for _ in range(num_episodes):
        begin_state = env.reset()
        current_state = begin_state[0]
        done = False
        total_rewards = 0
        step = 0
        while not done:
            step+=1
            observation = torch.tensor(current_state, dtype=torch.float32)
            observation = torch.unsqueeze(observation, 0)  # Add a batch dimension
            action_tensor = policy_net(observation)
            action = action_tensor.detach().numpy().squeeze()
            next_state, reward, done, _, _ = env.step(action)
            current_state = next_state
            total_rewards+=reward
            if step == 1000:
                break
        episode_rewards.append(total_rewards)

    return episode_rewards