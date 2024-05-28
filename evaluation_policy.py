import torch

def evaluate_policy(env, policy_net, parameters):
    state = env.reset()
    observation = state[0]
    observation = torch.from_numpy(observation).float()
    done = False
    r = 0
    while not done:
        with torch.no_grad():
            # Manually set the parameters for the policy network
            for param, perturbed_param in zip(policy_net.parameters(), parameters):
                param.copy_(perturbed_param)
            action = policy_net(observation)
            action = action.detach().numpy()
            next_state, reward, done, _, _ = env.step(action)
            observation = torch.from_numpy(next_state).float()
            r += reward
    return r