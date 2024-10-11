import gym
import gym_maze
import numpy as np
import random
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

import seaborn as sns
sns.set()

env = gym.make("Taxi-v3")
state_n = 500
action_n = 6


class CrossEntropyStochasticAgent():
    def __init__(self, state_n, action_n):
        self.state_n = state_n
        self.action_n = action_n

        #initialization with equal probabilities
        self.model = np.ones((self.state_n, self.action_n)) / self.action_n
    
    def get_det_policy(self):
        det_policy = np.zeros((self.state_n, self.action_n))
        for state in range(self.state_n):
            action = np.random.choice(np.arange(self.action_n), p=self.model[state])
            det_policy[state, action] = 1
        
        return det_policy

    def get_action(self, state, det_policy=None):
        if det_policy is None:
            action = np.random.choice(np.arange(self.action_n), p=self.model[state])
            return int(action)

        action = np.where(det_policy[state] == 1)[0]
        return int(action)


    def fit(self, elite_trajectories):
        new_model = np.zeros((self.state_n, self.action_n))
        for trajectory in elite_trajectories:
            for state, action in zip(trajectory['states'], trajectory['actions']):
                new_model[state][action] += 1
            
        for state in range(self.state_n):
            if np.sum(new_model[state]) > 0:
                new_model[state] /= np.sum(new_model[state])
            else:
                new_model[state] = self.model[state].copy()

        self.model = new_model
        return None


def get_state(obs):
    return obs


def get_trajectory(env, agent, det_policy=None, max_len=1000, visualize=False):
    trajectory = {'states': [], 'actions': [], 'rewards': []}

    obs = env.reset()
    state = get_state(obs)

    for _ in range(max_len):
        trajectory['states'].append(state)

        action = agent.get_action(state, det_policy)
        trajectory['actions'].append(action)

        obs, reward, done, _ = env.step(action)
        trajectory['rewards'].append(reward)

        state = get_state(obs)

        if visualize:
            time.sleep(0.01)
            env.render()

        if done:
            break

    return trajectory


def train(env, agent, q_param, iteration_n, trajectory_n, policies_n):
    rewards = []
    for iteration in tqdm(range(iteration_n)):

        #policy evaluation
        det_policies = [agent.get_det_policy() for _ in range(policies_n)]
        trajectories = [[get_trajectory(env, agent, det_policies[j]) for _ in range(trajectory_n)] for j in range(policies_n)]
        trajectory_rewards = [[np.sum(trajectories[j][i]['rewards']) for i in range(trajectory_n)] for j in range(policies_n)]
        policy_rewards = [np.mean(trajectory_rewards[j]) for j in range(policies_n)]
        mean_total_reward = np.mean(policy_rewards)
        rewards.append(mean_total_reward)
        print('iteration:', iteration, 'policy rewards:', mean_total_reward)

        #policy improvement
        quantile = np.quantile(policy_rewards, q_param)
        elite_trajectories = []
        for index, policy_reward in enumerate(policy_rewards):
            if policy_reward > quantile:
                elite_trajectories.append(trajectories[index])

        elite_trajectories = [item for sublist in elite_trajectories for item in sublist]

        agent.fit(elite_trajectories)

    return rewards


agent = CrossEntropyStochasticAgent(state_n, action_n)

q_param = 0.5
iteration_n = 30
trajectory_n = 50

policies_n = 500

rewards = train(env, agent, q_param, iteration_n, trajectory_n, policies_n)

fig, axs = plt.subplots(figsize=(14, 8))

axs.set_xlabel('Number of Iterations')
axs.set_ylabel('Mean Total Rewards')

labels = 'policies_n = ' + f'{policies_n}' + ', trajectory_n = ' + f'{trajectory_n}'

axs.plot(rewards, label=labels)

plt.legend()
plt.savefig('picture3.png')
plt.show()



