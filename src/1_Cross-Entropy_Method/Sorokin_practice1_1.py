import gymnasium as gym
import gym_maze
import numpy as np
import random
import time
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
sns.set()

env = gym.make("Taxi-v3")
state_n = 500
action_n = 6


class CrossEntropyAgent():
    def __init__(self, state_n, action_n):
        self.state_n = state_n
        self.action_n = action_n

        #initialization with equal probabilities
        self.model = np.ones((self.state_n, self.action_n)) / self.action_n

    def get_action(self, state):
        action = np.random.choice(np.arange(self.action_n), p=self.model[state])
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


def get_trajectory(env, agent, max_len=1000, visualize=False):
    trajectory = {'states': [], 'actions': [], 'rewards': []}

    obs, _ = env.reset()
    state = get_state(obs)

    for _ in range(max_len):
        trajectory['states'].append(state)

        action = agent.get_action(state)
        trajectory['actions'].append(action)

        obs, reward, terminated, truncated, _ = env.step(action)
        trajectory['rewards'].append(reward)

        state = get_state(obs)

        if visualize:
            time.sleep(0.01)
            env.render()

        if terminated or truncated:
            break

    return trajectory


def train(env, agent, q_param, iteration_n, trajectory_n):
    mean_rewards = []
    for iteration in range(iteration_n):

        # policy evaluation
        trajectories = [get_trajectory(env, agent) for _ in range(trajectory_n)]
        total_rewards = [np.sum(trajectory['rewards']) for trajectory in trajectories]
        mean_rewards.append(np.mean(total_rewards))
        print('iteration:', iteration, 'mean total reward:', np.mean(total_rewards))

        # policy improvement
        quantile = np.quantile(total_rewards, q_param)
        elite_trajectories = []
        for trajectory in trajectories:
            total_reward = np.sum(trajectory['rewards'])
            if total_reward > quantile:
                elite_trajectories.append(trajectory)

        agent.fit(elite_trajectories)

    return mean_rewards


def grid_search(env, params):
    fig, axs = plt.subplots(figsize=(14, 8))

    axs.set_xlabel('Number of Iterations')
    axs.set_ylabel('Mean Total Rewards')

    labels = [['q_param = ' + f'{q}' + ', trajectory_n = ' + f'{n}' for n in params['trajectory_n']] for q in params['q_param']] 

    result = np.zeros((len(params['q_param']),
                       len(params['trajectory_n'])))

    rewards = []
    for i, q_param in enumerate(params['q_param']):
        for k, trajectory_n in enumerate(params['trajectory_n']):
            new_agent = CrossEntropyAgent(state_n, action_n)
            mean_rewards = train(env, new_agent, q_param, iteration_n, trajectory_n)
            result[i, k] = mean_rewards[-1]
            print(f"{q_param}, {iteration_n}, {trajectory_n}: mean reward = {mean_rewards[-1]}")
            rewards.append(mean_rewards)

            print(mean_rewards)

            axs.plot(mean_rewards, label=labels[i][k])

    print(rewards)

    plt.legend()
    plt.savefig('picture1.png')
    plt.show()

    return result

params = {
    'q_param': [0.6],
    'trajectory_n': [500]
}

iteration_n = 20

agent = CrossEntropyAgent(state_n, action_n)

result = grid_search(env, params)




