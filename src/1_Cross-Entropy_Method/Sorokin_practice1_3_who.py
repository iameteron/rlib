import gym
import gym_maze
import numpy as np
import random
import time
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

env = gym.make("Taxi-v3")
state_n = 500
action_n = 6


class CrossEntropyAgent():
    def __init__(self, state_n, action_n, stochastic_env):
        self.state_n = state_n
        self.action_n = action_n
        self.stochastic_env = stochastic_env

        #initialization with equal probabilities
        self.model = np.ones((self.state_n, self.action_n)) / self.action_n

    def get_action(self, state):
        action = np.random.choice(np.arange(self.action_n), p=self.model[state])
        return int(action)
    
    def fit(self, elite_trajectories, lambda_1=0, lambda_2=0):
        new_model = np.zeros((self.state_n, self.action_n))
        for trajectory in elite_trajectories:
            for state, action in zip(trajectory['states'], trajectory['actions']):
                new_model[state][action] += 1
            
        if lambda_1 > 0:
            # with laplace smoothing
            for state in range(self.state_n):
                new_model[state] = (new_model[state] + lambda_1) / (np.sum(new_model[state]) + lambda_1 * action_n)
            
        else:
            # w/o laplace smooothing
            for state in range(self.state_n):
                if np.sum(new_model[state]) > 0:
                    new_model[state] /= np.sum(new_model[state])
                else:
                    new_model[state] = self.model[state].copy()

        if lambda_2 > 0:
            # with policy smoothing
            new_model = lambda_2 * new_model + (1 - lambda_2) * self.model


        self.model = new_model
        return None


def get_state(obs):
    return obs


def get_trajectory(env, agent, max_len=1000, visualize=False):
    trajectory = {'states': [], 'actions': [], 'rewards': []}

    obs = env.reset()
    state = get_state(obs)

    for _ in range(max_len):
        trajectory['states'].append(state)

        action = agent.get_action(state)
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


def train(env, agent, q_param, iteration_n, trajectory_n, policies_n, lambda_1=0, lambda_2=0):
    rewards = []
    for iteration in range(iteration_n):
        total_rewards = []
        if agent.stochastic_env:

            # policy evaluation
            stochastic_policy = agent.model.copy()
            trajectories = []
            for i in tqdm(range(policies_n),  desc="Iteration {0}".format(iteration)):
                det_policy = np.zeros((agent.state_n, agent.action_n))
                for state in range(agent.state_n):
                    action = np.random.choice(np.arange(agent.action_n), p=stochastic_policy[state])
                    det_policy[state][action] = 1
                agent.model = det_policy
                trs = [get_trajectory(env, agent) for _ in range(trajectory_n)]
                trajectories += trs
                local_reward = np.mean([np.sum(trajectory['rewards']) for trajectory in trs])
                #total_rewards += [local_reward for _ in trs]
                total_rewards += [local_reward]
            rewards.append(np.mean(total_rewards))
            agent.model = stochastic_policy
            #print(len(total_rewards))
            print('iteration:', iteration, 'policy rewards:', rewards[-1])

            #policy improvement
            quantile = np.quantile(total_rewards, q_param)
            elite_trajectories = []
            for index, policy_reward in enumerate(total_rewards):
                if policy_reward > quantile:
                    elite_trajectories.append(trajectories[index])

            agent.fit(elite_trajectories)

        if not agent.stochastic_env:

            #policy evaluation
            trajectories = [get_trajectory(env, agent) for _ in range(trajectory_n)]
            total_rewards = [np.sum(trajectory['rewards']) for trajectory in trajectories]
            rewards.append(np.mean(total_rewards))
            #print('iteration:', iteration, 'mean total reward:', np.mean(total_rewards))

            #policy improvement
            quantile = np.quantile(total_rewards, q_param)
            elite_trajectories = []
            for trajectory in trajectories:
                total_reward = np.sum(trajectory['rewards'])
                if total_reward > quantile:
                    elite_trajectories.append(trajectory)

            agent.fit(elite_trajectories, lambda_1, lambda_2)


    return rewards

def grid_search(env, params):
    result = np.zeros((len(params['lambda_1']),
                       len(params['lambda_2'])))

    for i, lambda_1 in enumerate(params['lambda_1']):
        for j, lambda_2 in enumerate(params['lambda_2']):
            new_agent = CrossEntropyAgent(state_n, action_n)
            mean_reward = train(env, new_agent, q_param, iteration_n, trajectory_n, lambda_1, lambda_2)
            result[i, j] = mean_reward
            print(f"{lambda_1}, {lambda_2}: mean reward = {mean_reward}")

    return result


agent = CrossEntropyAgent(state_n, action_n, stochastic_env=True)

q_param = 0.5
iteration_n = 20
trajectory_n = 25

policies_n = 250

rewards = train(env, agent, q_param, iteration_n, trajectory_n, policies_n, lambda_2=0.5)

fig, axs = plt.subplots(figsize=(12,8))
axs.plot(rewards)
plt.show()

for _ in range(10):
    trajectory = get_trajectory(env, agent, max_len=200, visualize=True)


