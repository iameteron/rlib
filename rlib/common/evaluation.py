import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation


def save_frames_as_gif(frames, path="./gifs/", filename="gym_animation.gif", fps=60):
    plt.figure(figsize=(frames[0].shape[1] / 50.0, frames[0].shape[0] / 50.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis("off")

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save(path + filename, writer="imagemagick", fps=fps)


def get_trajectory(env, agent, visualize=False, deterministic=True):
    trajectory = {
        "states": [],
        "actions": [],
        "rewards": [],
        "terminated": [],
        "truncated": [],
    }
    frames = []

    obs, _ = env.reset()

    while True:
        action, _ = agent.predict(obs, deterministic=deterministic)
        next_obs, reward, terminated, truncated, _ = env.step(action)

        trajectory["states"].append(obs)
        trajectory["actions"].append(action)
        trajectory["rewards"].append(reward)
        trajectory["terminated"].append(terminated)
        trajectory["truncated"].append(truncated)

        obs = next_obs

        if terminated or truncated:
            break

        if visualize:
            frames.append(env.render())

    if visualize:
        print("saving...")
        save_frames_as_gif(frames)

    return trajectory


def validation(env, agent, validation_n: int = 20, deterministic=False):
    total_rewards = []
    for _ in range(validation_n):
        trajectory = get_trajectory(env, agent, deterministic=deterministic)
        total_rewards.append(np.sum(trajectory["rewards"]))

    return np.mean(total_rewards)