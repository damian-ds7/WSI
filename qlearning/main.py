from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

PATH = Path(__file__).parent
MAX_WORKERS = 12


def base_reward(
    reward: float, old_state: int, new_state: int, terminated: bool, truncated: bool
) -> int:
    return reward


def negative_hole(
    reward: float, old_state: int, new_state: int, terminated: bool, truncated: bool
) -> int:
    if reward == 0 and (terminated or truncated):
        return -1
    return reward


def negative_stagnation_hole(
    reward: float, old_state: int, new_state: int, terminated: bool, truncated: bool
) -> int:
    if reward == 0 and (new_state == old_state or (terminated or truncated)):
        return -1
    return reward

def run_qlearning(
    env: gym.Env,
    reward_strategy: Callable[[float, int, int, bool, bool], int],
    episode_count: int,
    step_count: int,
) -> np.ndarray[float]:

    max_epsilon = 1.0
    min_epsilon = 0.01
    epsilon_decay = 0.95
    epsilon = max_epsilon

    learning_rate = 0.3
    discount_rate = 0.97
    qtable = np.zeros((env.observation_space.n, env.action_space.n))

    averaged_reward = np.zeros(episode_count)

    for episode in range(episode_count):
        state, _ = env.reset()

        for step in range(step_count):
            if np.random.uniform(0, 1) > epsilon and not np.all(qtable[state, :] == 0):
                action = np.argmax(qtable[state])
            else:
                action = env.action_space.sample()

            new_state, reward, terminated, truncated, _ = env.step(action)

            new_reward = reward_strategy(
                reward, state, new_state, terminated, truncated
            )

            delta = (
                new_reward
                + discount_rate * np.max(qtable[new_state, :])
                - qtable[state, action]
            )

            qtable[state, action] += learning_rate * delta

            averaged_reward[episode] += reward

            state = new_state

            if terminated or truncated:
                break

        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(
            -epsilon_decay * episode
        )

    return averaged_reward


def get_averaged(
    env: gym.Env,
    reward_strategy: Callable[[float, int, int, bool, bool], int],
    run_count: int = 25,
    episode_count: int = 1000,
    step_count: int = 200,
) -> np.ndarray[float]:

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(
                run_qlearning, env, reward_strategy, episode_count, step_count
            )
            for _ in range(run_count)
        ]

        results = [future.result() for future in futures]

    return np.mean(results, axis=0)

def generate_plot(
    filename: str,
    averaged_base: np.ndarray[float],
    averaged_modified: np.ndarray[float],
):
    plot_path = PATH / "plots"
    plot_path.mkdir(parents=True, exist_ok=True)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.spines["left"].set_position("center")
    ax.spines["bottom"].set_position("zero")
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    plt.plot(averaged_base, "r")
    plt.plot(averaged_modified, "b")
    plt.savefig(plot_path / f"{filename}.png")
    plt.clf()


def generate_normal_data():
    env = gym.make("FrozenLake-v1", desc=None, map_name="8x8", is_slippery=False)
    reward_strats = [
        base_reward,
        negative_hole,
        negative_stagnation_hole,
    ]
    for strat in reward_strats:
        filename = strat.__name__
        base_rewards = get_averaged(env, base_reward, episode_count=1000, run_count=50)
        modified_rewards = get_averaged(env, strat, episode_count=1000, run_count=50)

        generate_plot(filename, base_rewards, modified_rewards)


def generate_slippery_data():
    env = gym.make("FrozenLake-v1", desc=None, map_name="8x8", is_slippery=True)
    reward_strats = [negative_hole, negative_stagnation_hole]
    for strat in reward_strats:
        filename = strat.__name__ + "_slippery"
        base_rewards = get_averaged(env, base_reward, episode_count=10000)
        modified_rewards = get_averaged(env, strat, episode_count=10000)

        generate_plot(filename, base_rewards, modified_rewards)


if __name__ == "__main__":
    generate_normal_data()
    # generate_slippery_data()
