import random

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from . import config
from .algorithms.gloc import GLOC
from .algorithms.mgloc import MGLOC
from .environment import Environment
from .visualization import draw_plot


def _run_single_experiment(env: Environment, model: MGLOC, gloc: GLOC, seed: int):
    # This seed controls the data generation within a single experiment run
    np.random.seed(seed)
    random.seed(seed)
    regret_list = []
    reward_list = []
    reward_g_list = []

    for _ in range(config.T):
        features, rewards = env.get_data()
        is_nan = np.isnan(features[:, -1])

        action, selected = model.decide_action(features)
        reward = rewards[action].item()

        action_miss = is_nan[action].item()
        other_miss = is_nan[1 - action].item()

        if not action_miss:
            if not other_miss:  # t \in \tau_1
                model.update_beta(selected, reward)
            else:  # t \in \tau_2
                model.update_beta(selected, reward)
                model.update_gamma(selected, reward)
        else:
            if not other_miss:  # t \in \tau_3
                model.update_beta(selected, reward)
                model.update_gamma(features[1 - action], reward=None)
                model.update_alpha(selected, reward)
            else:  # t \in \tau_4
                model.update_alpha(selected, reward)

        regret = env.get_regret(features, action)
        regret_list.append(regret)
        reward_list.append(reward)

        action_g, selected_g = gloc.decide_action(features[:, :-1])
        reward_g = rewards[action_g].item()
        gloc.update(selected_g, reward_g)
        reward_g_list.append(reward_g)

    return regret_list, reward_list, reward_g_list


def run_and_visualize():
    """
    Runs the full simulation and returns the generated figures.
    """
    regret_list_all = []
    reward_list_all = []
    reward_g_list_all = []

    for i in tqdm(range(config.NUM_SIMULATIONS), desc="Running Simulations"):
        # This seed controls the initialization of the models for each simulation run
        np.random.seed(i)
        random.seed(i)

        env = Environment(**config.ENV_CONFIG)
        model = MGLOC(**config.MGLOC_CONFIG)
        gloc = GLOC(**config.GLOC_CONFIG)

        # The data for each of the 100 runs is identical (seed=42), but model initializations are different.
        # This matches the logic of the original script.
        regret_list, reward_list, reward_g_list = _run_single_experiment(env, model, gloc, seed=42)
        regret_list_all.append(np.cumsum(regret_list))
        reward_list_all.append(np.cumsum(reward_list))
        reward_g_list_all.append(np.cumsum(reward_g_list))

    regret_list_all = np.array(regret_list_all)
    reward_list_all = np.array(reward_list_all)
    reward_g_list_all = np.array(reward_g_list_all)

    # --- Create Visualizations ---
    figures = {}

    # Figure 1: Cumulative Regret
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 5))
    ax1 = draw_plot(ax1, regret_list_all, color="blue", label="M-GLOC", t_max=config.T)
    ax1.set_ylabel("Cumulative Regret")
    fig1.tight_layout()
    figures["cumulative_regret"] = fig1

    # Figure 2: Average Instantaneous Regret
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 5))
    avg_regret = regret_list_all / np.arange(1, config.T + 1)
    ax2 = draw_plot(ax2, avg_regret, color="blue", label="M-GLOC", t_max=config.T)
    ax2.set_ylabel("Average of Instantaneous Regret")
    fig2.tight_layout()
    figures["average_regret"] = fig2

    # Figure 3: Cumulative Reward Comparison
    fig3, ax3 = plt.subplots(1, 1, figsize=(10, 5))
    ax3 = draw_plot(ax3, reward_g_list_all, color="orange", label="GLOC", t_max=config.T)
    ax3 = draw_plot(ax3, reward_list_all, color="blue", label="M-GLOC", t_max=config.T)
    ax3.set_ylabel("Cumulative Reward")
    fig3.tight_layout()
    figures["cumulative_reward"] = fig3

    return figures
