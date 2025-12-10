import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray


def draw_plot(ax: plt.Axes, data: ndarray, color: str, label: str, t_max: int):
    tail_percent = 5

    lower_percentile = np.percentile(data, tail_percent, axis=0)
    upper_percentile = np.percentile(data, 100 - tail_percent, axis=0)

    ax.plot(data.mean(0), label=label, color=color)
    ax.fill_between(np.arange(t_max), lower_percentile, upper_percentile, color=color, alpha=0.2)
    ax.scatter(np.arange(t_max)[::1000], data.mean(0)[::1000], marker="x", color=color)

    ax.set_xlim(0, t_max)
    ax.set_xlabel("Time")
    ax.set_ylim(0, None)
    ax.legend()

    return ax
