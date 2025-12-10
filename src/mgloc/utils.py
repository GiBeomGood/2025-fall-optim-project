import numpy as np
from numpy import ndarray
from scipy.special import expit, softplus


def get_loss(value: float, reward: int) -> float:
    # loss = -reward * value + np.log(1 + np.exp(value))
    # loss = loss.item()
    loss = softplus((1 - 2 * reward) * value)
    return loss


def get_loss_der1(x: ndarray, y: int, weight: ndarray) -> float:
    # result = -y + m'(z)
    result = -y + expit(x @ weight)
    return result.item()


def get_alpha(beta: ndarray, gamma: ndarray) -> ndarray:
    alpha = np.empty((beta.shape[0] - 1))
    alpha[0] = beta[0] + beta[-1] * gamma[0] + beta[-1] * gamma[-1] / 2
    alpha[1:] = beta[1:-1] + beta[-1] * gamma[1:-1]
    alpha = np.array(alpha)
    return alpha
