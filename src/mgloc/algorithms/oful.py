import numpy as np
from numpy import ndarray


class OFUL:
    def __init__(self, feature_dim: int, lam: float, delta: float):
        self.gram_matrix = np.eye(feature_dim) * lam
        self.gram_matrix_inv = np.eye(feature_dim) / lam
        self.coef_part2 = np.zeros((feature_dim,))

        self.lam: float = lam
        self.delta: float = delta
        return

    def update(self, feature: ndarray, reward: float):
        temp = self.gram_matrix_inv @ feature
        self.gram_matrix_inv -= np.outer(temp, temp) / (1 + temp @ feature)
        self.gram_matrix += np.outer(feature, feature)
        self.coef_part2 += reward * feature
        return

    @property
    def coef_hat(self):
        return self.gram_matrix_inv @ self.coef_part2
