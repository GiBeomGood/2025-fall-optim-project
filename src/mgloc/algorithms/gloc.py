from math import log, sqrt

import numpy as np
from numpy import ndarray

from .ons import OnlineNewtonStep


class GLOC:
    def __init__(
        self,
        # common
        feature_dim: int,
        kappa: float,
        theta_max_norm: float,
        # GLOC
        lam: float,
        delta: float,
        # ONS
        lr: float = 1e-4,
        max_iteration: int = 100,
    ):
        self.ons = OnlineNewtonStep(feature_dim, kappa, lam, theta_max_norm, lr, max_iteration)

        self.gram_matrix = np.eye(feature_dim) * lam
        self.gram_matrix_inv = np.eye(feature_dim) / lam
        self.coef_part2 = np.zeros((feature_dim,))  # X_t^\top z_t
        self.z_t_l2_norm = 0.0  # for calculating ellipsoid radius

        self.var_proxy = 0.5  # variance proxy of SubGaussian error with respect to logistic regression
        self.lam = lam
        self.kappa = kappa
        self.theta_max_norm = theta_max_norm  # S
        self.delta = delta
        return

    def decide_action(self, features: ndarray) -> tuple[int, ndarray]:
        # features: (K x d)
        dots = features @ self.coef_hat  # (K x d) x (d) -> (K)
        radius = self.get_ellipsoid_radius()
        temp = (features * (features @ self.gram_matrix_inv)).sum(1)
        dots += radius * np.sqrt(np.clip(temp, 0, None))

        action = dots.argmax(0).item()
        selected = features[action]

        return action, selected

    def update(self, feature: ndarray, reward: int):
        temp = self.gram_matrix_inv @ feature
        self.gram_matrix_inv -= np.outer(temp, temp) / (1 + feature @ temp)
        self.gram_matrix += np.outer(feature, feature)

        temp = (feature @ self.ons.coef).item()
        self.coef_part2 += feature * temp
        self.z_t_l2_norm += temp**2

        self.ons.update(feature, reward, self.gram_matrix, self.gram_matrix_inv)

        return

    @property
    def coef_hat(self):
        return self.gram_matrix_inv @ self.coef_part2

    @property
    def coef(self):
        return self.ons.coef

    def get_ellipsoid_radius(self):
        temp = sqrt(1 + 2 / self.kappa * self.ons.ons_bound + (4 * self.var_proxy**4) / (self.kappa**4 * self.delta**2))
        radius = self.lam * self.theta_max_norm**2 + 1 + 4 * self.ons.ons_bound / self.kappa
        radius += 8 * (self.var_proxy / self.kappa) ** 2 * log(2 / self.delta * temp)

        radius -= self.z_t_l2_norm - self.coef_part2 @ self.coef_hat
        radius = sqrt(np.clip(radius, 1, None).item())

        return radius
