from math import sqrt
from random import choices

import numpy as np
from numpy import ndarray
from numpy.linalg import norm as get_norm
from numpy.random import binomial, standard_normal
from scipy.special import expit


class Environment:
    def __init__(
        self,
        beta_true: ndarray,
        action_size: int,
        feature_dim: int,
        x_std: float,
        gamma_true: ndarray,
        alpha_true: ndarray,
    ):
        self.beta_true = beta_true
        self.gamma_true = gamma_true
        self.alpha_true = alpha_true

        self.x_std = x_std
        self.action_size = action_size
        self.feature_dim = feature_dim
        return

    def get_data(self) -> ndarray:
        features = standard_normal((2, self.feature_dim)) * sqrt(10 / self.feature_dim)
        norm = get_norm(features[1:, :-1])
        if norm > 2.0:  # noqa: PLR2004
            features *= 2.0 / norm
        features[:, 0] = 1

        # features = np.random.random((self.action_size, self.feature_dim)) * sqrt(10 / (self.feature_dim - 1))
        prob = expit(features[:, :-1] @ self.alpha_true)
        rewards = binomial(1, prob)

        temp = np.concat([features[:, :-1], rewards.reshape(-1, 1)], axis=1)
        x = temp @ self.gamma_true + np.random.standard_normal((2,)) * self.x_std

        temp = np.array(choices([np.nan, 0], k=2)).astype(float)
        x += temp
        features[:, -1] = x

        return features, rewards

    def get_optimal(self, features: ndarray) -> tuple[int, ndarray]:
        # `features` must be imputed result
        rewards = features @ self.beta_true
        action = rewards.argmax(0).item()

        return action

    def get_regret(self, features: ndarray, action: int) -> float:
        features = self.impute(features)
        best_action = self.get_optimal(features)

        regret = expit(features[best_action] @ self.beta_true) - expit(features[action] @ self.beta_true)
        return regret.item()

    def impute(self, features: ndarray):
        temp = np.array(features)
        temp[:, -1] = 0.5  # [w; 0.5]
        x_impute = temp @ self.gamma_true  # v^\top \hat{\gamma}

        indices = np.isnan(features[:, -1])
        features[indices, -1] = x_impute[indices]

        return features
