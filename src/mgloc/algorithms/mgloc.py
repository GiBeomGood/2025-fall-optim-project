import numpy as np
from numpy import ndarray
from scipy.special import expit

from .gloc import GLOC
from .oful import OFUL
from .ons import OnlineNewtonStep


class MGLOC:
    def __init__(self, feature_dim, kappa, lam, theta_max_norm, lr, max_iteration, delta):
        self.ons = OnlineNewtonStep(feature_dim, kappa, lam, theta_max_norm, lr, max_iteration)
        self.oful = OFUL(feature_dim, lam, delta)
        self.gloc = GLOC(feature_dim - 1, kappa, theta_max_norm, lam, delta, lr, max_iteration)

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

    def impute(self, features: ndarray):
        temp = np.array(features)
        temp[:, -1] = 0.5  # [w; 0.5]
        x_impute = temp @ self.oful.coef_hat  # v^\top \hat{\gamma}

        indices = np.isnan(features[:, -1])
        features[indices, -1] = x_impute[indices]
        return features

    def decide_action(self, features: ndarray):
        count_nan = np.isnan(features[:, -1]).sum().item()
        if count_nan == 2:  # noqa: PLR2004  # t \in \tau_4
            action, _ = self.gloc.decide_action(features[:, :-1])

            return action, features[action]

        features = self.impute(features)
        radius = self.get_ellipsoid_radius()

        temp = (features * (features @ self.gram_matrix_inv)).sum(1)
        dots = features @ self.coef_hat + radius * np.sqrt(np.clip(temp, 0, None))
        action = dots.argmax(0).item()

        return action, features[action]  #! `selected` is \tilde{b}, possibly imputed

    def update_beta(self, feature: ndarray, reward: int):
        #! `feature` must be \tilde{b}
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

    def update_gamma(self, feature: ndarray, reward: int):
        #! `feature` must correspond to a_t', not a_t
        if reward is None:
            reward = expit(feature[:-1] @ self.gloc.coef_hat)  # use \hat{y} if needed

        x = feature[-1].item()
        feature = np.concat([feature[:-1], np.array([reward])]).astype(float)  # [w; y]
        self.oful.update(feature, x)
        return

    def update_alpha(self, feature: ndarray, reward: int):
        feature = feature[:-1]  # w
        self.gloc.update(feature, reward)
        return

    def get_ellipsoid_radius(self):
        # temp = sqrt(1 + 2 / self.kappa * self.ons.ons_bound + (4 * self.var_proxy**4) / (self.kappa**4 * self.delta**2))
        # radius = self.lam * self.theta_max_norm**2 + 1 + 4 * self.ons.ons_bound / self.kappa
        # radius += 8 * (self.var_proxy / self.kappa) ** 2 * log(2 / self.delta * temp)

        # radius -= self.z_t_l2_norm - self.coef_part2 @ self.coef_hat
        # radius = sqrt(radius.item())

        return 0.1
