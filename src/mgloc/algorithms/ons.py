import numpy as np
from numpy import ndarray
from numpy.linalg import norm as get_norm

from src.mgloc.utils import get_loss_der1


class OnlineNewtonStep:
    def __init__(
        self,
        feature_dim: int,
        kappa: float,
        lam: float,
        theta_max_norm: float,
        lr: float,
        max_iteration: int,
    ):
        self.coef = np.random.standard_normal(feature_dim)
        norm = get_norm(self.coef)
        if norm > theta_max_norm:
            self.coef *= theta_max_norm / norm
        self.ons_bound = 2 * kappa * lam * theta_max_norm**2
        self.ons_bound: float

        self.kappa = kappa
        self.theta_max_norm = theta_max_norm  # S, not S^2
        self.lr = lr
        self.max_iteration = max_iteration
        return

    def get_projection(self, guide_theta: ndarray, gram_matrix: ndarray):
        if get_norm(guide_theta).item() <= self.theta_max_norm:
            return guide_theta

        theta = np.array(guide_theta)
        for _ in range(self.max_iteration):
            theta_past = np.array(theta)

            grad = 2 * gram_matrix @ (theta - guide_theta)
            theta -= self.lr * grad

            norm = get_norm(theta)
            if get_norm(theta) > self.theta_max_norm:
                theta *= self.theta_max_norm / norm

            diff = get_norm(theta - theta_past)
            if diff < 1e-6:  # noqa: PLR2004
                break

        # assert diff < 1e-6, diff  # noqa: PLR2004

        return theta

    def update(self, x: ndarray, y: int, gram_matrix: ndarray, gram_matrix_inv: ndarray):
        # gram_matrix and gram_matrix_inv must include (x, y) information
        grad = get_loss_der1(x, y, self.coef)
        guide_theta = self.coef - grad / self.kappa * gram_matrix_inv @ x
        self.coef = self.get_projection(guide_theta, gram_matrix)

        temp = (gram_matrix_inv @ x) @ x
        self.ons_bound += grad**2 * temp.item() / (2 * self.kappa)

        return
