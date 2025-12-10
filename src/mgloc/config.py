import numpy as np
from numpy.linalg import norm as get_norm

from .utils import get_alpha

# --- Constants and Configurations ---
FEATURE_DIM = 18
ACTION_SIZE = 2
COEF_L2_NORM = 3.0
T = 10000
NUM_SIMULATIONS = 100

# This seed is for the initial generation of true parameters
np.random.seed(42)
BETA_TRUE = np.random.standard_normal(FEATURE_DIM)
BETA_TRUE[-1] = 3.0
BETA_TRUE *= COEF_L2_NORM / get_norm(BETA_TRUE)

GAMMA_TRUE = np.random.standard_normal(FEATURE_DIM)
GAMMA_TRUE *= COEF_L2_NORM / get_norm(GAMMA_TRUE)
GAMMA_TRUE[-1] = BETA_TRUE[-1] * 1.0

ALPHA_TRUE = get_alpha(BETA_TRUE, GAMMA_TRUE)

ENV_CONFIG = {
    "beta_true": BETA_TRUE,
    "action_size": ACTION_SIZE,
    "feature_dim": FEATURE_DIM,
    "x_std": 1.0,
    "gamma_true": GAMMA_TRUE,
    "alpha_true": ALPHA_TRUE,
}
MGLOC_CONFIG = {
    "feature_dim": FEATURE_DIM,
    "kappa": 0.5,
    "theta_max_norm": COEF_L2_NORM,
    "lam": 1.0,
    "delta": 0.05,
    "lr": 0.01,
    "max_iteration": 100,
}
GLOC_CONFIG = {
    "feature_dim": FEATURE_DIM - 1,
    "kappa": 0.5,
    "theta_max_norm": COEF_L2_NORM,
    "lam": 1.0,
    "delta": 0.05,
    "lr": 0.01,
    "max_iteration": 100,
}
