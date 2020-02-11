import emcee
import numpy as np
import pandas as pd
from scipy import optimize

# Load data
df = pd.read_csv("data/data.csv")

# Define constant values
n = df.shape[0]  # Number of samples
X = df.loc[:, ["latitude", "altitude", "longitude"]].values  # Independent vars
X = np.hstack([np.ones((n, 1)), X])  # Add intercept column
k = X.shape[1]  # Number of independent variables
d = k + 1  # Number of model parameters (betas and sigma)
y = df.loc[:, "temperature"].values  # Dependent var
beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]  # X * beta_hat = y
y_hat = X @ beta_hat
s = np.sqrt((y - y_hat).T @ (y - y_hat) / (n - k))


# Define log posterior and its gradient
def log_posterior(beta, sigma):
    """Log-posterior for linear regression with diffuse prior."""
    return -(n + 1) * np.log(sigma) - (0.5 / (sigma ** 2)) * (
        ((n - k) * (s ** 2)) + (beta - beta_hat).T @ X.T @ X @ (beta - beta_hat)
    )


def log_posterior_grad(beta, sigma):
    """Log-posterior gradient for linear regression with diffuse prior."""
    beta_grad = -(1 / (sigma ** 2)) * (beta - beta_hat).T @ X.T @ X
    sigma_grad = -((n + 1) / sigma) + (1 / sigma ** 3) * (
        ((n - k) * (s ** 2)) + (beta - beta_hat).T @ X.T @ X @ (beta - beta_hat)
    )
    return beta_grad, sigma_grad


# Optimization constraint representing the null hypothesis
A = np.array([[1, 0, 0, 0, 0], [0, 0, 0, 0, 1]])
lb = [0, 0]
ub = [0, np.inf]
constraint = optimize.LinearConstraint(A, lb, ub)

# Optimization starting point
x0 = np.array([0, 0, 0, 0, 1])


# Wrappers needed for stacked args and returns
def neg_log_posterior_opt_wrapper(beta_sigma):
    """Optimization wrapper for `log_posterior`."""
    return -log_posterior(beta_sigma[:-1], beta_sigma[-1])


def neg_log_posterior_grad_opt_wrapper(beta_sigma):
    """Optimization wrapper for `neg_log_posterior_grad`."""
    beta_grad, sigma_grad = log_posterior_grad(beta_sigma[:-1], beta_sigma[-1])
    return -np.hstack([beta_grad, sigma_grad])


# Run optimization
print("Running optimization step...")
res = optimize.minimize(
    neg_log_posterior_opt_wrapper,
    x0=x0,
    method="trust-constr",
    jac=neg_log_posterior_grad_opt_wrapper,
    constraints=constraint,
    options={"maxiter": 10000},
)

# Integration settings
n_walkers = 64
p0 = np.random.rand(n_walkers, d) - 0.5 + x0  # Starting points
n_steps = 10000


# Wrapper needed to enforce sigma > 0 constraint
def log_posterior_mcmc_wrapper(beta_sigma):
    """MCMC wrapper for `log_posterior`."""
    if beta_sigma[-1] <= 0:
        return -np.inf
    return log_posterior(beta_sigma[:-1], beta_sigma[-1])


# Run sampler
print("Running integration step...")
sampler = emcee.EnsembleSampler(n_walkers, d, log_posterior_mcmc_wrapper)
state = sampler.run_mcmc(p0, n_steps, progress=True, tune=True)

# Get chain samples to estimate the e-value
max_autocorr_time = sampler.get_autocorr_time().max()
discard = int(max_autocorr_time * 8)
thin = int(max_autocorr_time / 2)
flat_samples = sampler.get_chain(discard=discard, thin=thin, flat=True)


# Estimate e-value
def indicator_fn(beta_sigma):
    """Indicator function used to define the integral volume."""
    return int(neg_log_posterior_opt_wrapper(beta_sigma) > res.fun)


e_val = np.mean([indicator_fn(beta_sigma) for beta_sigma in flat_samples])
print(f"Hypothesis e-value: {e_val:.2f}")
