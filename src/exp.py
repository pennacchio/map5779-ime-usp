import emcee
import numpy as np
import pandas as pd
from scipy import optimize

# Load data
df = pd.read_csv("data/data.csv")

# Define constant values
n = df.shape[0]  # Number of samples
X = df.loc[:, ["latitude", "longitude", "altitude"]].values  # Independent vars
k = X.shape[1]  # Number of independent variables
X = np.hstack([np.ones((n, 1)), X])  # Add intercept column
d = k + 2  # Number of model parameters (betas and sigma)
y = df.loc[:, "temperature"].values  # Dependent var
beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]  # X * beta_hat = y
y_hat = X @ beta_hat
s = np.sqrt((y - y_hat).T @ (y - y_hat) / (n - k - 1))

# Set of hypotheses to test. Each element corresponds to the betas that are zero
# for the given hypothesis
hypotheses = [[1], [2], [3]]


# Define log posterior and its gradient
def log_posterior(beta, sigma):
    """Log-posterior for linear regression with diffuse prior."""
    return -(n + 1) * np.log(sigma) - (0.5 / (sigma ** 2)) * (
        ((n - k - 1) * (s ** 2)) + ((beta - beta_hat).T @ X.T) @ (X @ (beta - beta_hat))
    )


def log_posterior_grad(beta, sigma):
    """Log-posterior gradient for linear regression with diffuse prior."""
    beta_grad = -(1 / (sigma ** 2)) * (beta - beta_hat).T @ X.T @ X
    sigma_grad = -((n + 1) / sigma) + (1 / sigma ** 3) * (
        ((n - k - 1) * (s ** 2)) + ((beta - beta_hat).T @ X.T) @ (X @ (beta - beta_hat))
    )
    return beta_grad, sigma_grad


# Wrappers needed for stacked args and returns
def neg_log_posterior_opt_wrapper(beta_sigma):
    """Optimization wrapper for `log_posterior`."""
    return -log_posterior(beta_sigma[:-1], beta_sigma[-1])


def neg_log_posterior_grad_opt_wrapper(beta_sigma):
    """Optimization wrapper for `log_posterior_grad`."""
    beta_grad, sigma_grad = log_posterior_grad(beta_sigma[:-1], beta_sigma[-1])
    return -np.hstack([beta_grad, sigma_grad])


# Wrapper needed to enforce sigma > 0 constraint
def log_posterior_mcmc_wrapper(beta_sigma):
    """MCMC wrapper for `log_posterior`."""
    if beta_sigma[-1] <= 0:
        return -np.inf
    return log_posterior(beta_sigma[:-1], beta_sigma[-1])


def make_opt_constraint(zeroed_betas):
    """Make optimization constraint for a given null hypothesis."""
    A = []
    n_zeroed_betas = len(zeroed_betas)
    lb, ub = np.zeros(n_zeroed_betas + 1), np.zeros(n_zeroed_betas + 1)
    for zeroed_beta in zeroed_betas:
        beta_A = np.zeros(d)
        beta_A[zeroed_beta] = 1
        A.append(beta_A)

    # Add sigma constraint
    sigma_A = np.zeros(d)
    sigma_A[-1] = 1
    A.append(sigma_A)
    A = np.array(A)
    ub[-1] = np.inf

    return optimize.LinearConstraint(A, lb, ub)


def opt(zeroed_betas):
    """Run optimization step for a given null hypothesis."""
    constraint = make_opt_constraint(zeroed_betas)

    # Optimization starting point
    x0 = np.array([0, 0, 0, 0, 1])

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

    return res


def integrate(opt_res):
    """Run integration step for a given null hypothesis."""
    # Integration settings
    n_walkers = 64
    p0 = np.random.rand(n_walkers, d) - 0.5 + np.array([0, 0, 0, 0, 1])
    n_steps = 10000

    # Run sampler
    print("Running integration step...")
    sampler = emcee.EnsembleSampler(n_walkers, d, log_posterior_mcmc_wrapper)
    sampler.run_mcmc(p0, n_steps, progress=True, tune=True)

    # Get chain samples to estimate the e-value
    max_autocorr_time = sampler.get_autocorr_time().max()
    discard = int(max_autocorr_time * 8)
    thin = int(max_autocorr_time / 2)
    flat_samples = sampler.get_chain(discard=discard, thin=thin, flat=True)

    # Estimate e-value
    def indicator_fn(beta_sigma):
        """Indicator function used to define the integral volume."""
        return int(neg_log_posterior_opt_wrapper(beta_sigma) > opt_res.fun)

    e_val = np.mean([indicator_fn(beta_sigma) for beta_sigma in flat_samples])

    return e_val


def run():
    """Run experiment."""
    for hypothesis in hypotheses:
        hypothesis_str = ", ".join(f"ß{i}=0" for i in hypothesis)
        print(f"Testing hypothesis {hypothesis_str}")
        opt_res = opt(hypothesis)
        e_val = integrate(opt_res)
        print(f"Hypothesis e-value: {e_val:.2f}\n")


if __name__ == "__main__":
    run()
