import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from itertools import combinations

from prepare_data import u, u_tilde, M, Sigma_u, names, p

T, K = u.shape
m = K * (K - 1) // 2



def given_rotations(theta, K):
    """Q(theta) as in Equation (12): product of m = K(K-1)/2 Givens rotations."""
    R = np.eye(K)
    idx = 0
    for k in range(K):
        for i in range(k + 1, K):
            c, s = np.cos(theta[idx]), np.sin(theta[idx])
            idx += 1
            G = np.eye(K)
            G[[k, k, i, i], [k, i, k, i]] = [c, s, -s, c]
            R = R @ G
    return R.T


def log_pseudo_marginal(v, nu_k):
    """log f_k(v): log-density of a t(nu_k) rescaled to unit variance."""
    s = np.sqrt(nu_k / (nu_k - 2))          # scaling so that Var = 1
    z = v * s
    log_f_k = (gammaln((nu_k + 1) / 2) - gammaln(nu_k / 2)
               - 0.5 * np.log(np.pi * nu_k)
               - (nu_k + 1) / 2 * np.log1p(z**2 / nu_k)
               + np.log(s))
    return log_f_k

def PML_objective_fct(parameter_vec):
    """Pseudo Maximum Likelihood Function"""
    theta, nu = parameter_vec[:m], parameter_vec[m:]     # theta_1,...,theta15, nu1,...,n6
    if np.any(nu <= 2.001): # since nu has to be larger than 2, since s = np.sqrt(nu_k / (nu_k - 2))
      return 10000000 # as recommended by  L-BFGS-B
    Q_theta = given_rotations(theta, K)                  # Q(theta)
    V = u_tilde @ Q_theta                                # V[:, k] = q_k(theta)^T u_tilde_t
    return -sum(log_pseudo_marginal(V[:, k], nu[k]).mean() for k in range(K)) # has to be minus, since we use minimizer

random_nr = np.random.default_rng(42)
range_theta = [(0, 2*np.pi)] * m      # theta in [0, 2pi)^m
range_nu = [(2.05, np.inf)] * K       # nu in (2, +inf)^K
bounds = range_theta + range_nu

max_PML, values_PML = None, []
for _ in range(100): # 100 iterations
    start = np.concatenate([random_nr.uniform(0, 2*np.pi, m),
                            random_nr.uniform(3, 10, K)]) # since Lütkepohl & Strohsal have df between 3.27 and 7.58
    results = minimize(PML_objective_fct, start, method="L-BFGS-B", bounds=bounds)
    values_PML.append(-results.fun)          # -results.fun = + pseudo log-likelihood
    if max_PML is None or results.fun < max_PML.fun:
        max_PML = results

print(np.round(sorted(values_PML)[-10:], 4))     # last top values

theta_hat = max_PML.x[:m]
nu_hat    = max_PML.x[m:]
Q_pml     = given_rotations(theta_hat, K) # Q(theta)
B_pml     = M @ Q_pml
w_pml     = u_tilde @ Q_pml               # estimated shocks (T, K)

print(np.allclose(Q_pml @ Q_pml.T, np.eye(K)))          # orthogonality
print(np.allclose(B_pml @ B_pml.T, Sigma_u))            # decomposition holds
print(np.round(np.corrcoef(w_pml, rowvar=False), 3))    # approx I
print(np.round(nu_hat, 2))                              # cf. Table 1 in L&S
print(len(w_pml))



pairs = list(combinations(range(K), 2))

print("\n--- RESULTS ---")
print(f"Attained log-likelihood: {-max_PML.fun:.4f}") # -7.8535
#print(f"Maximum error B B' vs. Sigma_u: {max_diff:.10f}")

B_df = pd.DataFrame(B_pml, index=names, columns=[f"w_{i+1}" for i in range(K)])
nu_df = pd.DataFrame(nu_hat, index=names, columns=["df (hat_nu)"])

print("\n--- Estimated degrees of freedom ---")
print(nu_df)

print("\n--- Reconstructed structural matrix B ---")
print(B_df)

print("\n--- Estimated rotation angle pairs theta ---")
theta_deg = np.degrees(theta_hat)
for idx, (k, i) in enumerate(pairs):
    print(f"theta_({k+1},{i+1}) = {theta_deg[idx]:7.2f}°")
