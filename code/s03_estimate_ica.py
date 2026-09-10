import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.integrate import quad
from scipy.stats import norm
from itertools import combinations


from s01_prepare_data import u_tilde, M, names
from s02_estimate_pml import given_rotations, K, m


def G_1(v, a=1):
    """G_1(v) = (1/a) log cosh(a v), Eq. (26) in Hyvärinen and Oja (2000).
    Evaluated as (|av| + log1p(exp(-2|av|)) - log 2)/a to avoid overflow."""
    av = np.abs(a * v)
    return (av + np.log1p(np.exp(-2 * av)) - np.log(2)) / a

def G_2(v):
    """G_2(v) = -exp(-v^2 / 2), Eq. (26)."""
    return -np.exp(-v**2 / 2)



#  E[G(Z)] with Z ~ N(0,1)
def expected_G(G):
    """E[G(Z)] computed by numerical integration against the standard normal."""
    value, _ = quad(lambda v: G(v) * norm.pdf(v), -np.inf, np.inf)
    return value


def FastICA_objective_fct(theta, G, E_G):
    """Negative of the objective in Equation (FastICA)."""
    Q_theta = given_rotations(theta, K)          # Q(theta)
    V = u_tilde @ Q_theta                        # V[:, k] = q_k(theta)^T u_tilde_t
    return -sum((G(V[:, k]).mean() - E_G)**2 for k in range(K))


random_nr = np.random.default_rng(424)
bounds = [(0, 2*np.pi)] * m                      # theta in [0, 2pi)^m

results_ICA = {}
for name, G in [("log cosh", lambda v: G_1(v, a=1)),
                ("exp",      G_2)]:
    E_G = expected_G(G)
    max_ICA, vals = None, []
    for _ in range(100):
        start = random_nr.uniform(0, 2*np.pi, m)
        res = minimize(FastICA_objective_fct, start, args=(G, E_G),
                       method="L-BFGS-B", bounds=bounds)
        vals.append(-res.fun)
        if max_ICA is None or res.fun < max_ICA.fun:
            max_ICA = res

    theta_hat = max_ICA.x
    Q_ica = given_rotations(theta_hat, K)
    results_ICA[name] = {
        "theta": theta_hat,
        "Q": Q_ica,
        "B": M @ Q_ica,
        "w": u_tilde @ Q_ica,
        "objective": -max_ICA.fun,
        "E_G": E_G,
        "spread": np.round(sorted(vals)[-10:], 6),
    }
    print(f"{name:10s}: E[G(Z)] = {E_G:.4f}, objective = {-max_ICA.fun:.4f}")



pairs = list(combinations(range(K), 2))

angles = pd.DataFrame(
    {name: np.degrees(r["theta"]) for name, r in results_ICA.items()},
    index=[f"({k+1},{i+1})" for k, i in pairs]
)
print("\n--- Estimated rotation angles (degrees) ---")
print(angles.round(2))

for name, r in results_ICA.items():
    print(f"\n--- B, {name}  (objective = {r['objective']:.4f}) ---")
    print(pd.DataFrame(r["B"], index=names,
                       columns=[f"w{k+1}" for k in range(K)]).round(3))
