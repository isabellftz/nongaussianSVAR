"""Jarque-Bera test for all methods (PML, ICA)"""

import numpy as np
from scipy.stats import chi2
import pandas as pd

from prepare_data import u, u_df, Sigma_u, p
from estimate_pml import B_pml, w_pml, T, K
from estimate_ica import results_ICA

w_ica_G1 = results_ICA["log cosh"]["w"]
w_ica_G2 = results_ICA["exp"]["w"]
B_ica_G1 = results_ICA["log cosh"]["B"]
B_ica_G2 = results_ICA["exp"]["B"]

print(len(u), u_df.index[0], u_df.index[-1])
print(np.allclose(B_pml @ B_pml.T, Sigma_u))
print(np.allclose(B_ica_G1 @ B_ica_G1.T, Sigma_u))
print(np.allclose(B_ica_G2 @ B_ica_G2.T, Sigma_u))
print(np.round((w_pml**2).sum(axis=0) / (T - 1 - K*p), 4))   # should be 1
print(np.round((w_ica_G1**2).sum(axis=0) / (T - 1 - K*p), 4))   # should be 1
print(np.round((w_ica_G2**2).sum(axis=0) / (T - 1 - K*p), 4))   # should be 1
print(len(w_ica_G1))


def jb_table(w):
    """Jarque-Bera test as in Lütkepohl and Strohsal (2025), Table 1."""
    T = w.shape[0]
    z = w / w.std(axis=0, ddof=1)          # standardize to unit variance
    rows = []
    for k in range(z.shape[1]):
        sk = (z[:, k]**3).mean()
        ku = (z[:, k]**4).mean()
        jb = T/6 * sk**2 + T/24 * (ku - 3)**2
        rows.append({"shock": f"w_{k+1}", "JB": jb,
                     "p-value": 1 - chi2.cdf(jb, 2)})
    return pd.DataFrame(rows).set_index("shock").round(2)


for label, w in [("PML", w_pml),
                 ("FastICA, G1", w_ica_G1),
                 ("FastICA, G2", w_ica_G2)]:
    print(f"\n--- {label} ---")
    print(jb_table(w))
