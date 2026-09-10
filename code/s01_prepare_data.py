import scipy.io
import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
from statsmodels.stats.diagnostic import acorr_ljungbox

# (1) Data preparation
mat_data = scipy.io.loadmat('dataBaseM.mat')
type(mat_data) # is dictionary
for key, value in mat_data.items(): # get data types for each key
  print(f"{key} as {type(value).__name__}")

my_matrix = mat_data['data'] # access data
names = mat_data['varNames'] # acces variable names
names = [item[0] for item in names[0]]
# 1974M1-2017M12
# POIL = OILPRICE (WTI spot crude oil price (WTISPLC) deflated by U.S.)
# OILPROD = EIA1955;
# OILSTOCK = OECDSTOCKS (OECD crude oil inventories)
# WORLDIP = OECD+6IP (Industrial production of OECD + 6)
# IP = INDPRO (U.S. industrial production index)
# CPI = CPIAUCSL (U.S. CPI for all urban consumers: all items)
# print(my_matrix)
df = pd.DataFrame(my_matrix)
df.index = pd.period_range(start="1974-01", end="2017-12", freq="M")
df.columns = names
#df = df.loc["1984-01":"2017-12"]

# (2) VAR estimation
model = VAR(df)
p = 12
results = model.fit(maxlags=p, ic=None, trend='c')
# print(results.summary())

# reduced residuals u_i,t for all i = {POIL, OILPROD, OILSTOCK, WORLDIP, IP, CPI} and t
u_df = pd.DataFrame(results.resid, columns=names)
u = u_df.values
c = results.params.iloc[0] # vector of constants
A = results.coefs # lag coefficient matrices of shape (12, 6, 6), i.e. A1,...,A12 each of dim 6x6

#check time period: 408 observations from 1984M1 to 2017M12, of which 12 are used as starting values, hence 396 residuals from 1985M1 onwards.
print(df.index[0], df.index[-1], len(df))
print(u_df.index[0], u_df.index[-1], len(u))

# (3) Whitening
T, K = u.shape # (516, 6)
Sigma_u = u.T @ u / (T - 1 - K * p) # empirical covaraince matrix with correction of degrees of freedom as in R package "svars"

M = np.linalg.cholesky(Sigma_u) # Cholesky decomposition M:=L
M_inverse = np.linalg.inv(M) # M^-1 is whitening matirx

# u_tilde
u_tilde_array =  u @ M_inverse.T # equals  (M^{-1} u_t)^T componentwise
u_tilde_df = pd.DataFrame(u_tilde_array, index=u_df.index,
                          columns=[f"u_tilde{k+1}" for k in range(K)])
u_tilde = u_tilde_df.values
#print(u_tilde_df)


# check: Sigma_u_tilde should be identity matrix
print(np.allclose(M @ M.T, Sigma_u))
print(np.round(u_tilde.T @ u_tilde / (T - 1 - K * p), 4))

# (4) u_tilde jointly Gaussian?
print(results.test_normality()) # multivariate Jarque-Bera test as in Lütkepohl (2005, Kap. 4.5)


for k in range(K):
    lb  = acorr_ljungbox(u_tilde[:, k],    lags=[12], return_df=True)
    lb2 = acorr_ljungbox(u_tilde[:, k]**2, lags=[12], return_df=True)
    print(f"$\\utildehatk_{{{k+1},t}}$ & {lb['lb_stat'].values[0]:.2f} & "
          f"{lb['lb_pvalue'].values[0]:.3f} & {lb2['lb_stat'].values[0]:.2f} & "
          f"{lb2['lb_pvalue'].values[0]:.3f} \\\\")
