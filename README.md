# Non-Gaussian SVAR

Replication code for the statistical identification of structural shocks in a
non-Gaussian SVAR, applied to the oil market benchmark model of Känzig (2021).
The identifying rotation is estimated by maximizing the statistical independence
of the whitened residuals, using two estimators: a parametric pseudo maximum
likelihood (PML) estimator with t-distributed pseudo densities and a non-parametric FastICA estimator.

## Data

The data come from the replication package of Känzig (2021), available at
[aeaweb.org](https://www.aeaweb.org/articles?id=10.1257/aer.20190964) under
"Replication Package". Place `dataBaseM.mat` in the repository root before
running the scripts.

## Scripts

Run in order — each script imports the results of the previous one.

| Script | Content |
| --- | --- |
| `s01_prepare_data.py` | Loads the data, estimates the reduced-form VAR(12), whitens the residuals, and tests the model assumptions |
| `s02_estimate_pml.py` | PML estimation of the rotation angles and the degrees of freedom of the pseudo densities |
| `s03_estimate_ica.py` | FastICA estimation of the rotation angles for both contrast functions |
| `s04_diagnostics.py` | Jarque–Bera tests on the estimated shocks of all three methods |

```bash
python s01_prepare_data.py
python s02_estimate_pml.py
python s03_estimate_ica.py
python s04_diagnostics.py
```

## Requirements

Python 3, with `numpy`, `pandas`, `scipy` and `statsmodels`.

## References

Känzig, D. R. (2021). The macroeconomic effects of oil supply news: Evidence
from OPEC announcements. *American Economic Review* 111(4), 1092–1125.

Lütkepohl, H. and Strohsal, T. (2025). Revisiting oil supply news shocks: Proxy
vs. non-Gaussian structural vector autoregressions. DIW Discussion Paper 2146.
