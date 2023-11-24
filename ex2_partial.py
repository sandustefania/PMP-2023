import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import arviz as az
import pymc as pm
#1.
#generez 100 de timpi medii de asteptare folosind o distributie poissson
timpi_de_asteptare = stats.poisson.rvs(20.0, size=100)
for x in timpi_de_asteptare:
    alfa = pm.Normal('alfa', mu=0, sigma=10)
    beta = pm.Normal('beta', mu=0, sigma=1)
    timpi_generati = pm.Deterministic('niu', x * beta + alfa)

#2.
    with pm.Model() as model_regression: 
      timpi_de_asteptare = stats.poisson.rvs(20.0, size=100)
      for x in timpi_de_asteptare:       
        alfa = pm.Normal('alfa', mu=0, sigma=10)
        beta = pm.Normal('beta', mu=0, sigma=1)
        eps = pm.HalfCauchy('eps', 5)
        niu = pm.Deterministic('niu', x * beta + alfa)
        idata = pm.sample(2000, tune=2000, return_inferencedata=True)

    az.plot_trace(idata, var_names=['alfa', 'beta', 'eps'])
    plt.show()