import pymc as pm
import numpy as np
import pandas as pd
import pytensor as pt
import scipy.stats as stats
from scipy.special import expit as logistic
import matplotlib.pyplot as plt
import arviz as az
az.style.use('arviz-darkgrid')

admission = pd.read_csv('./Admission.csv')
admission.head()

df = admission.query("Admission == ('0', '1')")
y_1 = pd.Categorical(df['Admission']).codes
x_n = ['GRE', 'GPA']
x_1 = df[x_n].values

with pm.Model() as model:
  alfa = pm.Normal('alfa', mu=0, sigma=10)
  beta = pm.Normal('beta', mu=0, sigma=2, shape=len(x_n))

  μ = alfa + pm.math.dot(x_1, beta)
  θ = pm.Deterministic('θ', 1 / (1 + pm.math.exp(-μ)))
  bd = pm.Deterministic('bd', -alfa/beta[1] - beta[0]/beta[1] * x_1[:,0])

  yl = pm.Bernoulli('yl', p=θ, observed=y_1)

  idata = pm.sample(2000, return_inferencedata=True)


idx = np.argsort(x_1[:,0])
bd = idata.posterior['bd'].mean(("chain", "draw"))[idx]
plt.scatter(x_1[:,0], x_1[:,1], c=[f'C{x}' for x in y_1])
plt.plot(x_1[:,0][idx], bd, color='k')
az.plot_hdi(x_1[:,0], idata.posterior['bd'], color='k')
plt.xlabel(x_n[0])
plt.ylabel(x_n[1])