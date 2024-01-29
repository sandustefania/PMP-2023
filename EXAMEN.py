import pymc as pm
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import arviz as az
import pytensor as pt

#Subiectul 1
#a - stergerea liniilor in care Age nu este specificat
df = pd.read_csv('Titanic.csv')
df = df.dropna(subset=['Age'])
df.to_csv('modified_file.csv', index=False)

#idee asupra mediei si deviatiei standard
x_1 = df['Age'].values
x_2 = df['Pclass'].values
y = df['Survived'].values
X = np.column_stack((x_1,x_2))
X_mean = X.mean(axis=0, keepdims=True)
print("Medii:")
print(X_mean)   
print(y.mean())
print("Deviatii standard")
print(X.std(axis=0, keepdims=True))
print(y.std())

#b -  construirea modelului 
with pm.Model() as model_mlr:
    α = pm.Normal('α', mu=0, sigma=10)
    β = pm.Normal('β', mu=0, sigma=10, shape=2)
    ϵ = pm.HalfCauchy('ϵ', 50)
    ν = pm.Exponential('ν', 1/30)

    X_shared = pm.MutableData('x_shared',X) #pentru d
    μ = pm.Deterministic('μ',α + pm.math.dot(X_shared, β))
 
    y_pred = pm.StudentT('y_pred', mu=μ, sigma=ϵ, nu=ν, observed=y)

    idata_mlr = pm.sample(1000, return_inferencedata=True)

#d - aflarea posibilitatii de a supravietui a unui pasager care are 30 de ani si face parte din clasa a 2-a
pm.set_data({"x_shared":[[30,2]]}, model=model_mlr)
ppc = pm.sample_posterior_predictive(idata_mlr, model=model_mlr)
y_ppc = ppc.posterior_predictive['y_pred'].stack(sample=("chain", "draw")).values
az.plot_posterior(y_ppc,hdi_prob=0.9)

# Subiectul 2
#a
N = 10000
x, y = np.random.uniform(-1,1,size=(2,N))
condition = x > y*y
pi = condition.sum()*4/N
error = abs((pi - np.pi) / pi) * 100

