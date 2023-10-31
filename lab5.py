import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

data = pd.read_csv('trafic.csv')
model = pm.Model()

with model:
    lambda_47 = pm.Exponential("lambda_47", lam=1)
    lambda_78 = pm.Exponential("lambda_78", lam=1)
    lambda_816 = pm.Exponential("lambda_816", lam=1)
    lambda_1619 = pm.Exponential("lambda_1619", lam=1)
    lambda_1924 = pm.Exponential("lambda_1924", lam=1)

    traffic_47 = pm.Poisson("traffic_47", mu=lambda_47, observed=data[0:180])
    traffic_78 = pm.Poisson("traffic_78", mu=lambda_78, observed=data[180:240])
    traffic_816 = pm.Poisson("traffic_816", mu=lambda_816, observed=data[240:720])
    traffic_1619 = pm.Poisson("traffic_1619", mu=lambda_1619, observed=data[720:900])
    traffic_1924 = pm.Poisson("traffic_1924", mu=lambda_1924, observed=data[900:])

with model:
    trace = pm.sample(100)