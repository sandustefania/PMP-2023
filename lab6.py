import pymc as pm
import numpy as np
import arviz as az

Y = [0, 5, 10]
T = [0.2, 0.5]

d = {
    0.2: [0, 5, 10],  
    0.5: [0, 5, 10],  
}

with pm.Model() as model:
    n = pm.Poisson("n", mu=10)
    for t in T:
        Y = d[t]
        Y_obs = pm.Binomial("Y_obs", n=n, p=t, observed=Y)
        
    trace = pm.sample(1000, tune=100, cores=2)

az.plot_posterior(trace, var_names=["n"], credible_interval=0.95)

#OBSERVATII
#Daca numarul de clienti care cumpara este mai mare, atunci distributia va varia mai putin si se va restrange la un interval mic de valori
#Daca probabilitatea de a cumpara produsul este mai mare, atunci distributia va fi concentratata in jurul unei valori mai mari