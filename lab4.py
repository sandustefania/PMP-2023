import numpy as np
import scipy.stats as stats

lambdaa = 20  
timp_plasare_si_plata = 2  
deviatie_standard_comanda = 0.5  

poisson_dist = stats.poisson(lambdaa)
normal_dist = stats.norm(timp_plasare_si_plata, deviatie_standard_comanda)
exp_dist = stats.expon.rvs(0, 15, size=20)

max_alfa = stats.expon.ppf(0.95, scale=15)

#timpul mediu de asteptare pentru a fi servit un client
s = 0
for i in exp_dist:
    s = s+i
s = s/20
print(s)