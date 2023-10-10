import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az

np.random.seed(1)

latenta = stats.expon(0,1/4)
server1 = stats.gamma(4,0,1/3) 
server2 = stats.gamma(4,0,1/2) 
server3 = stats.gamma(5,0,1/2) 
server4 = stats.gamma(5,0,1/3) 

count = 0
simulari = 1000
for i in range(simulari):
    server_ales = np.random.choice([server1,server2,server3,server4],p=[0.25,0.25,0.30,0.20])
    timp = server_ales.rvs()
    timp_total = timp + latenta.rvs()
    if timp_total>3:
        count += 1
probabilitate = count/simulari
print(probabilitate)