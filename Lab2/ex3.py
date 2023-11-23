import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az

np.random.seed(1)
def arunca_doua_monede():
    aruncari = np.random.choice(['s', 'b'], size=2, p=[0.3, 0.7])
    rezultat = ''.join(aruncari)
    return rezultat

rez = []
ss = 0
sb = 0
bs = 0
bb = 0
for i in range(100):
    rez = arunca_doua_monede()
    if rez == 'ss':
        ss += 1
    elif rez == 'sb':
        sb += 1
    elif rez =='bs':
        bs += 1
    else:
        bb += 1


etichete = ['ss', 'sb', 'bs', 'bb']
valori = [ss, sb, bs, bb]

plt.bar(etichete, valori)
plt.xlabel('Rezultate')
plt.ylabel('Numar de aparitii')
plt.title('Distributia rezultatelor în 100 de experimente')
plt.show()
