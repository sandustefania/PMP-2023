import numpy as np
import arviz as az
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy import stats

clusters = 2
n_cluster = [200, 150,150]
n_total = sum(n_cluster)
means = [5, 0, 1]
std_devs = [2, 2, 2]
mix = np.random.normal(np.repeat(means, n_cluster),
np.repeat(std_devs, n_cluster))
az.plot_kde(np.array(mix))

plt.hist(mix, bins=30, density=True, alpha=0.7, color='blue')
plt.title('Histograma Mixturii de 3 Distribuții Gaussiene')
plt.xlabel('Valori')
plt.ylabel('Frecvență')
plt.show()

components = [2, 3, 4]

models = [GaussianMixture(n, random_state=0).fit(mix.reshape(-1, 1)) for n in components]

bics = [model.bic(mix.reshape(-1, 1)) for model in models]

for n, bic in zip(components, bics):
    print(f"Număr de componente: {n}, Scor BIC: {bic}")

plt.plot(components, bics, marker='o')
plt.title('Scorul BIC în funcție de numărul de componente')
plt.xlabel('Număr de componente')
plt.ylabel('Scor BIC')
plt.show()
