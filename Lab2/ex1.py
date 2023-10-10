import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az

np.random.seed(1)

mecanic1 = stats.expon.rvs(scale= 1/4, size=10000)
mecanic2 = stats.expon.rvs(scale= 1/6, size=10000)
X = 4/10*mecanic1 + 6/10*mecanic2

#media si dev. standard pentru primul mecanic
media_mecanic1 = np.mean(mecanic1)
deviatia_standard_mecanic1 = np.std(mecanic1)

#media si deviatia standard pentru al doilea mecanic
media_mecanic2 = np.mean(mecanic2)
deviatia_standard_mecanic2 = np.std(mecanic2)

#media si deviatia standard pentru X
mediaX = 4/10*media_mecanic1 + 6/10*media_mecanic2
deviatia_standard_X = 4/10*deviatia_standard_mecanic1 + 6/10*deviatia_standard_mecanic2

az.plot_posterior({'mecanic1':mecanic1,'mecanic2':mecanic2,'X':X}) 
plt.show() 
