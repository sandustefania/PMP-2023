import random
import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az

from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt
import networkx as nx
#a
#mai intai aflu cine incepe, dupa care in functie de cine incepe setez jucatorii. Pentru a afla
#numarul de steme de la fiecare runda, verific cine joaca si folosesc o distributie binomiala corespunzatoare.
#pentru a vedea cine castiga, compar numarul de steme obtinute de fiecare jucator. Fac asta de 1000 de ori si aflu
#probabilitatea mai mare de castiga dintre cei doi jucatori
win_j1 = 0
win_j2 = 0
for i in range(1000):
    cine_incepe = stats.binom.rvs(1,0.5, size=1)
    if cine_incepe[0]==0:
        j1 = 0
        j2 = 1
    elif cine_incepe[0]==1:
        j1 = 1
        j2 = 0
    if j1==0:
        j1_steme = stats.binom.rvs(1,0.5, size=1)
    else:
        j1_steme = stats.binom.rvs(1,2/3, size=1)
    
    if j2==0:
        j2_steme_gen = stats.binom.rvs(1,0.5,size = j1_steme[0]+1)
    else:
        j2_steme_gen = stats.binom.rvs(1,2/3,size = j1_steme[0]+1)
    
    j2_steme = 0
    for j in j2_steme_gen:
        if(j==1): j2_steme += 1
    
    print(j1_steme,j2_steme)

    if(j1_steme[0]>=j2_steme):
        win_j1 += 1
    else:
        win_j2 += 1

#print(win_j1,win_j2)

# [0] 0
# [0] 1
# [1] 1
# [1] 2
# [1] 1
# 3 2


if(win_j1>win_j2):print("Sansele mai mari le are jucatorul 1")
else: print("Sansele mai mari le are jucatorul 2")
# 558 442
#Sansele mai mari le are jucatorul 1

#b
model = BayesianNetwork([('ci','r1'),('ci','r2'),('r1','r2')])

#cine incepe jocul
cpd_ci = TabularCPD(variable='ci', variable_card=2, values=[[0.5], [0.5]]) # ci=0 incepe 0, ci =1 incepe 1

#probabilitatea de a obtine stema la prima runda bazat pe cine incepe jocul
cpd_r1_stema = TabularCPD(variable='r1', variable_card=2,
                    values=[[0.5, 1/3], [0.5, 2/3]],
                    evidence=['ci'], evidence_card=[2])  #stema = 1 

cpd_r2_stema = TabularCPD(variable='r2',variable_card=2,
                          values=[[1/3,0.5,1/3,0.5],
                                  [2/3,0.5,2/3,0.5]
                              
                          ],
                          evidence=['ci','r1'],evidence_card=[2, 2])

#c
model.add_cpds(cpd_ci, cpd_r1_stema, cpd_r2_stema)
assert model.check_model()
infer = VariableElimination(model)
prob_cine_a_inceput = infer.query(variables=['ci'], evidence={'r2': 1})
print(prob_cine_a_inceput)

pos = nx.circular_layout(model)
nx.draw(model, pos=pos, with_labels=True, node_size=4000, font_weight='bold', node_color='yellow')
plt.show()