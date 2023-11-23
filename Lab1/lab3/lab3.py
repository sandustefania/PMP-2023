from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt
import networkx as nx

# Defining the model structure. We can define the network by just passing a list of edges.
model = BayesianNetwork([('C', 'I'), ('C', 'A'),('i', 'A')])

# Defining individual CPDs.
cpd_c = TabularCPD(variable='C', variable_card=2, values=[[0.9995], [0.0005]]) 
cpd_i = TabularCPD(variable='I', variable_card=2, values=[[0.99], [0.01]]) 
cpd_a = TabularCPD(variable='A', variable_card=2, values=[[0.9999], [0.0001]]) 

# The CPD for I is defined using the conditional probabilities based on C
cpd_ic = TabularCPD(variable='I', variable_card=2, 
                   values=[[0.99, 0.01], 
                           [0.97, 0.03]],
                  evidence=['C'],
                  evidence_card=[2])

# The CPD for A is defined using the conditional probabilities based on C
cpd_ac = TabularCPD(variable='A', variable_card=2, 
                   values=[[0.9999, 0.0001], 
                           [0.98, 0.02]],
                  evidence=['C'],
                  evidence_card=[2])

# The CPD for A is defined using the conditional probabilities based on I
cpd_ai = TabularCPD(variable='A', variable_card=2, 
                   values=[[0.9999, 0.0001], 
                           [0.05, 0.95]],
                  evidence=['I'],
                  evidence_card=[2])

# The CPD for A is defined using the conditional probabilities based on C and I
cpd_aci = TabularCPD(variable='A', variable_card=2, 
                   values=[[0.9999, 0.02, 0.95, 0.98], 
                           [0.0001, 0.98, 0.05, 0.02]],
                  evidence=['C', 'I'],
                  evidence_card=[2, 2])

# Associating the CPDs with the network
model.add_cpds(cpd_c, cpd_i, cpd_a,cpd_ic,cpd_ac,cpd_ai,cpd_aci)

# Verifying the model
assert model.check_model()

# Performing exact inference using Variable Elimination - sub 2
infer = VariableElimination(model)
result = infer.query(variables=['C'], evidence={'A': 1})
print(result)

# Performing exact inference using Variable Elimination - sub 3
infer = VariableElimination(model)
result = infer.query(variables=['I'], evidence={'A': 0})
print(result)

pos = nx.circular_layout(model)
nx.draw(model, pos=pos, with_labels=True, node_size=4000, font_weight='bold', node_color='skyblue')
plt.show()