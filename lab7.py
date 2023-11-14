import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm

#sub a
df = pd.read_csv('auto-mpg.csv')

print(df)
df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
df.dropna(subset=['horsepower'], inplace=True)
df.reset_index(drop=True, inplace=True)
print(df)

# au fost sterse 6 coloane in care valorile nu erau numerice

plt.scatter(df['horsepower'], df['mpg'], alpha=0.5)
plt.xlabel('horsepower')
plt.ylabel('mpg')
plt.grid(True)
plt.show()

#sub b
x = df['horsepower'].values
y = df['mpg'].values
with pm.Model() as model:
  alfa = pm.Normal('alfa', mu=0, sigma=10)
  beta = pm.Normal('beta', mu=0, sigma=1)
  ε = pm.HalfCauchy('ε', 5)
  μ = pm.Deterministic('μ', alfa + beta * x)
  y_pred = pm.Normal('y_pred', mu=μ, sigma=ε, observed=y)
  idata_g = pm.sample(2000, tune=2000, return_inferencedata=True)   

with model:
    trace = pm.sample(2000, tune=1000)

pm.summary(trace).round(2)
plt.show()

#sub c
plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.5, label='Date observate')

for i in range(25):
    y_pred = trace['alpha'][i] + trace['beta'][i] * x
    plt.plot(x, y_pred, color='green', alpha=0.1)

plt.title('Dreapta de Regresie')
plt.xlabel('horsepower')
plt.ylabel('mpg')
plt.show()





