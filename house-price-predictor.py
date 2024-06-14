import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

df = pd.read_csv('Housing.csv', delimiter=',')
y = df.iloc[:,0]
X = df.iloc[:,1:-1]

X = torch.tensor(X.values, dtype=torch.float32)
y = torch.tensor(y.values, dtype=torch.float32).reshape(-1, 1)

print(X)
print(y)
