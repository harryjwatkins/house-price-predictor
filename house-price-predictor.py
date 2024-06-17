import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

class PimaClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(12, 12)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(12, 12)
        self.act2 = nn.ReLU()
        self.output = nn.Linear(12, 1)
        self.act_output = nn.Sigmoid()
 
    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.act_output(self.output(x))
        return x
    
model = PimaClassifier()
print(model)

yes_no_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
yes_no_conversion = {'yes': 1, 'no': 0}
furnishing_conversion = {'furnished': 3, 'semi-furnished': 2, 'unfurnished': 1}

df = pd.read_csv('Housing.csv', delimiter=',')
for column in yes_no_columns:
    df[column] = df[column].map(yes_no_conversion)
df['furnishingstatus'] = df['furnishingstatus'].map(furnishing_conversion)

y = df.iloc[:,0]
X = df.iloc[:,1:-1]

X = torch.tensor(X.values, dtype=torch.float32)
y = torch.tensor(y.values, dtype=torch.float32).reshape(-1, 1)
