import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

class HousePricePredictor(nn.Module):
    def __init__(self, D_in, H1, H2, H3, D_out):
        super(HousePricePredictor, self).__init__()
        
        self.linear1 = nn.Linear(D_in, H1)
        self.linear2 = nn.Linear(H1, H2)
        self.linear3 = nn.Linear(H2, H3)
        self.linear4 = nn.Linear(H3, D_out)
        
    def forward(self, x):
        y_pred = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(y_pred).clamp(min=0)
        y_pred = self.linear3(y_pred).clamp(min=0)
        y_pred = self.linear4(y_pred)
        return y_pred

yes_no_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
yes_no_conversion = {'yes': 1, 'no': 0}
furnishing_conversion = {'furnished': 3, 'semi-furnished': 2, 'unfurnished': 1}

df = pd.read_csv('Housing.csv', delimiter=',')
for column in yes_no_columns:
    df[column] = df[column].map(yes_no_conversion)
df['furnishingstatus'] = df['furnishingstatus'].map(furnishing_conversion)

df = (df - df.mean()) / (df.max() - df.min())
y = df.iloc[:,0]
X = df.iloc[:,1:]

X = torch.tensor(X.values, dtype=torch.float32)
y = torch.tensor(y.values, dtype=torch.float32).reshape(-1, 1)

H1, H2, H3 = 500, 1000, 200
D_in, D_out = X.shape[1], y.shape[1]
model = HousePricePredictor(D_in, H1, H2, H3, D_out)

loss_fn = nn.MSELoss(reduction='sum')
optimizer = optim.Adam(model.parameters(), lr=1e-4 * 2)

means, maxes, mins = dict(), dict(), dict()
for col in df:
    means[col] = df[col].mean()
    maxes[col] = df[col].max()
    mins[col] = df[col].min()

losses = []

for t in range(500):
    y_pred = model(X)
    
    loss = loss_fn(y_pred, y)
    print(t, loss.item())
    losses.append(loss.item())
    
    if torch.isnan(loss):
        break
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


