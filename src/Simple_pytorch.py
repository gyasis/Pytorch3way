# %%
%load_ext autotime 
# %%
print(torch.__version__)
conda install pytorch-lightning -c conda-forge
# %%
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
# %%
# Dummy Data 
X = torch.randn(20,10)
y = torch.randint(0,2, (20,1)).type(torch.FloatTensor)

X.to("cuda")
y.to("cuda")

input_units = 10
hidden_units = 5
output_units = 1
# %%

model = nn.Sequential(nn.Linear(input_units, hidden_units), \
    nn.ReLU(), \
    nn.Linear(hidden_units, output_units), \
    nn.Sigmoid())

loss_function = nn.MSELoss()
# %%
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# %%
loss_funct = nn.MSELoss()

losses= []
for i in range(100):
    # Call to the model to perform a prediction
    y_pred = model(X)
    loss = loss_funct(y_pred, y)
    losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i % 5 == 0:
        print(i, loss.item())
    
    
    
    
# %%
plt.plot(range(0,100), losses)
plt.show()

# %%

import glo
