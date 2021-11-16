# %%
%load_ext autotime 
# %%

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

print(model)

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

os.getcwd()


#getting dict after training

# Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])


# %%
model2 =  nn.Sequential(nn.Linear(input_units, hidden_units), \
    nn.ReLU(), \
    nn.Linear(hidden_units, output_units), \
    nn.Sigmoid())

model2.load_state_dict(torch.load('../mdl/testmodel.pt'))
print(model2)
model2.eval()
# %%
