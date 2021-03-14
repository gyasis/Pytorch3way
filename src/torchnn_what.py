# %%

# %%
from pathlib import Path
import requests

DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)

URL = "https://github.com/pytorch/tutorials/raw/master/_static/"
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():
        content = requests.get(URL + FILENAME).content
        (PATH / FILENAME).open("wb").write(content)
    
    
# %%
import pickle
import gzip

with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")
# %%
from matplotlib import pyplot
import numpy as np

pyplot.imshow(x_train[0].reshape((28,28)), cmap="gray")
print(x_train.shape)


# %%
import torch

x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)
n, c = x_train.shape
x_train, x_train.shape, y_train.min(), y_train.max()
print(x_train, y_train)
print(x_train.shape)
print(y_train.min(), y_train.max())

# %%
### neural network from scratch
import math

weights = torch.randn(784, 10) / math.sqrt(784)
weights.requires_grad_()
bias = torch.zeros(10, requires_grad=True)




# %%
def log_softmax(x):
    return x -x.exp().sum(-1).log().unsqueeze(-1)

def model(xb):
    return log_softmax(xb @ weights + bias)

# %%
bs = 64
xb = x_train[0:64]
preds = model(xb)
preds[0], preds.shape
print(preds[0], preds.shape)

# %%
def nll(input, target):
    return -input[range(target.shape[0]), target].mean(-1)

loss_func = nll

yb= y_train[0:bs]
print(loss_func(preds, yb))
# %%
def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()

print(accuracy(preds, yb))
# %%
from IPython.core.debugger import set_trace
# %%
lr = 0.5
epochs = 2 

for epoch in range(epochs):
    for i in range((n-1) // bs +1):
        start_i = i * bs
        end_i = start_i + bs
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        pred = model(xb)
        loss = loss_func(pred, yb)
        
        loss.backward()
        with torch.no_grad():
            weights -= weights.grad * lr
            bias -= bias.grad * lr
            weights.grad.zero_()
            bias.grad.zero_()
            
# %%
print(loss_func(model(xb), yb), accuracy(model(xb), yb))
# %%
# Now with torch.nn

import torch.nn.functional as F

loss_func = F.cross_entropy

def model(xb):
    return xb @ weights + bias

print(loss_func(model(xb), yb), accuracy(model(xb), yb))
    # %%
    
from torch import nn

class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(784, 10) / math.sqrt(784))
        self.bias = nn.Parameter(torch.zeros(10))

    def forward(self, xb):
        return xb @ self.weights + self.bias    
# %%
model = Mnist_Logistic()
# %%
print(loss_func(model(xb), yb))
# %%
# previous loop 


# for epoch in range(epochs):
#     for i in range((n-1) // bs +1):
#         start_i = i * bs
#         end_i = start_i + bs
#         xb = x_train[start_i:end_i]
#         yb = y_train[start_i:end_i]
#         pred = model(xb)
#         loss = loss_func(pred, yb)
        
#         loss.backward()
#         with torch.no_grad():
#             weights -= weights.grad * lr
#             bias -= bias.grad * lr
#             weights.grad.zero_()
#             bias.grad.zero_()


def fit():
    for epoch in range(epochs):
        for i in range((n-1) // bs +1):
            start_i = i * bs
            end_i = start_i + bs
            xb = x_train[start_i:end_i]
            yb = y_train[start_i:end_i]
            pred = model(xb)
            loss = loss_func(pred, yb)
            
            loss.backward()
            with torch.no_grad():
                for p in model.parameters():
                    p -= p.grad * lr
                model.zero_grad()    
            
fit()
# %%
print(loss_func(model(xb), yb))
# %%
class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(784,10)
        
    def forward(self, xb):
        return self.lin(xb)
# %%
model = Mnist_Logistic()
print(loss_func(model(xb), yb))
# %%
fit()
print(loss_func(model(xb),yb))
# %%
from torch import optim

def get_model():
    model = Mnist_Logistic()
    return model, optim.SGD(model.parameters(), lr=lr)

model, opt  = get_model()
print(loss_func(model(xb), yb))

for epoch in range(epochs):
    for i in range((n-1) // bs + 1):
        start_i = i * bs 
        end_i = start_i + bs 
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        pred = model(xb)
        loss = loss_func(pred, yb)
        
        loss.backward()
        opt.step()
        opt.zero_grad()
        
print(loss_func(model(xb), yb))        
# %%
from torch.utils.data import TensorDataset

train_ds = TensorDataset(x_train, y_train)
# %%
model, opt = get_model()

for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
        xb, yb =train_ds[i * bs: i * bs + bs]
        pred = model(xb)
        loss = loss_func(pred, yb)
        loss.backward()
        opt.step()
        opt.zero_grad()
        
        
print(loss_func(model(xb), yb))        
        
              
# %%
from torch.utils.data import DataLoader

train_d1 = DataLoader(train_ds, batch_size=bs)

for xb, yb in train_d1:
    pred = model(xb)
    
model, opt = get_model()
for epoch in range(epochs):
    for xb, yb in train_d1:
        pred = model(xb)
        loss = loss_func(pred, yb)
        
        loss.backward()
        opt.step()
        opt.zero_grad()
        
print(loss_func(model(xb), yb))
# %%
train_ds = TensorDataset(x_train, y_train)
train_