# %% 
# Module imports
import numpy as np 
import torch 
import monai
from torchvision.datasets import CIFAR10
from monai.data import Dataset
from monai.engines import SupervisedTrainer
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

# %% 
#Dataset build (loading from Torchvison) and  split data

cifar10 = CIFAR10(root=".", train=True, download=True)
train_data, val_data = train_test_split(cifar10, test_size=0.2)

# %%
from monai.transforms import Compose, LoadImage, ToTensor, Lambda, Transpose

# Create a custom dataset
class MyDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __getitem__(self, index):
        image, label = self.data[index]
        
        # Load image
        # Image is already loaded from the previous dataset, now we just need to convert first to a numpy array for monai transforms to work
        image = np.array(image)
        
        # Apply transformation
        if self.transform:
            image = self.transform(image)

        return image, label



# %% 

# Define your normalize function
def normalize(data):
    data = data.float()
    mean = torch.mean(data)
    std = torch.std(data)
    return (data - mean) / std



# Use your normalize function in a composed transform
transform = Compose([
    ToTensor(),
    Lambda(lambda x: normalize(x)),
    Transpose((2,0,1))
    
])

vtransform = Compose([
    ToTensor(),
    Lambda(lambda x: x.float()),
    Transpose((2,0,1))
])

# wrap the PyTorch dataset in a MONAI dataset


# %%
train = MyDataset(train_data, 
                transform=transform
                )
val = MyDataset(val_data, 
              transform=vtransform
              )
# %%
def get_num_workers():
    if torch.cuda.is_available():
        # set a higher value for num_workers if there is a GPU
        num_workers = 4
    else:
        # set a lower value for num_workers if there is no GPU
        num_workers = 0
    return num_workers

train_loader = DataLoader(train, batch_size=32, shuffle=True, num_workers=get_num_workers())
val_loader = DataLoader(val, batch_size=32, shuffle=True, num_workers=get_num_workers())

# %%
#Build Model Architecture

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)   # add dropout layer
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

# %%
model = SimpleCNN()

# move the model and data to the GPU if available
if torch.cuda.is_available():
    model = model.to("cuda")
    train_loader = train_loader.to("cuda")
    val_loader = val_loader.to("cuda")
# %%
# define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), weight_decay=1e-5)
# %%
# Build the trainer torch style
import tqdm
# %% 
for epoch in range(15):
    train_loss = 0.0
    for i, (inputs, labels) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    print(f"Epoch {epoch+1}, train loss: {train_loss/(i+1):.4f}")

    # Validation
    val_loss = 0.0
    for i, (inputs, labels) in tqdm.tqdm(enumerate(val_loader), total=len(val_loader)):
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    print(f"Epoch {epoch+1}, val loss: {val_loss/(i+1):.4f}")

# %% 
from monai.engines import SupervisedTrainer

# create the trainer
trainer = SupervisedTrainer(
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    max_epochs=10,
    optimizer=optimizer,
    loss_function=criterion,
    train_data_loader=train_loader,
   
    network=model
)

# train the model
trainer.run()

# %%
from sklearn.metrics import f1_score
import torch

# %%
def compute_f1_score(y_pred, y):
    y_pred = y_pred.argmax(dim=1).flatten().cpu().numpy()
    y = y.argmax(dim=1).flatten().cpu().numpy()
    return torch.tensor(f1_score(y, y_pred, average='macro'))

evaluator = SupervisedEvaluator(
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    val_data_loader=val_loader,
    network=model,
    
    key_val_metric={"f1_score": compute_f1_score}
)

# run evaluation
state = evaluator.run()

# print the final results
print(f"Validation Results - Epoch: {state.epoch} Avg F1 score: {state.metrics['f1_score']:.4f}")



# %%
from monai.metrics import DiceMetric
from monai.engines import SupervisedEvaluator
from ignite.engine import Events
from monai.metrics import DiceMetric
from monai.engines import SupervisedEvaluator

# create the evaluator
evaluator = SupervisedEvaluator(
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    val_data_loader=val_loader,
    network=model,
)

# create the metric
metric = DiceMetric(include_background=True, reduction="mean")

# attach the metric to the evaluator
@evaluator.on(Events.ITERATION_COMPLETED(every=50))
def print_shape(engine):
    y_pred = engine.state.output["pred"]
    y = engine.state.output["label"]
    print("y_pred shape:", y_pred.shape)
    print("y shape:", y.shape)

evaluator.add_event_handler(Events.ITERATION_COMPLETED, metric)

# run evaluation
state = evaluator.run()

# print the final results
print(f"Validation Results - Epoch: {state.epoch} Avg dice score: {state.metrics['dice']:.4f}")




# %%
import seaborn as sns
import matplotlib.pyplot as plt

train_losses = []
val_losses = []
val_accuracies = []

for epoch in range(5):
    train_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_losses.append(train_loss/(i+1))

    # Validation
    evaluator = SupervisedEvaluator(
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        val_data_loader=val_loader,
        network=model,
        key_val_metric={"accuracy": Accuracy(output_transform=lambda x: (x["pred"], x["label"]))}
    )

    state = evaluator.run()
    val_loss = val_loss/(i+1)
    val_losses.append(val_loss)
    val_accuracies.append(state.metrics['accuracy'])

    print(f"Epoch {epoch+1}, train loss: {train_loss/(i+1):.4f}, val loss: {val_loss:.4f}, val accuracy: {state.metrics['accuracy']:.4f}")

sns.set()
sns.lineplot(x=range(1, 6), y=train_losses, label='Train Loss')
sns.lineplot(x=range(1, 6), y=val_losses, label='Val Loss')
sns.lineplot(x=range(1, 6), y=val_accuracies, label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Metric')
plt.title('Training and Validation Metrics')
plt.legend()
plt.show()


# %%
from sklearn.metrics import accuracy_score

for epoch in range(5):
    train_loss = 0.0
    train_acc = 0.0
    for i, (inputs, labels) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_acc += accuracy_score(labels.cpu().numpy(), outputs.argmax(1).cpu().numpy())
    print(f"Epoch {epoch+1}, train loss: {train_loss/(i+1):.4f}, train acc: {train_acc/(i+1):.4f}")

    # Validation
    val_loss = 0.0
    val_acc = 0.0
    for i, (inputs, labels) in tqdm.tqdm(enumerate(val_loader), total=len(val_loader)):
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_acc += accuracy_score(labels.cpu().numpy(), outputs.argmax(1).cpu().numpy())
    print(f"Epoch {epoch+1}, val loss: {val_loss/(i+1):.4f}, val acc: {val_acc/(i+1):.4f}")

# %%
import seaborn as sns
import matplotlib.pyplot as plt

train_losses = []
val_losses = []

for epoch in range(20):
    train_loss = 0.0
    for i, (inputs, labels) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss = train_loss / (i+1)
    train_losses.append(train_loss)
    print(f"Epoch {epoch+1}, train loss: {train_loss:.4f}")

    # Validation
    val_loss = 0.0
    for i, (inputs, labels) in tqdm.tqdm(enumerate(val_loader), total=len(val_loader)):
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    val_loss = val_loss / (i+1)
    val_losses.append(val_loss)
    print(f"Epoch {epoch+1}, val loss: {val_loss:.4f}")

# plot the loss and accuracy graphs side by side
sns.set_style("darkgrid")
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

axs[0].plot(train_losses, label="Train Loss")
axs[0].plot(val_losses, label="Validation Loss")
axs[0].set_xlabel("Epoch")
axs[0].set_ylabel("Loss")
axs[0].legend()

axs[1].plot(train_accs, label="Train Accuracy")
axs[1].plot(val_accs, label="Validation Accuracy")
axs[1].set_xlabel("Epoch")
axs[1].set_ylabel("Accuracy")
axs[1].legend()

plt.show()


# %%
