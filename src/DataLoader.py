# %%
%load_ext autotime
# %%
from torch.utils.data import DataLoader

# just some http cleanup
from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

# %%
from pathlib import Path
working = Path.cwd()

working.as_uri()

# %%
dataset =  '/home/gyasis/Public/GD/Google Drive/Databank/MNIST'
# %%
from torchvision.datasets import MNIST
data_train = MNIST(dataset, train=True, download=True)

print(data_train)
print(data_train[12])
# %%

# %%
#example of a dataloader
`DataLoader(  
           dataset,
           batch_size=1, #number of training samples in one iteration
           shuffle=False,
           num_workers=0,
           collate_fn=None, #if merging datasets is necessary
           pin_memory=True #if you want to load into Cuda then true
           )`
# %%
import matplotlib.pyplot as plt

random_image = data_train[0][0]
random_image_label = data_train[0][1]

plt.imshow(random_image)
print("the labe of the image is:", random_image_label)
# %%
import os
import os.path
dataset =  '/home/gyasis/Public/GD/Google Drive/Databank/MNIST'
# %%
# %%
# do the same with a dataloader


import torch
from torchvision import transforms

data_train = torch.utils.data.DataLoader(
    MNIST(dataset, train=True, download=True,
     transform = transforms.Compose([
        transforms.ToTensor()    
     ])),
    batch_size=64,
    shuffle=True   
    )

for batch_idx, samples in enumerate(data_train):
    print(batch_idx, samples)


# %%


# this checks for a gpu and loads the dataset into gpu for processing
device = "cuda" if torch.cuda.is_available() else "cpu"
kwargs = {'num_workers': 1, 'pin_memory': True} if device =='cuda' else {}

train_loader = torch.utils.data.DataLoader( 
    torchvision.datasets.MNIST(dataset, train=True, download=True),
    batch_size=batch_size_train, **kwargs)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(dataset, train=False, download=True),
    batch_size=batch_size, **kwargs)
  
  
  
# %%
# new code with cifar


transform = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5,0.5], [0.5, 0.5, 0.5])
    ## these are others that can be used 
    # torchvision.transforms.RandomCrop(  )
    #                         .RandomHorizontalFlip
    
])

trainset = torchvision.datasets.CIFAR10(root='/home/gyasis/Public/GD/Google Drive/Databank/cifar/', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=False)


# %%
classes = ('plane','car','bird','cat','deer', 'dog','frog','horse','ship','truck'  )

def imshow(img):
    img = img /2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()

dataiter = iter(trainloader)
images, labels = dataiter.next()

imshow(torchvision.utils.make_grid(images))

print(' '.join('%5s' % classes[labels[j]] for j in range(4)))    
# %%
## Creating Custom Datasets in Pytorch
# __getitem__()  returns selected sample in the dataset by indexing
# __len__() gets lenth of sample

# generic example
# class Dataset(object):
#     def __getitem__(self, index):
#         raise NotImplementedError
    
#     def __len__(self):
#         raise NotImplementedError   
# %%
# example 
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader 
from torchvision import datasets, transforms

class SquareDataset(Dataset):
    def __init__(self, a=0, b=1):
        super(Dataset, self).__init__(  )
        assert a <= b
        self.a = a
        self.b = b
        
    def __len__(self):
        return self.b - self.a + 1
        
    def __getitem__(self, index): 
        assert self.a <= index <= self.b
        return index, index **2
    
    
data_train = SquareDataset (a=1, b=64)
data_train_loader = DataLoader(data_train, batch_size=64, shuffle=True)
print(len(data_train))          
# %%
print(data_train)
# %%
