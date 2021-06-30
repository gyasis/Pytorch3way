# %%
%load_ext autotime

import icecream as ic
from pathlib import Path
import numpy as np
import pandas as pd
import os
import glob
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

# %%
working = Path.cwd()
data  = Path('/home/gyasis/Public/space/Data/vinbigdata/train/png/') 

# %%
from torch.utils.data import DataLoader

# %% 

""" this part will grab the dataframe and create paths for png files 
that were created earlier """

df = pd.read_csv('/home/gyasis/Public/space/Data/vinbigdata/train.csv')

def create_link(x):
    imagepath = '/home/gyasis/Public/space/Data/vinbigdata/train/png/'
    getlink = imagepath+ x +'.png'
    return getlink

df['link'] = df.image_id.apply(lambda x: create_link(x))
df.head(10)

# %%
df2 = df.sample(frac=0.01,     #any fraction
          replace=True, #inplace replace dataframe
          random_state=1)

df2.shape

# %%
from torch.utils.data.dataset import Dataset

class MyDataset(Dataset):
    
    
    transform = transforms.Compose([
    transforms.Resize(1800),
    transforms.CenterCrop(1000),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    
    def __init__(self, dataset, transform=transform):
        
        # self.to_tensor = transforms.ToTensor()
        self.image_arr = np.asarray(df2.link)
        self.label_arr = np.asarray(df2.class_id)
        self.data_len = len(df2.index)
        # self.to_tensor = transforms.ToTensor()
        self.transformations = transform
        
        
        
    def __getitem__(self, index):
        single_image_name = self.image_arr[index]
        img_as_img = Image.open(single_image_name)
        print(img_as_img.size)
        img_as_img = img_as_img.convert('RGB')
        imgtransform = self.transformations(img_as_img)
        print(imgtransform.size())
        # img_as_tensor = self.to_tensor(imgtransform)
        
        
        single_image_label = self.label_arr[index]
        return (imgtransform,
                single_image_label
                )
    
    def __len__(self):
        return self.data_len
# %%
datasetter = MyDataset(df2)

# %%
import torch
from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize(1800),
    transforms.CenterCrop(1000),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5,0.5], [0.5, 0.5, 0.5])
])

batchsize = 12

freeloader = DataLoader(dataset=datasetter, 
                            batch_size=batchsize,
                            shuffle=True,
                            pin_memory=True
                            
                            )
# %%
import torchvision
from torchvision import *
def imshowu(img):
    print(img)
    # img = img /2 + 0.5
    print(img)
    # npimg = img.np()
    for element in img:
        print(element)
        print(element.shape)
        # plt.imshow(img)
        # plt.imshow(np.transpose(npimg, (1,2,0)))
        # plt.show()
    



dataiter = iter(freeloader)
images, labels = dataiter.next()

imshowu(images)

print(' '.join('%5s' % labels[j] for j in range(batchsize)))    
# %%
def test(x):
    print(x)
    
print(images)
# %%

from torchvision.utils import make_grid
torchvision.utils.make_grid(images)


# %%
def show(img):
    for i in images:
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1,2,0)),interpolation='nearest')
    
show(make_grid(images))
# %%
i 
# %%
