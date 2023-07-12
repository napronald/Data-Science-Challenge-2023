import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
import glob, re, os
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from cardiac_ml_tools import read_data_dirs, get_standard_leads, get_activation_time


data_dirs = []
regex = r'data_hearts_dd_0p2*'
DIR='/home/rnap/scratch/dsc/task3/intracardiac_dataset/' 
for x in os.listdir(DIR):
    if re.match(regex, x):
        data_dirs.append(DIR + x)
file_pairs = read_data_dirs(data_dirs)


batch_size = 64
torch.manual_seed(42)


# class CustomImageDataset(Dataset):
#     def __init__(self, annotations_file):
#         self.img_labels = annotations_file

#     def __len__(self):
#         return len(self.img_labels)

#     def __getitem__(self, idx):
#         vector = self.img_labels[0]
#         vector = torch.tensor(vector, dtype=torch.float32)

#         label = self.img_labels[1]
#         label = torch.tensor(label, dtype=torch.float32)
        
#         return vector, label

class CustomImageDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]

        return feature, label


# class CustomImageDataset(Dataset):
#     def __init__(self, annotations_file):
#         self.img_labels = annotations_file
#         self.transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.5], std=[0.5])
#         ])

#     def __len__(self):
#         return len(self.img_labels)

#     def __getitem__(self, idx):
#         vector = self.img_labels[idx][0]
#         vector = self.transform(vector)

#         label = self.img_labels[idx][1]
#         label = self.transform(label)

#         return vector, label



feature = []
label = []

for case in range(len(file_pairs)):
    pECGData = np.load(file_pairs[case][0])
    pECGData = get_standard_leads(pECGData)

    for lead in range(pECGData.shape[1]):
        pECGData[:, lead] = preprocessing.normalize([pECGData[:, lead]])[0]
    
    VmData = np.load(file_pairs[case][1])
    VmData = get_activation_time(VmData)

    feature.append(torch.tensor(pECGData))
    label.append(torch.tensor(VmData))


cid = CustomImageDataset(feature, label)

print(cid[0][0].shape)

print(cid[0][1].shape)

data_loader = torch.utils.data.DataLoader(
    cid,
    batch_size=2,
    shuffle=True,
    drop_last=True,
    num_workers=0,
)

for feature, label in data_loader:
    # print(feature[0][0].shape)
    # print(label[0][1].shape)
    print(feature.shape)
    print(label.shape)
    break
