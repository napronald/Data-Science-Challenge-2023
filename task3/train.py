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
DIR='/home/rnap/scratch/dsc/task3/intracardiac_dataset/' # This should be the path to the intracardiac_dataset, it can be downloaded using data_science_challenge_2023/download_intracardiac_dataset.sh
for x in os.listdir(DIR):
    if re.match(regex, x):
        data_dirs.append(DIR + x)
file_pairs = read_data_dirs(data_dirs)



batch_size = 64
torch.manual_seed(42)

# for case in range(len(file_pairs)):
#     pECGData = np.load(file_pairs[case][0])
#     for lead in range(pECGData.shape[1]):
#         pECGData[:, lead] = preprocessing.normalize([pECGData[:, lead]])[0]




class CustomImageDataset(Dataset):
    def __init__(self, annotations_file):
        self.img_labels = annotations_file

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        vector = self.img_labels[0]
        vector = torch.tensor(vector, dtype=torch.float32)

        label = self.img_labels[1]
        label = torch.tensor(label, dtype=torch.float32)
        
        return vector, label


feature = []
label = []

for case in range(len(file_pairs)):
    pECGData = np.load(file_pairs[case][0])
    pECGData = get_standard_leads(pECGData)

    for lead in range(pECGData.shape[1]):
        pECGData[:, lead] = preprocessing.normalize([pECGData[:, lead]])[0]
    
    VmData = np.load(file_pairs[case][1])
    VmData = get_activation_time(VmData)

    # for i in range(VmData.shape[1]):
        # VmData[:, i] = preprocessing.normalize([VmData[:, i]])[0]

    feature.append(pECGData)
    label.append(VmData)

data = [feature, label]
# data = [pECGData, VmData]
# data = [ [[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]] ]
cid = CustomImageDataset(data)

print(cid[0][0][0].shape)
print(cid[0][0][0])

print(cid[0][1][0].shape)
print(cid[0][1][0])

data_loader = torch.utils.data.DataLoader(
    cid,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=0,
)
