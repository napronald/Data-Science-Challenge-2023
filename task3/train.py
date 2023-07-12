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

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]

        return feature, label


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


cid = CustomDataset(feature, label)


data_loader = torch.utils.data.DataLoader(
    cid,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=0,
)


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.to(torch.float32)  # Convert input tensor to double precision
        out, (hn, cn) = self.rnn(x)
        out = self.fc(hn[-1])
        return out


model = LSTMClassifier(input_dim=12, hidden_dim=128, layer_dim=1 ,output_dim=75)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100


for epoch in range(1, num_epochs):
    loss_epoch = 0
    for step, (feature, label) in enumerate(data_loader):
        label = label.float()
        optimizer.zero_grad()
        logits = model(feature)
        
        loss = criterion(logits.unsqueeze(2), label)
        loss.backward()
        optimizer.step()

        if step % 250 == 0:
            print(
                f"Step [{step}/{len(data_loader)}]\t loss_instance: {loss.item()}")
        loss_epoch += loss.item()

    print(f"Epoch [{epoch}/{num_epochs}]\t Loss: {loss_epoch / len(data_loader)}")
