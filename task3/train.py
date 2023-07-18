import torch
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import r2_score
import glob, re, os
import numpy as np
from typing import List
from cardiac_ml_tools import read_data_dirs, get_standard_leads, get_activation_time


data_dirs = []
regex = r'data_hearts_dd_0p2*'
DIR='/home/rnap/scratch/dsc/task3/intracardiac_dataset/' 
for x in os.listdir(DIR):
    if re.match(regex, x):
        data_dirs.append(DIR + x)
file_pairs = read_data_dirs(data_dirs)


batch_size = 512
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
        
    VmData = np.load(file_pairs[case][1])
    VmData = get_activation_time(VmData)
        
    label.append(torch.tensor(VmData))
    feature.append(np.array(pECGData))


lead_data_min = np.min(np.min(feature, axis=1), axis=0)
lead_data_min = lead_data_min.reshape(1, 1, -1)

lead_data_max = np.max(np.max(feature, axis=1), axis=0)
lead_data_max = lead_data_max.reshape(1, 1, -1)

feature = (feature - lead_data_min) / (lead_data_max - lead_data_min)

act_data_min = torch.min(torch.min(torch.stack(label, dim=0), dim=1).values, dim=0).values
act_data_min = act_data_min.reshape(1, 1, -1)

act_data_max = torch.max(torch.max(torch.stack(label, dim=0), dim=1).values, dim=0).values
act_data_max = act_data_max.reshape(1, 1, -1)

label = [(l - act_data_min) / (act_data_max - act_data_min) for l in label]

feature = [torch.tensor(f, dtype=torch.float32) for f in feature]
label = [torch.tensor(l, dtype=torch.float32) for l in label]


cid = CustomDataset(feature[:12894], label[:12894])

train_data = torch.utils.data.DataLoader(
    cid,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=0,
)

cid = CustomDataset(feature[12894:], label[12894:])

test_data = torch.utils.data.DataLoader(
    cid,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=0,
)

class SqueezeNet1D(nn.Module):
    def __init__(self, output_dim):
        super(SqueezeNet1D, self).__init__()

        self.model = nn.Sequential(
            nn.Conv1d(500, 1024, kernel_size=3),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Conv1d(1024, 512, kernel_size=3),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 256, kernel_size=3),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 75, kernel_size=3),
            nn.BatchNorm1d(75),
            nn.ReLU(inplace=True),
        )
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  
        self.fc = nn.Linear(75, output_dim) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        # print(x.shape)
        x = self.avg_pool(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.fc(x)
        # print(x.shape)
        x = self.sigmoid(x)
        # print(x.shape)
        return x


model = SqueezeNet1D(output_dim=75)

criterion = nn.MSELoss(reduction='sum')

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

epochs = 50
for epoch in range(epochs):
    err = 0
    avg_error = []
    for step, (feature, label) in enumerate(train_data):
        feature = feature.to(torch.float32)
        optimizer.zero_grad()
        y_pred = model(feature)
        label = label.float()
        loss = criterion(y_pred.unsqueeze(2), label.squeeze(1))
        loss.backward()
        optimizer.step()

        sum=0

        y_pred = y_pred.unsqueeze(2)
        label = label.squeeze(1)

        y_pred = y_pred * (act_data_max - act_data_min) + act_data_min
        label = label * (act_data_max - act_data_min) + act_data_min

        for i in range(batch_size):
            diff = (y_pred[i, :, :] - label[i, :, :])
            sum += torch.sqrt(torch.norm(diff, p=2))
        err = sum/batch_size
        avg_error.append(err)

    scheduler.step(loss)
    avg_error = torch.tensor(avg_error)
    print(torch.sum(avg_error)/len(avg_error))
    print(f'Epoch: {epoch} Loss: {loss.item()}')


model.eval()
err = 0
avg_error = []
with torch.no_grad():
    for feature, label in test_data:
        feature = feature.to(torch.float32)
        label = label.float()

        y_pred = model(feature)
        label = label.squeeze(1)
        y_pred = y_pred.unsqueeze(2)

        y_pred = y_pred * (act_data_max - act_data_min) + act_data_min
        label = label * (act_data_max - act_data_min) + act_data_min

        sum=0
        for i in range(batch_size):
            diff = (y_pred[i, :, :] - label[i, :, :])
            sum += torch.sqrt(torch.norm(diff, p=2))
        err = sum/batch_size
        print(err)
        avg_error.append(err)

    avg_error = torch.tensor(avg_error)
    print(torch.sum(avg_error)/len(avg_error))
print(y_pred[0])
print(label[0])
