import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
from sklearn import preprocessing
import glob, re, os
import numpy as np
from typing import List
from cardiac_ml_tools import read_data_dirs, get_standard_leads, get_activation_time
import warnings
warnings.filterwarnings("ignore")

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
_max = []

for case in range(len(file_pairs)):
    pECGData = np.load(file_pairs[case][0])
    pECGData = get_standard_leads(pECGData)
        
    VmData = np.load(file_pairs[case][1])
    VmData = get_activation_time(VmData)
    
    _max.append(np.max(VmData))
    
    label.append(torch.tensor(VmData))
    feature.append(np.array(pECGData))
        
feature = np.array(feature)
    
lead_data_min = np.min(np.min(feature, axis=1), axis=0)
lead_data_min = lead_data_min.reshape(1, 1, -1)

lead_data_max = np.max(np.max(feature, axis=1), axis=0)
lead_data_max = lead_data_max.reshape(1, 1, -1)

feature = (feature - lead_data_min) / (lead_data_max - lead_data_min)

# label = np.array(label)

# act_data_min = np.min(np.min(label, axis=1), axis=0)
# act_data_min = act_data_min.reshape(1, 1, -1)

# act_data_max = np.max(np.max(label, axis=1), axis=0)
# act_data_max = act_data_max.reshape(1, 1, -1)

# label = (feature - act_data_min) / (act_data_max - act_data_min)

feature = list(feature)
for i in range(len(feature)):
    feature[i] = torch.tensor(feature[i], dtype=torch.float32)

# label = list(label)    

# torch.tensor(feature)
    
cid = CustomDataset(feature, label)

data_loader = torch.utils.data.DataLoader(
    cid,
    batch_size=512,
    shuffle=True,
    drop_last=True,
    num_workers=0,
)


# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.model = nn.Sequential(
#             nn.Conv1d(12, 75, kernel_size=500),
#             nn.BatchNorm1d(75),
#             nn.ReLU(),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         decoded = self.model(x)
#         # print(decoded.shape)
#         return decoded


# model = CNN()

class SqueezeNet1D(nn.Module):
    def __init__(self, output_dim):
        super(SqueezeNet1D, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(12, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(16, output_dim, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    

model = SqueezeNet1D(output_dim=75)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 250
for epoch in range(epochs):
    for step, (feature, label) in enumerate(data_loader):
        feature = feature.to(torch.float32)
        # print(feature.shape)
        optimizer.zero_grad()
        y_pred = model(feature.permute(0, 2, 1))
        y_pred = y_pred.unsqueeze(2)
        # print(y_pred.shape)
        # print(label.shape)
        label = label.float()
        loss = criterion(y_pred, label/200)
        loss.backward()
        optimizer.step()
        y_pred_np = y_pred.detach().numpy().reshape(-1)
        label_np = label.detach().numpy().reshape(-1)
        r2 = r2_score(label_np/200, y_pred_np)
    print(r2)

    print(f'Epoch: {epoch} loss: {loss.item()}')
