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

for case in range(len(file_pairs)):
    pECGData = np.load(file_pairs[case][0])
    pECGData = get_standard_leads(pECGData)
        
    VmData = np.load(file_pairs[case][1])
    VmData = get_activation_time(VmData)
    
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
    batch_size=64,
    shuffle=True,
    drop_last=True,
    num_workers=0,
)


class LSTMCNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim):
        super(LSTMCNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.cnn = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3)
 
    def forward(self, x):
        out, (hn, cn) = self.rnn(x)
        print(out.shape)
        out = out.unsqueeze(1)
        decoded = self.cnn(out)
        return decoded

model = LSTMCNN(input_dim=6000, hidden_dim=77, layer_dim=5)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 10
for epoch in range(epochs):
    for step, (feature, label) in enumerate(data_loader):
        feature = feature.reshape(-1, 12*500)
        feature = feature.to(torch.float32)

        optimizer.zero_grad()
        print(feature.shape)
        y_pred = model(feature)

        label = label.reshape(-1, 75*1)
        label = label.float()
        loss = criterion(150*y_pred, label)

        loss.backward()
        optimizer.step()

    print(f'Epoch: {epoch} {loss.item()}')
