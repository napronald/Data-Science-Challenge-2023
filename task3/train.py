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


# class Arch(nn.Module):
#     def __init__(self, input_dim=6000, hidden_dim=77, layer_dim=5):
#         super(Arch, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.layer_dim = layer_dim
#         self.rnn = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
#         self.cnn = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3)
#         self.sigmoid = nn.Sigmoid()
#         # self.batch = nn.BatchNorm1d()
 
#         self.model = nn.Sequential(
#             nn.Conv1d(1 ,75, kernel_size=3),
#             nn.BatchNorm1d(75),
#             nn.ReLU(),
#             nn.Sigmoid()
#         )


#     def forward(self, x):
#         # out, (hn, cn) = self.rnn(x)
#         # decoded = self.cnn(x)
#         # decoded = self.sigmoid(decoded)
#         decoded = self.model(x)
#         print(decoded.shape)
#         return decoded

# model = Arch()

# lr = 0.001
# weight_decay = 0
# epochs = 10
# loss_fn = nn.MSELoss()

# optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

# for epoch in range(epochs + 1):
#     for step, (X_train, y_train) in enumerate(data_loader):
#         # print(X_train.shape)
#         X_train = X_train.reshape(-1, 12*500)
#         # print(X_train.shape)
#         X_train = X_train.to(torch.float32)
#         optimizer.zero_grad()
#         y_pred = model(X_train.unsqueeze(1))
#         y_train = y_train.reshape(-1, 75*1)
#         y_train = y_train.unsqueeze(1)
#         # print(y_train.shape)
#         y_train = y_train.float()
#         loss = loss_fn(y_pred*200, y_train)
#         # print(y_pred.shape, y_train.shape)
#         loss.backward()
#         optimizer.step()
#     print('-----------------------------')
#     print(f'Epoch: {epoch}')
#     print(f'Loss: {loss.item()}')
#     # print(f'Accuracy: {}')



class LSTMCNN(nn.Module):
    def __init__(self):
        super(LSTMCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(12, 75, kernel_size=500),
            nn.BatchNorm1d(75),
            nn.ReLU(),
            nn.Sigmoid()
        )

    def forward(self, x):
        decoded = self.model(x)
        # print(decoded.shape)
        return decoded


model = LSTMCNN()

lr = 0.001
weight_decay = 0
criterion = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


epochs = 100
for epoch in range(epochs):
    for step, (feature, label) in enumerate(data_loader):
        feature = feature.permute(0, 2, 1)  # Permute dimensions for input tensor
        feature = feature.to(torch.float32)

        optimizer.zero_grad()
        y_pred = model(feature)

        label = label.float()
        loss = criterion(y_pred, label/200)

        loss.backward()
        optimizer.step()

    print(f'Epoch: {epoch} {loss.item()}')
