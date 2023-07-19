import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        
    label.append(torch.tensor(VmData).to(device))
    feature.append(torch.tensor(pECGData, dtype=torch.float32))

feature = torch.stack(feature, dim=0)

lead_data_min = torch.min(torch.min(feature, dim=1).values, dim=0).values
lead_data_min = lead_data_min.reshape(1, 1, -1)

lead_data_max = torch.max(torch.max(feature, dim=1).values, dim=0).values
lead_data_max = lead_data_max.reshape(1, 1, -1)

feature = (feature - lead_data_min) / (lead_data_max - lead_data_min)


act_data_min = torch.min(torch.min(torch.stack(label, dim=0), dim=1).values, dim=0).values
act_data_min = act_data_min.reshape(1, 1, -1).to(device)

act_data_max = torch.max(torch.max(torch.stack(label, dim=0), dim=1).values, dim=0).values
act_data_max = act_data_max.reshape(1, 1, -1).to(device)

label = [(l - act_data_min) / (act_data_max - act_data_min) for l in label]

cid = CustomDataset(feature[:15311], label[:15311])

train_data = torch.utils.data.DataLoader(
    cid,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=0,
)

cid = CustomDataset(feature[15311:], label[15311:])

valid_data = torch.utils.data.DataLoader(
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
            nn.Conv1d(12, 1024, kernel_size=3),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=1),
            nn.Conv1d(1024, 768, kernel_size=3),
            nn.BatchNorm1d(768),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=1),
            nn.Conv1d(768, 512, kernel_size=3),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=1),
            nn.Conv1d(512, 256, kernel_size=3),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=1),
            nn.Conv1d(256, 128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=1),
            nn.Conv1d(128, 75, kernel_size=3),
            nn.BatchNorm1d(75),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=1),
        )
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  
        self.fc = nn.Linear(75, output_dim) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x


model = SqueezeNet1D(output_dim=75)

model = model.to(device)

criterion = nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=25, verbose=True, min_lr=1e-6)

def checkpoint(model, filename):
    torch.save(model.state_dict(), filename)

epochs = 500
strikes = 0
warnings = 100
lowest_valid_error = float('inf')
for epoch in range(epochs):
    train_err = 0
    train_avg_error = torch.tensor([]).to(device)
    valid_err = 0
    valid_avg_error = torch.tensor([]).to(device)
        
    model.train()
    for step, (feature, label) in enumerate(train_data):
        feature = feature.to(torch.float32).to(device)
        optimizer.zero_grad()
        y_pred = model(feature.permute(0,2,1))
        label = label.float().to(device)

        y_pred = y_pred.unsqueeze(2)
        label = label.squeeze(1)

        loss = criterion(y_pred, label)
        loss.backward()
        optimizer.step()

        y_pred = y_pred * (act_data_max - act_data_min) + act_data_min
        label = label * (act_data_max - act_data_min) + act_data_min

        # sum=0
        # for i in range(batch_size):
        #     diff = (y_pred[i, :, :] - label[i, :, :])
        #     sum += torch.sqrt(torch.norm(diff, p=2))
        # train_err = sum / batch_size
        # train_avg_error.append(train_err)

        train_avg_error = torch.cat((torch.reshape((torch.sum(abs(label - y_pred))/(batch_size*75)),(-1,)),train_avg_error), dim=0).to(device)

    scheduler.step(loss)
    train_avg_error = torch.tensor(train_avg_error)
    print(f'Epoch: {epoch+1} Loss: {loss.item()} Train Err: {float(torch.sum(train_avg_error)/len(train_avg_error))}')
    
    model.eval()
    with torch.no_grad():
        for feature, label in valid_data:
            feature = feature.to(torch.float32).to(device)
            label = label.float().to(device)

            y_pred = model(feature.permute(0, 2, 1))
            label = label.squeeze(1)
            y_pred = y_pred.unsqueeze(2)

            y_pred = y_pred * (act_data_max - act_data_min) + act_data_min
            label = label * (act_data_max - act_data_min) + act_data_min

        # sum=0
        #     for i in range(batch_size):
        #         diff = (y_pred[i, :, :] - label[i, :, :])
        #         sum += torch.sqrt(torch.norm(diff, p=2))
        #     valid_err = sum / batch_size
        #     valid_avg_error.append(valid_err)

        # valid_avg_error = torch.tensor(valid_avg_error)

            valid_avg_error = torch.cat((torch.reshape((torch.sum(abs(label - y_pred))/(batch_size*75)),(-1,)),valid_avg_error), dim=0).to(device)

        if lowest_valid_error > float(torch.sum(valid_avg_error)/len(valid_avg_error)):
            lowest_valid_error = float(torch.sum(valid_avg_error)/len(valid_avg_error))
            best_epoch = epoch
            checkpoint(model, f"best_epoch.pth")
            strikes = 0
            
        else:
            strikes += 1
            
        print(f'Valid Avg Err: {float(torch.sum(valid_avg_error)/len(valid_avg_error))}')
        
    if warnings == strikes:
        break
        
print(best_epoch)
