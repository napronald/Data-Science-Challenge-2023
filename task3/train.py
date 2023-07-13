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

    for i in range(12):
        print(pECGData[:, i].shape)
        print(pECGData[:, i])

        min = np.min(pECGData[:,i])
        max = np.max(pECGData[:,i])
        print(min)
        print(max)

        pECGData[:,i] = (pECGData[:,i] - min) / (max-min)
        print(pECGData[:,i].shape)
        std = np.std(pECGData)
        mean = np.mean(pECGData)
        # print(std)
        # print(mean)

    # VmData = ( ( (pECGData - mean) / std) + 1) / 2

    label.append(torch.tensor(VmData))
    feature.append(np.array(pECGData))
    
feature = np.array(feature)

# for lead in range(12):
#     lead_data = feature[:][:][lead]
#     mean = np.mean(lead_data)
#     std = np.std(lead_data)
#     feature[:][:][lead] = ( ( (lead_data - mean) / std) + 1) / 2


for i in range(12):
    feature[:][i][:] = ( (x[:][i][:] - np.min(feature[:][i][:])) / (np.max(feature[:][i][:]) - np.min(feature[:][i][:])) )
    
    print(feature.shape)
    # mean = np.mean(lead_data)
    # std = np.std(lead_data)
    # feature[:][:][lead] = ( ( (lead_data - mean) / std) + 1) / 2

# feature = list(feature)
# for i in range(len(feature)):
#     feature[i] = torch.tensor(feature[i])
    

cid = CustomDataset(feature, label)


for index in range(600):
    mean_value = np.mean(cid[index][0])
    print("Mean:", mean_value)
    
for index in range(600):
    std_value = np.std(cid[index][0])
    print("Standard Deviation:", std_value)

data_loader = torch.utils.data.DataLoader(
    cid,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=0,
)


class EncoderRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, (hn, cn) = self.rnn(x)
        out = self.fc(hn[-1])
        return out, hn

MAX_LENGTH = 100
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long).fill_(0)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden  = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                decoder_input = target_tensor[:, i].unsqueeze(1) 
            else:
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden, None

    def forward_step(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden


encoder = EncoderRNN(input=12, hidden_size=50)
decoder = DecoderRNN(hidden_size=50, output_size=75)

encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.001)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=0.001)

criterion = nn.NLLLoss()

num_epochs = 100

for epoch in range(1, num_epochs):
    loss_epoch = 0
    for step, (feature, label) in enumerate(data_loader):

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(feature)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, label)

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            label.view(-1)
        )
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        loss_epoch += loss.item()

        if step % 250 == 0:
            print(
                f"Step [{step}/{len(data_loader)}]\t loss: {loss.item()}")
        loss_epoch += loss.item()

    print(f"Epoch [{epoch}/{num_epochs}]\t Loss: {loss_epoch / len(data_loader)}")
