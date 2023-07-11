import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

batch_size = 64
torch.manual_seed(42)

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file):
        # self.img_labels = pd.read_csv(annotations_file)
        self.img_labels = annotations_file

        pass

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        vector = self.img_labels[idx][0]
        # vector = self.img_labels.iloc[idx, 0]
        # label = self.img_labels.iloc[idx, 1]
        # label = self.img_labels.iloc[idx, 1]
        label = self.img_labels[idx][1]
        # vector = self.transform(vector)
        # label = self.transform(label)

        # vector = torch.stack([torch.Tensor(eval(v)) for v in vector])
        # label = torch.stack([torch.Tensor(eval(l)) for l in label])

        return vector, label

# import csv
# import numpy as np

# class CustomImageDataset:
#     def __init__(self, csv_file):
#         self.data = []
#         with open(csv_file, 'r') as file:
#             reader = csv.reader(file)
#             next(reader)
#             for row in reader:
#                 feature_vector = 0
#                 # feature_vector = np.fromstring(row[0][1:-1], dtype=np.float32, sep=' ')
#                 # print(row[1][1:-1])
#                 # print(row[0][1:-1].replace("[","").replace("]",""))

#                 label = np.fromstring(row[0][1:-1].replace("[","").replace("]",""), dtype=np.float32, sep=' ')
#                 self.data.append((feature_vector, label))
    
#     def __getitem__(self, index):
#         feature_vector, label = self.data[index]
#         return feature_vector, torch.from_numpy(label)
    
#     def __len__(self):
#         return len(self.data)


# csvfile = "dataset.csv"
csvfile = [[0, 1, 2], [0, 2, 4]]
cid = CustomImageDataset(csvfile)

print(cid[0][2])
print(cid[1])

# print(cid[0][0].shape)
# print(cid[0][1].shape)


data_loader = torch.utils.data.DataLoader(
    cid,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=0,
)

