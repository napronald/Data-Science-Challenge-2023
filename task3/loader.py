import glob, re, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
import torch

from torch.utils.data import Dataset
from cardiac_ml_tools import read_data_dirs, get_standard_leads, get_activation_time

data_dirs = []
regex = r'data_hearts_dd_0p2*'
DIR='/home/rnap/scratch/dsc/task3/intracardiac_dataset/' # This should be the path to the intracardiac_dataset, it can be downloaded using data_science_challenge_2023/download_intracardiac_dataset.sh
for x in os.listdir(DIR):
    if re.match(regex, x):
        data_dirs.append(DIR + x)
file_pairs = read_data_dirs(data_dirs)

num_timesteps = 500
titles = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
reorder = {1:1,2:5,3:9,4:2,5:6,6:10,7:3,8:7,9:11,10:4,11:8,12:12} 


# for case in range(16116):
#     # print('Case {} : {}'.format(case, file_pairs[case][0]))
#     pECGData = np.load(file_pairs[case][0])
#     pECGData = get_standard_leads(pECGData) # X-> [500,12]

#     # print(pECGData)
#     print(pECGData.shape)


#     print('Case {} : {}'.format(case, file_pairs[case][0]))
#     VmData = np.load(file_pairs[case][1])
#     ActTime = get_activation_time(VmData) # A->[75,1]

#     # print(ActTime)
#     print(ActTime.shape)


feature = []
label = []

for case in range(16116):
    pECGData = np.load(file_pairs[case][0])
    pECGData = get_standard_leads(pECGData)

    VmData = np.load(file_pairs[case][1])
    ActTime = get_activation_time(VmData)

    feature.append(pECGData)
    label.append(ActTime)

feature_vectors = np.array(feature)
label_vectors = np.array(label)

df = pd.DataFrame({
    'Feature': [vec1 for vec1 in feature_vectors],
    'Label': [vec2 for vec2 in label_vectors],
})

print(df)

df.to_csv(os.path.join(os.getcwd(), "dataset.csv"), index=False)


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file):
        self.img_labels = annotations_file

        pass

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        vector = self.img_labels[idx][0]
        label = self.img_labels[idx][1]

        return vector, label


# csvfile = [[0, 1, 2], [0, 2, 4]]
csvfile = df
cid = CustomImageDataset(csvfile)

print(cid[0])



data_loader = torch.utils.data.DataLoader(
    cid,
    batch_size=64,
    shuffle=True,
    drop_last=True,
    num_workers=0,
)
