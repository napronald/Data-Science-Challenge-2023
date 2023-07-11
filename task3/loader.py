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

case = 16116 #16117

num_timesteps = 500
titles = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
reorder = {1:1,2:5,3:9,4:2,5:6,6:10,7:3,8:7,9:11,10:4,11:8,12:12} 

print('Case {} : {}'.format(case, file_pairs[case][0]))
pECGData = np.load(file_pairs[case][0])
pECGData = get_standard_leads(pECGData) # X-> [500,12]

print(pECGData)
print(pECGData.shape)

# # create a figure with 12 subplots
# for i in range(pECGData.shape[1]):
#     plt.plot(pECGData[0:num_timesteps,i],'r')
#     plt.title(titles[i])
#     plt.grid(visible=True, which='major', color='#666666', linestyle='-')
#     plt.minorticks_on()
#     plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
#     plt.xlabel('msec')
#     plt.ylabel('mV')
# plt.tight_layout()
# plt.show()
# # close
# plt.close()


# make a plot with the "pECGData" -> "ActTime"
print('Case {} : {}'.format(case, file_pairs[case][0]))
VmData = np.load(file_pairs[case][1])
ActTime = get_activation_time(VmData)

print(ActTime)
print(ActTime.shape)


# # plot in row the tensors pECGData and ActTime with an arrow pointing to the activation time
# row = 1
# column = 3
# plt.figure(figsize=(20, 5))
# plt.subplot(row, column, 1)
# # plot pECGData transposed
# plt.imshow(pECGData.T, cmap='jet', interpolation='nearest', aspect='auto')
# plt.title('pECGData')
# plt.subplot(row, column, 2)
# # print an arrow
# plt.text(0.5, 0.5, '-------->', horizontalalignment='center', verticalalignment='center', fontsize=20)
# plt.axis('off')
# plt.subplot(row, column, 3)
# # plot ActTime
# plt.imshow(ActTime, cmap='jet', interpolation='nearest', aspect='auto')
# # not xticks
# plt.xticks([])
# plt.title('ActTime')
# plt.show()
# plt.close()