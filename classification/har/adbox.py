import os
import numpy as np
import pandas as pd
import torch.utils.data as td

class ADBox(td.Dataset):
    def __init__(self, root, split='train1'):
        super(ADBox, self).__init__()
        self.data_root = root + '/' + split
        self.files = os.listdir(self.data_root + '/depth_feature')
        self.files = [f.split('.')[0] for f in self.files]
        self.label = [int(f[4:6]) - 1 for f in self.files]

    def __getitem__(self, index: int):
        file = self.files[index]
        depth = np.load(self.data_root + '/depth_feature/' + file + '.npy')
        radar = np.load(self.data_root + '/radar/' + file + '.npy')
        imu = pd.read_csv(self.data_root + '/imu/' + file + '.csv').to_numpy()[:, :-1]
        label = self.label[index]
        print(depth.shape, radar.shape, imu.shape)
        return depth, radar, imu, label

    def __len__(self) -> int:
        return len(self.files)

if __name__ == "__main__":
    dataset = ADBox('../dataset/adbox')
    dataset.__getitem__(0)
