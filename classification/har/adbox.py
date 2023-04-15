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
        depth = np.load(self.data_root + '/depth_feature/' + file + '.npy').astype(np.float32)
        radar = np.load(self.data_root + '/radar/' + file + '.npy').astype(np.float32)
        radar = radar.transpose((0, 1, 4, 2, 3)).reshape((30, -1, 16, 32))[::3, ...]
        imu = pd.read_csv(self.data_root + '/imu/' + file + '.csv').to_numpy()[:, :-1].astype(np.float32)
        label = self.label[index]
        return depth, radar, imu, label

    def __len__(self) -> int:
        return len(self.files)

if __name__ == "__main__":
    dataset = ADBox('../dataset/adbox')
    dataset.__getitem__(0)
