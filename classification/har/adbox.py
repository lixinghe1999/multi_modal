import os
import numpy as np
import pandas as pd
import torch.utils.data as td
import scipy
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
        radar = radar[::3, ...].transpose((0, 1, 4, 2, 3)).reshape((-1, 16, 32))
        imu = pd.read_csv(self.data_root + '/imu/' + file + '.csv').to_numpy()[:, :-1].astype(np.float32)
        imu = np.abs(scipy.signal.stft(imu.transpose(1, 0), fs=100, nperseg=32, noverlap=28)[-1])
        label = self.label[index]
        return depth, radar, imu, label

    def __len__(self) -> int:
        return len(self.files)

if __name__ == "__main__":
    dataset = ADBox('../dataset/adbox')
    dataset.__getitem__(0)
