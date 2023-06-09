from .video_records import EpicKitchens55_VideoRecord, EpicKitchens100_VideoRecord
import torch.utils.data as data

import librosa
from PIL import Image
import os
import os.path
from pathlib import Path
import pandas as pd
import numpy as np
from numpy.random import randint
import torch
import torchvision
from .transform import GroupScale, GroupCenterCrop, GroupOverSample, GroupNormalize, Stack, ToTorchFormatTensor, GroupRandomHorizontalFlip, GroupMultiScaleCrop
def get_augmentation(modality, input_size):
        augmentation = {}
        if 'RGB' in modality:
            augmentation['RGB'] = torchvision.transforms.Compose([GroupMultiScaleCrop(input_size['RGB'], [1, .875, .75, .66]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])
        if 'Flow' in modality:
            augmentation['Flow'] = torchvision.transforms.Compose([GroupMultiScaleCrop(input_size['Flow'], [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=True)])
        if 'RGBDiff' in modality:
            augmentation['RGBDiff'] = torchvision.transforms.Compose([GroupMultiScaleCrop(input_size['RGBDiff'], [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])

        return augmentation
def get_normalization(modality, input_mean, input_std):
    normalize = {}
    for m in modality:
        normalize[m] = GroupNormalize(input_mean[m], input_std[m])
    return normalize
# mean: spec, rgb, flow
# std: spec, rgb, flow
# tensor([-0.0322]) tensor([0.4290, 0.3740, 0.3360]) tensor([0.5006, 0.5108])
# tensor([0.0137]) tensor([0.2387, 0.2269, 0.2275]) tensor([0.1581, 0.1154])
def get_train_transform(modality=['RGB', 'Spec', 'Flow'], test_crops=10, scale_size={'RGB': 256, 'Spec': 256, 'Flow': 256}, input_size={'RGB': 224, 'Spec': 256, 'Flow': 224}, input_mean={'RGB': [0.4290, 0.3740, 0.3360], 'Flow': [0.5006, 0.5108], 'Spec':[0]}, input_std={'RGB': [0.2387, 0.2269, 0.2275], 'Flow': [0.1581, 0.1154], 'Spec': [1]}):
    train_transform = {}
    val_transform = {}
    train_augmentation = get_augmentation(modality, input_size)
    normalize = get_normalization(modality, input_mean, input_std)
    for m in modality:
        if (m != 'Spec'):
            train_transform[m] = torchvision.transforms.Compose([
                train_augmentation[m],
                Stack(),
                ToTorchFormatTensor(),
                normalize[m],
            ])

            val_transform[m] = torchvision.transforms.Compose([
                GroupScale(int(scale_size[m])),
                GroupCenterCrop(input_size[m]),
                Stack(),
                ToTorchFormatTensor(),
                normalize[m],
            ])
        else:
            # Prepare train/val dictionaries containing the transformations
            # (augmentation+normalization)
            # for each modality
            train_transform[m] = torchvision.transforms.Compose([
                Stack(),
                ToTorchFormatTensor(),
                normalize[m],
            ])

            val_transform[m] = torchvision.transforms.Compose([
                Stack(),
                ToTorchFormatTensor(),
                normalize[m],
            ])
    return train_transform, val_transform
            
class EPICKitchen(data.Dataset):
    def __init__(self, dataset='epic-kitchens-100', list_file=pd.read_pickle('EPIC_val.pkl'),
                 new_length={'RGB': 1, 'Flow': 1, 'Spec': 1, 'IMU': 1}, modality= ['RGB', 'Spec', 'Flow', 'IMU'], image_tmpl={'RGB': 'frame_{:010d}.jpg', 'Flow': 'frame_{:010d}.jpg'},  visual_path='../EPIC-KITCHENS', audio_path='./audio_dict.pkl', imu_path='../EPIC-KITCHENS', resampling_rate=24000, num_segments=1, transform=None,
                 mode='test', use_audio_dict=True, narration_embedding='train_vector.npy'):
        self.dataset = dataset
        if audio_path is not None:
            if not use_audio_dict:
                self.audio_path = Path(audio_path)
            else:
                self.audio_path = audio_path
        self.visual_path = visual_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.mode = mode
        self.resampling_rate = resampling_rate
        self.use_audio_dict = use_audio_dict
        # self.narration_embedding = np.load(narration_embedding)
        self.imu_path = imu_path

        self.acc_dict = {}
        self.gyro_dict = {}
        for p in os.listdir(self.imu_path):
            path = os.path.join(self.imu_path, p, 'meta_data')
            for f in os.listdir(path):
                data = np.loadtxt(os.path.join(path, f))[:, 1:]
                video_id = f.split('-')[0]
                if f.split('-')[1] == 'accl.csv':
                    self.acc_dict[video_id] = data
                else:# gyro
                    self.gyro_dict[video_id] = data
        if 'RGBDiff' in self.modality:
            self.new_length['RGBDiff'] += 1  # Diff needs one more image to calculate diff

        self._parse_list()

    def _log_specgram(self, audio, window_size=10,
                     step_size=5, eps=1e-6):
        nperseg = int(round(window_size * self.resampling_rate / 1e3))
        noverlap = int(round(step_size * self.resampling_rate / 1e3))

        spec = librosa.stft(audio, n_fft=511,
                            window='hann',
                            hop_length=noverlap,
                            win_length=nperseg,
                            pad_mode='constant')

        spec = np.log(np.real(spec * np.conj(spec)) + eps)
        return spec

    def _extract_sound_feature(self, record, idx):

        centre_sec = (record.start_frame + idx) / record.fps['Spec']
        left_sec = centre_sec - 0.639
        right_sec = centre_sec + 0.639
        if not self.use_audio_dict:
            audio_fname = record.untrimmed_video_name + '.wav'
            samples, sr = librosa.core.load(self.audio_path / audio_fname,
                                            sr=None, mono=True)
        else:
            audio_fname = record.untrimmed_video_name
            samples = self.audio_path[audio_fname]

        duration = samples.shape[0] / float(self.resampling_rate)

        left_sample = int(round(left_sec * self.resampling_rate))
        right_sample = int(round(right_sec * self.resampling_rate))

        if left_sec < 0:
            samples = samples[:int(round(self.resampling_rate * 1.279))]

        elif right_sec > duration:
            samples = samples[-int(round(self.resampling_rate * 1.279)):]
        else:
            samples = samples[left_sample:right_sample]

        return self._log_specgram(samples)

    def _get_imu_data(self, record, idx):
        center_frame = (record.start_frame + idx) / record.fps['RGB'] * record.fps['IMU']
        left_frame = center_frame - 100
        right_frame = center_frame + 100
        acc_data = record.acc_dict[record.untrimmed_video_name][left_frame: right_frame]
        gyro_data = record.gyro_dict[record.untrimmed_video_name][left_frame: right_frame]
        imu_data = np.concatenate((acc_data, gyro_data), axis=1)
        return imu_data
    def _load_data(self, modality, record, idx):
        if modality == 'RGB' or modality == 'RGBDiff':
            idx_untrimmed = record.start_frame + idx
            return [Image.open(os.path.join(self.visual_path, record.participant_id, 'rgb_frames', record.untrimmed_video_name, self.image_tmpl[modality].format(idx_untrimmed))).convert('RGB')]
        elif modality == 'Flow':
            rgb2flow_fps_ratio = record.fps['Flow'] / float(record.fps['RGB'])
            idx_untrimmed = int(np.floor((record.start_frame * rgb2flow_fps_ratio))) + idx
            x_img = Image.open(os.path.join(self.visual_path, record.participant_id, 'flow_frames', record.untrimmed_video_name, 'u', self.image_tmpl[modality].format(idx_untrimmed))).convert('L')
            y_img = Image.open(os.path.join(self.visual_path, record.participant_id, 'flow_frames', record.untrimmed_video_name, 'v', self.image_tmpl[modality].format(idx_untrimmed))).convert('L')
            return [x_img, y_img]
        elif modality == 'Spec':
            spec = self._extract_sound_feature(record, idx)
            return [Image.fromarray(spec)]
        else:
            imu = self._get_imu_data(record, idx)
            return [Image.fromarray(imu)]

    def _parse_list(self):
        if self.dataset == 'epic-kitchens-55':
            self.video_list = [EpicKitchens55_VideoRecord(tup) for tup in self.list_file.iterrows()]
        elif self.dataset == 'epic-kitchens-100':
            self.video_list = [EpicKitchens100_VideoRecord(tup) for tup in self.list_file.iterrows()]

    def _sample_indices(self, record, modality):
        """

        :param record: VideoRecord
        :return: list
        """
        average_duration = (record.num_frames[modality] - self.new_length[modality] + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(1, average_duration+1, size=self.num_segments)
        # elif record.num_frames[modality] > self.num_segments:
        #     offsets = np.sort(randint(record.num_frames[modality] - self.new_length[modality] + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets

    def _get_val_indices(self, record, modality):
        if record.num_frames[modality] > self.num_segments + self.new_length[modality] - 1:
            tick = (record.num_frames[modality] - self.new_length[modality] + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets

    def _get_test_indices(self, record, modality):

        tick = (record.num_frames[modality] - self.new_length[modality] + 1) / float(self.num_segments)

        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])

        return offsets

    def __getitem__(self, index):

        input = {}
        record = self.video_list[index]
        # narration = self.narration_embedding[index]

        for m in self.modality:
            if self.mode == 'train':
                segment_indices = self._sample_indices(record, m)
            elif self.mode == 'val':
                segment_indices = self._get_val_indices(record, m)
            elif self.mode == 'test':
                segment_indices = self._get_test_indices(record, m)

            # We implement a Temporal Binding Window (TBW) with size same as the action's length by:
            #   1. Selecting different random indices (timestamps) for each modality within segments
            #      (this is similar to using a TBW with size same as the segment's size)
            #   2. Shuffling randomly the segments of Flow, Audio (RGB is the anchor hence not shuffled)
            #      which binds data across segments, hence making the TBW same in size as the action.
            #   Example of an action with 90 frames across all modalities:
            #    1. Synchronous selection of indices per segment:
            #       RGB: [12, 41, 80], Flow: [12, 41, 80], Audio: [12, 41, 80]
            #    2. Asynchronous selection of indices per segment:
            #       RGB: [12, 41, 80], Flow: [9, 55, 88], Audio: [20, 33, 67]
            #    3. Asynchronous selection of indices per action:
            #       RGB: [12, 41, 80], Flow: [88, 55, 9], Audio: [67, 20, 33]

            if m != 'RGB' and self.mode == 'train':
                np.random.shuffle(segment_indices)
            img, label, metadata = self.get(m, record, segment_indices)
            input[m] = img
        # label['feature'] = narration
        return input['Spec'], input['RGB'], input['Flow'], label
    def get(self, modality, record, indices):

        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length[modality]):
                seg_imgs = self._load_data(modality, record, p)
                images.extend(seg_imgs)
                if p < record.num_frames[modality]:
                    p += 1
        process_data = self.transform[modality](images)
        # classification
        return process_data, record.label, record.metadata
    def __len__(self):
        return len(self.video_list)

if __name__ == "__main__":
    import h5py
    print('pre-load audio dict.....')
    audio_path = h5py.File('../split_EPIC_audio.hdf5', 'r')
    print('finish loading....')
    train_transform, val_transform = get_train_transform()
    dataset = EPICKitchen(list_file=pd.read_pickle('EPIC_train.pkl'),  transform=train_transform, mode='train', audio_path=audio_path)
    train_loader = torch.utils.data.DataLoader(dataset=dataset, num_workers=4, batch_size=16, drop_last=True)
    spec_mean = 0; rgb_mean = 0; flow_mean = 0
    spec_std = 0; rgb_std = 0; flow_std = 0
    for batch in train_loader:
        spec, rgb, flow, label = batch
        spec_mean += spec.reshape(16 * 3, 1, 256, 256).mean(dim=(0, 2, 3))
        rgb_mean += rgb.reshape(16 * 3, 3, 224, 224).mean(dim=(0, 2, 3))
        flow_mean += flow.reshape(16 * 3, 2, 224, 224).mean(dim=(0, 2, 3))
        spec_std += spec.reshape(16 * 3, 1, 256, 256).std(dim=(0, 2, 3)) ** 2 
        rgb_std += rgb.reshape(16 * 3, 3, 224, 224).std(dim=(0, 2, 3)) ** 2
        flow_std += flow.reshape(16 * 3, 2, 224, 224).std(dim=(0, 2, 3)) ** 2
    spec_mean /= len(train_loader)
    rgb_mean /= len(train_loader)
    flow_mean /= len(train_loader)
    spec_std = (spec_std / len(train_loader)) ** 0.5
    rgb_std = (rgb_std / len(train_loader)) ** 0.5
    flow_std = (flow_std / len(train_loader)) ** 0.5
    print(spec_mean, rgb_mean, flow_mean) 
    print(spec_std, rgb_std, flow_std)