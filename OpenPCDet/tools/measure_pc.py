import spconv.pytorch as spconv
from pcdet.models.backbones_3d.spconv_backbone import VoxelBackBone8x
from pcdet.datasets.processor.data_processor import DataProcessor
from torch import nn
import torch 
import numpy as np
import time

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import calibration_kitti


def get_fov_flag(pts_rect, img_shape, calib):
    pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
    val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
    val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
    val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
    pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)
    return pts_valid_flag

num_iterations = 50
start_count = 10
spatial_shape=[1408, 1600, 40]
model = VoxelBackBone8x(model_cfg={'NAME': 'VoxelBackBone8x'}, input_channels=4, grid_size=np.array(spatial_shape),
            voxel_size=[0.05, 0.05, 0.1], point_cloud_range=[0, -40, -3, 70.4, 40, 1]).to('cuda')
model.eval()
dataset_cfg = cfg_from_yaml_file('../tools/cfgs/dataset_configs/kitti_dataset.yaml', cfg)
data_dict = dict()
sample_idx = 1203
sample_idx = str(sample_idx).zfill(6)

for down_sample in [4]:
    point_cloud = np.fromfile('../data/kitti/training/velodyne/' + sample_idx + '.bin', dtype=np.float32).reshape(-1, 4)
    calib_file = '../data/kitti/training/calib/' + sample_idx + '.txt'
    calib =  calibration_kitti.Calibration(calib_file)
    pts_rect = calib.lidar_to_rect(point_cloud[:, 0:3])
    fov_flag = get_fov_flag(pts_rect, np.array([ 375, 1242]), calib)
    point_cloud = point_cloud[fov_flag]

    point_cloud = point_cloud[:len(point_cloud)//down_sample]
    data_dict['points'] = point_cloud
    data_dict['use_lead_xyz'] = True
    data_dict['batch_size'] = 1
    generator = DataProcessor(dataset_cfg.DATA_PROCESSOR, point_cloud_range=np.array([0, -40, -3, 70.4, 40, 1]),
                training=True, num_point_features=4)
    data_dict = generator.mask_points_and_boxes_outside_range(data_dict, config=dataset_cfg.DATA_PROCESSOR[0])  
    data_dict = generator.shuffle_points(data_dict, config=dataset_cfg.DATA_PROCESSOR[1])
    data_dict = generator.transform_points_to_voxels(data_dict, config=dataset_cfg.DATA_PROCESSOR[2])
    data_dict['voxel_features'] = np.mean(data_dict['voxels'], axis=1)
    data_dict['voxel_coords'] = np.concatenate([np.zeros((data_dict['voxel_coords'].shape[0], 1)), data_dict['voxel_coords']], axis=1)

    data_dict['voxel_features'] = torch.from_numpy(data_dict['voxel_features']).to('cuda')
    data_dict['voxel_coords'] = torch.from_numpy(data_dict['voxel_coords']).to('cuda')
    # data_dict['voxel_features'] = data_dict['voxel_features'][::down_sample]
    # data_dict['voxel_coords'] = data_dict['voxel_coords'][::down_sample]

    for i in range(num_iterations):
        if i == start_count:
            t_start = time.time()
        model(data_dict)
    print(data_dict['voxel_features'].shape[0], (time.time() - t_start)/(num_iterations-start_count) * 1000)