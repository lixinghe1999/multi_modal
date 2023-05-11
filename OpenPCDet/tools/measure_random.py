import spconv.pytorch as spconv
from pcdet.models.backbones_3d.spconv_backbone import VoxelBackBone8x
import torch 
import random
import numpy as np
def box_indices(block_size, zyx):
    indices = torch.zeros(block_size**3, 4)
    indices[:, 1] = torch.arange(zyx[0], zyx[0]+block_size).repeat_interleave(block_size * block_size)
    indices[:, 2] = torch.arange(zyx[1], zyx[1]+block_size).repeat_interleave(block_size).repeat(block_size)
    indices[:, 3] = torch.arange(zyx[2], zyx[2]+block_size).repeat(block_size * block_size)
    return indices
def get_data(spatial_shape=[1408, 1600, 40], N=15000, distribution='normal'):
    num_channels=4; # batch size of your sparse tensor
    features = 100 * (torch.rand(N, num_channels) - 0.5)# your features with shape [N, num_channels]
    if distribution == 'block':
        block_size = int(N ** (1/3))
        indices = box_indices(block_size, [10, 100, 100])
        features = features[:block_size**3]
    else:
        if distribution == 'normal': 
            indices = torch.tensor(random.sample(range(spatial_shape[0] * spatial_shape[1] * spatial_shape[2]), N))
        elif distribution == 'clip':
            indices = torch.arange(1*N, 2*N)
        indices_z = torch.div(indices, (spatial_shape[0] * spatial_shape[1]), rounding_mode='trunc')
        indices_y = torch.div((indices - indices_z * (spatial_shape[0] * spatial_shape[1])), spatial_shape[0], rounding_mode='trunc')
        indices_x = indices - indices_z * (spatial_shape[0] * spatial_shape[1]) - indices_y * spatial_shape[0]

        indices = torch.zeros(N, 4)
        indices[:, 1] = indices_z
        indices[:, 2] = indices_y
        indices[:, 3] = indices_x

    # your indices/coordinates with shape [N, ndim + 1], batch index must be put in indices[:, 0]
    features = features.to('cuda')
    indices = indices.to('cuda')
    batch_dict = dict()
    batch_dict['voxel_features'] = features
    batch_dict['voxel_coords'] = indices
    batch_dict['batch_size'] = 1
    return batch_dict

spatial_shape=[1408, 1600, 40]
model = VoxelBackBone8x(model_cfg={'NAME': 'VoxelBackBone8x'}, input_channels=4, grid_size=np.array(spatial_shape),
            voxel_size=[0.05, 0.05, 0.1], point_cloud_range=[0, -40, -3, 70.4, 40, 1]).to('cuda')
model.eval()
num_iterations = 100
for num_points in [300000]:
    data = [get_data(spatial_shape, num_points + random.randint(-200, 200), distribution='block') for _ in range(num_iterations)]
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings=np.zeros((num_iterations, 1))
    for _ in range(10):
        model(data[0])
    with torch.no_grad():
        for i in range(num_iterations):
            starter.record()
            model(data[i])
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[i] = curr_time
    mean_syn = np.sum(timings) / num_iterations
    std_syn = np.std(timings)
    print(num_points, mean_syn)

