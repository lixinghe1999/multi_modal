import numpy as np
import torch
import torch.nn as nn
import numba
from .torch_box_ops import center_to_corner_box3d, project_to_image, box_lidar_to_camera
from torchvision.ops.boxes import _box_inter_union, nms
from ...model_utils import model_nms_utils

def IoU(boxes, query_boxes, scores_3d, scores_2d, dis_to_lidar_3d):
    """
    Args:
        input_boxes: Tensor of shape (N, 4) or (4,).
        target_boxes: Tensor of shape (N, 4) or (4,).
        eps (float): small number to prevent division by zero
    """
    inter, union = _box_inter_union(boxes, query_boxes)
    iou = inter / union
    iou = iou.reshape(-1)
    iou_mask = iou>0
    index = iou_mask.nonzero()
    iou = iou[iou_mask]
    return iou, index
class LateFusion(nn.Module):
    def __init__(self, nms_config):
        super().__init__()
        self.fuse_2d_3d = nn.Sequential(
            nn.Conv2d(4,18,1),
            nn.ReLU(),
            nn.Conv2d(18,36,1),
            nn.ReLU(),
            nn.Conv2d(36,36,1),
            nn.ReLU(),
            nn.Conv2d(36,1,1),
        )
        self.nms_config = nms_config
    def get_iou(self, box_2d, box_3d, calib, pred_score, image_shape):
        '''
        box_2d: [1, k, 4]
        box_3d: [1, N, 7], note that N is very large
        mostly the first dimension is discarded
        '''
        img_height, img_width = image_shape
        R0, V2C, P2 = torch.from_numpy(calib.R0).to(device=box_3d.device), torch.from_numpy(calib.V2C).to(device=box_3d.device), torch.from_numpy(calib.P2).to(device=box_3d.device)
        final_box_preds_camera = box_lidar_to_camera(box_3d, R0, V2C)
        locs = final_box_preds_camera[:, :3]
        dims = final_box_preds_camera[:, 3:6]
        angles = final_box_preds_camera[:, 6]
        camera_box_origin = [0.5, 1.0, 0.5]
        box_corners = center_to_corner_box3d(
            locs, dims, angles, camera_box_origin, axis=1)
        box_corners_in_image = project_to_image(box_corners, P2)
        minxy = torch.min(box_corners_in_image, dim=1)[0]
        maxxy = torch.max(box_corners_in_image, dim=1)[0]
        minxy[:,0] = torch.clamp(minxy[:,0],min = 0,max = img_width)
        minxy[:,1] = torch.clamp(minxy[:,1],min = 0,max = img_height)
        maxxy[:,0] = torch.clamp(maxxy[:,0],min = 0,max = img_width)
        maxxy[:,1] = torch.clamp(maxxy[:,1],min = 0,max = img_height)
        box_2d_preds = torch.cat([minxy, maxxy], dim=1)
        dis_to_lidar = torch.norm(box_3d[:,:2],p=2,dim=1,keepdim=True)/82.0
        # print(box_2d_preds)
        # print(box_2d)
        non_empty_iou_test_tensor, non_empty_tensor_index_tensor = IoU(box_2d_preds, box_2d,
                                            pred_score, torch.ones((box_2d.shape[0], 1)), dis_to_lidar)
       
        return non_empty_iou_test_tensor, non_empty_tensor_index_tensor
    def forward(self, batch_dict):
        # box = batch_dict['gt_boxes2d']
        # # box_3d = batch_dict['batch_box_preds']
        # box_3d = batch_dict['gt_boxes']

        # for b2, b3, c, p, image_shape in zip(box, box_3d, batch_dict['calib'], batch_dict['batch_cls_preds'], batch_dict['image_shape']):
        #     # cls_preds = torch.sigmoid(p)
        #     # cls_preds, label_preds = torch.max(cls_preds, dim=-1)
        #     # selected, selected_scores = model_nms_utils.class_agnostic_nms(
        #     #     box_scores=cls_preds, box_preds=b3,
        #     #     nms_config=self.nms_config,
        #     #     score_thresh=None
        #     # )
        #     # b3 = b3[selected]
        #     # p = selected_scores
        #     print(b2.shape, b3.shape)
        #     iou, tensor_index = self.get_iou(b2, b3, c, p, image_shape)
        #     print(iou.shape)
        #     iou = iou.reshape(1, 1, 1, -1).repeat(1, 4, 1, 1)
        #     x = self.fuse_2d_3d(iou)

        #     out = torch.zeros(1, b2.shape[0] * b3.shape[0], dtype=x.dtype,device = x.device)
        #     out[:, tensor_index[:, 0]] = x[0, 0, 0,:]
        #     out = out.reshape(1, b2.shape[0], b3.shape[0])
        #     x = torch.max(out, dim=1)[0]
            
        return batch_dict
