
import torch.nn as nn
import torch
from torch.cuda.amp import autocast
from .convnext_model import AdaConvNeXt
class MMTM(nn.Module):
      def __init__(self, dim_visual, dim_skeleton, ratio):
        super(MMTM, self).__init__()
        dim = dim_visual + dim_skeleton
        dim_out = int(2*dim/ratio)
        self.fc_squeeze = nn.Linear(dim, dim_out)

        self.fc_visual = nn.Linear(dim_out, dim_visual)
        self.fc_skeleton = nn.Linear(dim_out, dim_skeleton)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

      def forward(self, visual, skeleton):
        squeeze_array = []
        for tensor in [visual, skeleton]:
          tview = tensor.view(tensor.shape[:2] + (-1,))
          squeeze_array.append(torch.mean(tview, dim=-1))
        squeeze = torch.cat(squeeze_array, 1)

        excitation = self.fc_squeeze(squeeze)
        excitation = self.relu(excitation)

        vis_out = self.fc_visual(excitation)
        sk_out = self.fc_skeleton(excitation)

        vis_out = self.sigmoid(vis_out)
        sk_out = self.sigmoid(sk_out)

        dim_diff = len(visual.shape) - len(vis_out.shape)
        vis_out = vis_out.view(vis_out.shape + (1,) * dim_diff)

        dim_diff = len(skeleton.shape) - len(sk_out.shape)
        sk_out = sk_out.view(sk_out.shape + (1,) * dim_diff)

        return visual * vis_out, skeleton * sk_out

class HARnet(nn.Module):
    def __init__(self, pretrained=False):
        super(HARnet, self).__init__()
        # image, imu, radar
        self.branch = nn.ModuleList()
        self.branch.append(AdaConvNeXt(in_chans=16, sparse_ratio=[0.8, 0.6, 0.4], pruning_loc=[3,6,9], num_classes=10, depths=[3, 3, 27, 3]))
        self.branch.append(AdaConvNeXt(in_chans=320, sparse_ratio=[0.8, 0.6, 0.4], pruning_loc=[3,6,9],
                                       num_classes=10, depths=[3, 3, 27, 3], down_sample=False))
        self.branch.append(AdaConvNeXt(in_chans=15, sparse_ratio=[0.8, 0.6, 0.4], pruning_loc=[3,6,9],
                                       num_classes=10, depths=[3, 3, 27, 3], down_sample=False))
        if pretrained:
            pass
        self.head = nn.Sequential(nn.Linear(768 * 3, 14))
    def fusion_parameter(self):
        parameter = [{'params': self.head.parameters()},]
        return parameter

    @autocast()
    def forward(self, depth, radar, imu):
        feat = []
        for x, branch in zip((depth, radar, imu), self.branch):
            print(x.shape)
            x = branch.preproces(x)
            x, mask, decisions = branch.main_stage(x)
            x, featmap = branch.final(x, mask)
            print(x.shape)
            feat.append(x)
        output = self.head(torch.cat(feat, dim=-1))
        return output
