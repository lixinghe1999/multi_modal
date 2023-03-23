from model.dyn_slim.dyn_slim_ops import DSpwConv2d, DSdwConv2d, DSBatchNorm2d, \
    DSAvgPool2d, DSAdaptiveAvgPool2d, DSConv2d, DSLinear
import torch.nn as nn
import random
import torch
from typing import Callable, Optional, List
def make_divisible(v, divisor=8, min_value=None):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v
def drop_path(inputs, training=False, drop_path_rate=0.):
    """Apply drop connect."""
    if not training:
        return inputs

    keep_prob = 1 - drop_path_rate
    random_tensor = keep_prob + torch.rand(
        (inputs.size()[0], 1, 1, 1), dtype=inputs.dtype, device=inputs.device)
    random_tensor.floor_()  # binarize
    output = inputs.div(keep_prob) * random_tensor
    return output

def gumbel_softmax(logits, tau=1, hard=False, dim=1, training=True):
    """ See `torch.nn.functional.gumbel_softmax()` """
    # if training:
    # gumbels = -torch.empty_like(logits,
    #                             memory_format=torch.legacy_contiguous_format).exponential_().log()  # ~Gumbel(0,1)
    # gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    # # else:
    # #     gumbels = logits
    # y_soft = gumbels.softmax(dim)

    gumbels = -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)
    with torch.no_grad():
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        #  **test**
        # index = 0
        # y_hard = torch.Tensor([1, 0, 0, 0]).repeat(logits.shape[0], 1).cuda()
    ret = y_hard - y_soft.detach() + y_soft
    return y_soft, ret, index

def set_exist_attr(m, attr, value):
    if hasattr(m, attr):
        setattr(m, attr, value)

class SlimBlock(nn.Module):
    expansion: int = 4
    def __init__(
        self,
        inplanes,
        planes,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        width = [int(p * (base_width / 64.0)) * groups for p in planes]
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = DSConv2d(inplanes, width, 1)
        self.bn1 = DSBatchNorm2d(width)
        self.conv2 = DSConv2d(width, width, 3, stride, dilation, groups)
        self.bn2 = DSBatchNorm2d(width)
        self.conv3 = DSConv2d(width, [p * self.expansion for p in planes], 1)
        self.bn3 = DSBatchNorm2d([p * self.expansion for p in planes])
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out
class SlimResNet(nn.Module):
    def __init__(
        self,
        block,
        layers: List[int],
        num_classes: int = 1000,
        dims=[[32, 64], [64, 128], [128, 256], [256, 512]],
        groups: int = 1,
        width_per_group: int = 64,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = DSBatchNorm2d
        self.inplanes = dims[0]
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = DSConv2d(3, dims[0], 7, stride=2)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, dims[0], layers[0])
        self.layer2 = self._make_layer(block, dims[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, dims[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, dims[3], layers[3], stride=2)
        self.blocks = [self.layer1, self.layer2, self.layer3, self.layer4]
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = DSLinear([p * block.expansion for p in dims[3]], num_classes)

    def _make_layer(
        self,
        block,
        planes,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        norm_layer = DSBatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != [p * block.expansion for p in planes]:
            downsample = nn.Sequential(
                DSConv2d(self.inplanes, [p * block.expansion for p in planes], 1, stride),
                norm_layer([p * block.expansion for p in planes]),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample)
        )
        self.inplanes = [p * block.expansion for p in planes]
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes)
            )
        return nn.Sequential(*layers)

    def set_gate(self, channel_choice):
        for n, m in self.named_modules():
            set_exist_attr(m, 'channel_choice', channel_choice)
            set_exist_attr(m, 'prev_channel_choice', channel_choice)
    def preprocess(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x
    def forward(self, x):
        x = self.preprocess(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


