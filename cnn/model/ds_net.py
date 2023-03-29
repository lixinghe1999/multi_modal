from model.dyn_slim.dyn_slim_ops import DSpwConv2d, DSBatchNorm2d, \
    DSAdaptiveAvgPool2d, DSConv2d, DSLinear
import torch.nn as nn
import random
import torch
from typing import Optional


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
class MultiHeadGate(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None, act_layer=nn.ReLU,
                 attn_act_fn=nn.Sigmoid(), divisor=1, channel_gate_num=None, gate_num_features=1024):
        super(MultiHeadGate, self).__init__()
        self.attn_act_fn = attn_act_fn
        self.channel_gate_num = channel_gate_num
        reduced_chs = make_divisible((reduced_base_chs or in_chs[-1]) * se_ratio, divisor)
        self.avg_pool = DSAdaptiveAvgPool2d(1, channel_list=in_chs)
        self.conv_reduce = DSpwConv2d(in_chs, [reduced_chs], bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = DSpwConv2d([reduced_chs], in_chs, bias=True)

        self.has_gate = False
        if channel_gate_num > 1:
            self.has_gate = True
            self.gate = nn.Sequential(DSpwConv2d([reduced_chs], [channel_gate_num], bias=False))

        self.mode = 'largest'
        self.keep_gate, self.print_gate, self.print_idx = None, None, None
        self.channel_choice = None
        self.initialized = False

    def forward(self, x):
        x_pool = self.avg_pool(x)
        x_reduced = self.conv_reduce(x_pool)
        x_reduced = self.act1(x_reduced)
        attn = self.conv_expand(x_reduced)
        if self.attn_act_fn == 'tanh':
            attn = (1 + attn.tanh())
        else:
            attn = self.attn_act_fn(attn)
        x = x * attn

        if self.mode == 'dynamic' and self.has_gate:
            channel_choice = self.gate(x_reduced).squeeze(-1).squeeze(-1)
            self.keep_gate, self.print_gate, self.print_idx = gumbel_softmax(channel_choice, dim=1, training=self.training)
            self.channel_choice = self.print_gate, self.print_idx
        else:
            self.channel_choice = None

        return x

    def get_gate(self):
        return self.channel_choice
class Bottleneck(nn.Module):
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
        self.conv1 = DSConv2d(inplanes, width, 1, bias=False)
        self.bn1 = DSBatchNorm2d(width)
        self.conv2 = DSConv2d(width, width, 3, stride, dilation, groups, bias=False)
        self.bn2 = DSBatchNorm2d(width)
        self.conv3 = DSConv2d(width, [p * self.expansion for p in planes], 1, bias=False)
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
        pretrained=False,
        block = Bottleneck,
        layers = [3, 4, 6, 3],
        dims=[[32, 64], [64, 128], [128, 256], [256, 512]],
        num_classes: int = 1000,
        groups: int = 1,
        width_per_group: int = 64,
    ) -> None:
        super().__init__()
        self.expansion = block.expansion
        self.inplanes = dims[0]
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self.mode = 'largest'
        self.channel_choice = -1

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, dims[0], layers[0])
        self.layer2 = self._make_layer(block, dims[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, dims[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, dims[3], layers[3], stride=2)
        self.blocks = [self.layer1, self.layer2, self.layer3, self.layer4]
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # has_gate = False
        # self.score_predictor = nn.ModuleList([MultiHeadGate([p * self.expansion for p in dim],
        #                                                     channel_gate_num=4 if has_gate else 0) for dim in dims])
        if pretrained:
            self.load_state_dict(torch.load('assets/resnet50.pth'), strict=False)
        self.fc = DSLinear([p * self.expansion for p in dims[3]], num_classes)
    def _make_layer(
        self,
        block,
        planes,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != [p * self.expansion for p in planes]:
            downsample = nn.Sequential(
                DSConv2d(self.inplanes, [p * self.expansion for p in planes], 1, stride, bias=False),
                DSBatchNorm2d([p * self.expansion for p in planes]),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample)
        )
        self.inplanes = [p * self.expansion for p in planes]
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes)
            )
        return nn.Sequential(*layers)

    def set_layer_choice(self, l, choice):
        for m in l.modules():
            set_exist_attr(m, 'channel_choice', choice)
    def set_layer_mode(self, l):
        for m in l.modules():
            set_exist_attr(m, 'mode', self.mode)
    def set_mode(self, mode):
        self.mode = mode
        if mode == 'largest' or mode == 'dynamic':
            self.channel_choice = -1
        elif mode == 'smallest':
            self.channel_choice = 0
        elif mode == 'random':
            self.channel_choice = random.randint(0, len(self.inplanes)-1)
        else:
            self.channel_choice = mode
    def preprocess(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x
    def forward(self, x):
        x = self.preprocess(x)
        for i, block in enumerate(self.blocks):
            self.set_layer_choice(block)
            self.set_layer_mode(block)
            x = block(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


