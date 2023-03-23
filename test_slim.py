from model.slim_model import SlimBlock, make_layer
from model.resnet_model import Bottleneck, resnet50
import torch

model = resnet50()
print(resnet50().layer1)
# data = torch.rand(1, 16, 56, 56)
# print(model(data))

model = make_layer(SlimBlock, [8, 16], [32, 64], 3)
# print(model)
# data = torch.rand(1, 16, 56, 56)
# print(model(data).shape)