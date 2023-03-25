import matplotlib.pyplot as plt
from model.ds_net import Bottleneck, SlimResNet
from model.resnet_model import resnet50
import torch
import time

def throughput(images, model, s):
    model.eval()
    batch_size, token_length = images[0].shape[0:2]
    for i in range(50):
        model.set_gate(s)
        model(*images)
    torch.cuda.synchronize()
    tic1 = time.time()
    for i in range(30):
        model.set_gate(s)
        model(*images)
    torch.cuda.synchronize()
    tic2 = time.time()
    print(f"batch_size {batch_size} token_length {token_length} throughput {30 * batch_size / (tic2 - tic1)}")
    MB = 1024.0 * 1024.0
    print('memory:', torch.cuda.max_memory_allocated() / MB)
    return (tic2 - tic1) / (30 * batch_size)

dims = [[int(0.25 * d), int(0.5 * d), int(0.75 * d), int(1 * d)] for d in [64, 128, 256, 512]]
weight = torch.load('resnet50.pth')
model = SlimResNet(Bottleneck, [3, 4, 6, 3], num_classes=1000, dims=dims)
for name, param in model.named_modules():
    print(name)
    if name.split('.')[-1] in ['0', '1', '2', '3']:
        name = name[:-9]
    for k in weight:
        weight_name = k.split('.')[-1]
        # if weight_name in ['weight', 'bias']:
        prefix = k[:-len(weight_name) - 1]
        if name == prefix:
            print(prefix, weight_name)
            try:
                model_shape = getattr(param, weight_name).shape
                weight_shape = weight[k].shape
                if model_shape == weight_shape:
                    setattr(param, weight_name, torch.nn.Parameter(weight[k]))
                else:
                    setattr(param, weight_name, torch.nn.Parameter(weight[k][:model_shape]))
            except:
                pass

model.to('cuda')
res_model = resnet50().to('cuda')
res_model.load_state_dict(weight)

data = torch.rand(1, 3, 224, 224).to('cuda')
output1 = model(data)
output2 = res_model(data)
print((output1 == output2).all())
# latency = []
# for s in range(4):
#     data = torch.rand(1, 3, 1080, 1080).to('cuda')
#     latency.append(throughput([data], model, s))
# plt.plot(range(4), latency)
# plt.show()

# latency = []
# for s in range(32, 1080, 96):
#     data = torch.rand(1, 3, s, s).to('cuda')
#     latency.append(throughput([data], model))
# plt.plot(range(32, 1080, 96), latency)
# plt.savefig('slim_resnet.png')
# plt.show()