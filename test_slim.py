import matplotlib.pyplot as plt
from model.ds_net import SlimBlock, SlimResNet
from model.resnet_model import Bottleneck, resnet50
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
model = SlimResNet(SlimBlock, [3, 4, 6, 3], num_classes=1000, dims=dims).to('cuda')
model.load_state_dict(torch.load('resnet50.pth'), strict=False)
model.set_mode(1)
print(model(torch.rand(1, 3, 224, 224).to('cuda')).shape)
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