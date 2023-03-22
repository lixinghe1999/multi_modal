import torch
from model.dynamicvit_runtime import AVnet_Runtime
from model.dynamicvit_legacy import AVnet_Dynamic
from model.gate_model import AVnet_Gate
from model.dyn_slim import DSNet
import time
import argparse
from fvcore.nn import FlopCountAnalysis
import numpy as np
def rfft_flop_jit(inputs, outputs):
    """
    Count flops for the rfft/rfftn operator.
    """
    input_shape = inputs[0].type().sizes()
    B, H, W, C = input_shape
    N = H * W
    flops = N * C * np.ceil(np.log2(N))
    return flops
def calc_flops(model, input, show_details=False, ratios=None):
    with torch.no_grad():
        model.default_ratio = ratios
        fca1 = FlopCountAnalysis(model, input)
        handlers = {
            'aten::fft_rfft2': rfft_flop_jit,
            'aten::fft_irfft2': rfft_flop_jit,
        }
        fca1.set_op_handle(**handlers)
        flops1 = fca1.total()
        if show_details:
            print(fca1.by_module())
        print("#### GFLOPs: {} for ratio {}".format(flops1 / 1e9, ratios))
    return flops1 / 1e9

@torch.no_grad()
def throughput(images, model):
    model.eval()
    batch_size = images[0].shape[0]
    for i in range(50):
        model(*images)
    torch.cuda.synchronize()
    tic1 = time.time()
    for i in range(30):
        model(*images)
    torch.cuda.synchronize()
    tic2 = time.time()
    print(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
    MB = 1024.0 * 1024.0
    print('memory:', torch.cuda.max_memory_allocated() / MB)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', default='gate')
    parser.add_argument('-d', '--device', default='cuda')
    parser.add_argument('-b', '--batch', default=32, type=int)
    parser.add_argument('-e', '--exits', nargs='+', default='11 11')
    parser.add_argument('-l', '--locations', nargs='+', default='3 6 9')
    parser.add_argument('-r', '--rate', default=0.3, type=float)
    args = parser.parse_args()
    task = args.task
    device = torch.device(args.device)
    exits = torch.tensor([int(i) for i in args.exits.split()])
    pruning_loc = [int(i) for i in args.locations.split()]
    base_rate = args.rate
    token_ratio = [base_rate, base_rate ** 2, base_rate ** 3]

    audio = torch.randn(args.batch, 384, 128).to(device, non_blocking=True)
    image = torch.randn(args.batch, 3, 224, 224).to(device, non_blocking=True)

    if task == 'gate':

        model = AVnet_Gate().to(device)
        throughput([audio, image, exits], model)

    elif task == 'dynamic':

        model = AVnet_Runtime(pruning_loc=pruning_loc, token_ratio=token_ratio, pretrained=False).to(device)
        throughput([audio, image], model)
        # calc_flops(model, (audio, image))

        # model = AVnet_Runtime(pruning_loc=(), pretrained=False).to(device)
        # throughput([audio, image], model)

        model = AVnet_Dynamic(pruning_loc=(), pretrained=False).to(device)
        throughput([audio, image], model)
        # calc_flops(model, (audio, image))

    elif task == 'dsnet':
        image = torch.randn(args.batch, 3, 224, 224).to(device)
        model = DSNet().to(device)
        model.set_mode('largest')
        throughput([image], model)

        model.set_mode('smallest')
        throughput([image], model)
        # model(*[image])