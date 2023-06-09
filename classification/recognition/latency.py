import torch
from models.dynamicvit_legacy import DynToken
from models.early_exit import Early_Exit
from models.cnn_baseline import CNN
import models
import time
import argparse
import numpy as np
from tqdm import tqdm
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
    from fvcore.nn import FlopCountAnalysis, parameter_count_table
    model.eval()
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
        print("#### GFLOPs: {}".format(flops1 / 1e9))
        # print(parameter_count_table(model))
    return flops1 / 1e9

@torch.no_grad()
def throughput(model, images):
    model.eval()
    num_iterations = 100
    batch_size = images[0].shape[0]
    print('start warm-up')
    for i in range(30):
        model(*images)
    print('finish warm-up')
    torch.cuda.synchronize()
    tic1 = time.time()
    for i in range(num_iterations):
        model(*images)
    torch.cuda.synchronize()
    tic2 = time.time()
    print(f"batch_size {batch_size} latency {(tic2 - tic1) / (num_iterations * batch_size)}")
    # MB = 1024.0 * 1024.0
    # print('memory:', torch.cuda.max_memory_allocated() / MB)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='MBT')
    parser.add_argument('-s', '--scale', default='base')
    parser.add_argument('-c', '--cuda', default=1, type=int)
    parser.add_argument('-f', '--flops', action='store_true', default=False)
    parser.add_argument('-l', '--locations', nargs='+', default=[], type=int)
    parser.add_argument('-r', '--rate', default=0.6, type=float)

    parser.add_argument('-merge', '--merge', default=12, type=int)
    parser.add_argument('-e', '--exits', default=None, type=int)
    parser.add_argument('-cnn', action='store_true', default=False)
    args = parser.parse_args()
    print(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #torch.cuda.set_device(args.cuda)
    audio = torch.randn(1, 1, 256, 256).to(device, non_blocking=True)
    image = torch.randn(1, 3, 224, 224).to(device, non_blocking=True)
  
    if args.exits is not None:
        print('measure the early-exits latency')
        model = Early_Exit(getattr(models, args.model), args.scale, pretrained=False, num_class=(97, 300)).to(device)
        model.eval()
        throughput(model, (audio, image, args.exits))
        calc_flops(model, (audio, image, args.exits))
    elif args.cnn:
        print('measure the CNN latency')
        model = CNN(args.scale, pretrained=False, num_class=(97, 300)).to(device)
        model.eval()
        throughput(model, (audio, image))
        if args.flops:
            calc_flops(model, (audio, image))
    else:
        print('measure the dynamic token latency')
        pruning_loc = args.locations
        token_ratio = args.rate
    
        model = DynToken(pruning_loc=pruning_loc, token_ratio=token_ratio, distill=True, backbone=getattr(models, args.model), scale=args.scale, pretrained=False, num_class=(97, 300)).to(device)
        model.apply_merge(r=args.merge)
        model.eval()
        throughput(model, (audio, image))
        if args.flops:
            calc_flops(model, (audio, image))