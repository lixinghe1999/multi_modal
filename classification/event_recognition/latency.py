import torch
from models.dynamicvit_legacy import DynToken
from models.early_exit import Early_Exit
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
    batch_size = images[0].shape[0]
    print('start warm-up')
    for i in range(30):
        model(*images)
    print('finish warm-up')
    torch.cuda.synchronize()
    tic1 = time.time()
    for i in range(100):
        model(*images)
    torch.cuda.synchronize()
    tic2 = time.time()
    print(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
    MB = 1024.0 * 1024.0
    print('memory:', torch.cuda.max_memory_allocated() / MB)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='MBT')
    parser.add_argument('-s', '--scale', default='base')
    parser.add_argument('-c', '--cuda', default=1, type=int)
    parser.add_argument('-b', '--batch', default=1, type=int)
    parser.add_argument('-l', '--locations', nargs='+', default='3 6 9')
    parser.add_argument('-r', '--rate', default=0.6, type=float)

    parser.add_argument('-e', '--exits', default=None, type=int)
    args = parser.parse_args()
    print(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #torch.cuda.set_device(args.cuda)
    audio = torch.randn(args.batch, 1, 256, 256).to(device, non_blocking=True)
    image = torch.randn(args.batch, 3, 224, 224).to(device, non_blocking=True)
  
    if args.exits is not None:
        print('measure the early-exits latency')
        model = Early_Exit(getattr(models, args.model), args.scale, pretrained=False, num_class=(97, 300)).to(device)
        model.eval()
        throughput(model, (audio, image, args.exits))
        calc_flops(model, (audio, image, args.exits), show_details=False)
    else:
        print('measure the dynamic token latency')
        pruning_loc = [int(i) for i in args.locations.split()]
        # pruning_loc = ()
        token_ratio = args.rate
    
        model = DynToken(pruning_loc=pruning_loc, token_ratio=token_ratio, distill=True, backbone=getattr(models, args.model), scale=args.scale, pretrained=False, num_class=(97, 300)).to(device)
        model.apply_merge(r=12)
        model.eval()
        throughput(model, (audio, image))
        calc_flops(model, (audio, image), show_details=False)

    # torch.save(model.state_dict(), 'dynamic_token.pth')
    # torch.onnx.export(model, (audio, image), 'dynamic.onnx', input_names=['input_1', 'input_2'],
    #                  output_names=['output'], export_params=True)

    # import onnx
    # onnx_model = onnx.load("dynamic.onnx")
    # onnx.checker.check_model(onnx_model)


    # import onnxruntime
    # torch_out = model(audio, image)
    # ort_session = onnxruntime.InferenceSession("dynamic.onnx")
    # ort_outs = ort_session.run(None,  {
    # "input_1": audio.detach().cpu().numpy().astype(np.float32),
    # "input_2": image.detach().cpu().numpy().astype(np.float32)
    # })
    # # compare ONNX Runtime and PyTorch results
    # np.testing.assert_allclose(torch_out.detach().cpu().numpy(), ort_outs[0], rtol=1e-02, atol=1e-05)
    # print("Exported model has been tested with ONNXRuntime, and the result looks good!")
