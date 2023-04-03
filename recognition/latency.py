import torch
from model.dynamicvit_legacy import AVnet_Dynamic
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
        print("#### GFLOPs: {} for ratio {}".format(flops1 / 1e9, ratios))
    return flops1 / 1e9

@torch.no_grad()
def throughput(model, images):
    model.eval()
    batch_size = images[0].shape[0]
    print('start warm-up')
    for i in range(10):
        model(*images)
    print('finish warm-up')
    torch.cuda.synchronize()
    tic1 = time.time()
    for i in tqdm(range(30)):
        model(*images)
    torch.cuda.synchronize()
    tic2 = time.time()
    print(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
    MB = 1024.0 * 1024.0
    print('memory:', torch.cuda.max_memory_allocated() / MB)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', default='dynamic')
    parser.add_argument('-d', '--device', default='cuda')
    parser.add_argument('-b', '--batch', default=1, type=int)
    parser.add_argument('-e', '--exits', nargs='+', default='11 11')
    parser.add_argument('-l', '--locations', nargs='+', default='3 6 9')
    parser.add_argument('-r', '--rate', default=0.7, type=float)
    args = parser.parse_args()
    task = args.task
    device = torch.device(args.device)
    exits = torch.tensor([int(i) for i in args.exits.split()])
    pruning_loc = [int(i) for i in args.locations.split()]
    base_rate = args.rate
    token_ratio = [base_rate, base_rate ** 2, base_rate ** 3]
    print(args)
    audio = torch.randn(args.batch, 384, 128).to(device, non_blocking=True)
    image = torch.randn(args.batch, 3, 224, 224).to(device, non_blocking=True)

    model = AVnet_Dynamic(pruning_loc=pruning_loc, token_ratio=token_ratio, pretrained=False).to(device)
    model.eval()
    throughput(model, (audio, image))
    # calc_flops(model, (audio, image), show_details=False)
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
