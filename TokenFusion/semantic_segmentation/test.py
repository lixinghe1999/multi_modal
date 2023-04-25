from models.dynamic_model import Dynamic_Model
import torch
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
if __name__ == '__main__':
    device = 'cuda'
    torch.seed()
    data1 = torch.rand(1, 3, 468, 625)
    data2 = torch.ones(1, 3, 468, 625)
    # model = mit_b3()
    # model.load_state_dict(weight)
    # output = model(data)
    # for o in output:
    #     print(o.shape)
    model = Dynamic_Model('mit_b2', pretrained=False).eval()
    output = model([data1, data2])
