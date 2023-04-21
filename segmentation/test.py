from src.models.efficientformer_model import main
from src.models.convnext import AdaConvNeXt, ConvNeXt
from src.models.convnext_model import ConvNextRGBD
import torch
if __name__ == '__main__':
    main()
    # weight = torch.load('trained_models/upernet_convnext_small_1k_512x512.pth')['state_dict']
    # weight = {k[9:]:v for k, v in weight.items() if k.split('.')[0] == 'backbone'}
    # model = ConvNeXt(dims=[96, 192, 384, 768], depths=[3, 3, 27, 3])
    # model.load_state_dict(weight)
    #
    # model = ConvNextRGBD()
    # torch.save(model.state_dict(), '1.pth')
    # model.encoder_rgb.load_state_dict(weight)
    # model.encoder_depth.load_state_dict(weight)