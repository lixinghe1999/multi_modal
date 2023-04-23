from models.dynamic_segformer import mit_b3
from models.dynamic_model import Dynamic_Model
import torch
if __name__ == '__main__':
    device = 'cuda'
    weight = torch.load('../../assets/mit_b3.pth')
    data = torch.rand(1, 3, 468, 625)
    # model = mit_b3()
    # model.load_state_dict(weight)
    # output = model(data)
    # for o in output:
    #     print(o.shape)
    model = Dynamic_Model('mit_b3')
    model.encoder_rgb.load_state_dict(weight)
    model.encoder_depth.load_state_dict(weight)
    output = model(data, data)
    for o in output:
        print(o.shape)