from models.segformer import mit_b3
from models.dynamic_model import Dynamic_Model
import torch
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
