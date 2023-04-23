from models.dynamic_segformer import mit_b3
import torch
if __name__ == '__main__':
    device = 'cuda'
    model = mit_b3()
    weight = torch.load('../../assets/mit_b3.pth')
    model.load_state_dict(weight)
    data = torch.rand(1, 3, 468, 625)
    output = model(data)
    for o in output:
        print(o.shape)