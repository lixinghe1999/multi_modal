from vggsound import VGGSound
import numpy as np
import torch
from models import AudioTransformerDiffPruning, VisionTransformerDiffPruning

import argparse
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")
# remove annoying librosa warning
def test_epoch(model, test_loader):
    model.eval()
    acc = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            if args.modal == 'A':
                input_data = [batch[0].to(device)]
            else:
                input_data = [batch[1].to(device)]
            predict = model(*input_data)
            acc.append((torch.argmax(predict, dim=-1).cpu() == batch[-1]).sum() / len(batch[-1]))
    return np.mean(acc)
def train_step(model, input_data, optimizer, criteria, label):
    output = model(*input_data)
    # Backward
    optimizer.zero_grad()
    if isinstance(output, tuple):
        output = output[0]
    loss = criteria(output, label)
    loss.backward()
    optimizer.step()
    return loss.item()
def train(model, train_dataset, test_dataset, test=False):
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, num_workers=workers, batch_size=batch_size, shuffle=True,
                                               drop_last=True, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, num_workers=workers, batch_size=batch_size, shuffle=False)
    best_acc = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=.0001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)
    criteria = torch.nn.CrossEntropyLoss()
    if test:
        acc = test_epoch(model, test_loader)
        print('test acc =', acc)
    else:
        for epoch in range(10):
            model.train()
            for _, batch in enumerate(tqdm(train_loader)):
                if args.modal == 'A':
                    input_data = [batch[0].to(device)]
                else:
                    input_data = [batch[1].to(device)]
                train_step(model, input_data=input_data, optimizer=optimizer,
                            criteria=criteria, label=batch[-1].to(device))
            scheduler.step()
            acc = test_epoch(model, test_loader)
            print('epoch', epoch, 'acc =', acc)
            if acc > best_acc:
                best_acc = acc
                best_model = model.state_dict()
        torch.save(best_model, args.modal + '_' + str(best_acc) + '.pth')
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--modal', default='A')
    parser.add_argument('-w', '--worker', default=4, type=int)
    parser.add_argument('-b', '--batch', default=4, type=int)
    parser.add_argument('-c', '--cuda', default=0, type=int)
    parser.add_argument('-test', action='store_true', default=False)
    args = parser.parse_args()
    workers = args.worker
    batch_size = args.batch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(args.cuda)

    config = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                    pruning_loc=())
    embed_dim = 768
    pretrained_weight = torch.load('pretrained/deit_base_patch16_224.pth')['model']
    if args.modal == 'A':
        model = AudioTransformerDiffPruning(config, pretrained=pretrained_weight).to(device)
    else:
        model = VisionTransformerDiffPruning(**config).to(device)
        model.load_state_dict(pretrained_weight, strict=False)
    if args.test: 
        if args.modal == 'A':
            model.load_state_dict(torch.load('checkpoints/vanilla_A_6_0.5303089942924621.pth'))
        else:
            model.load_state_dict(torch.load('checkpoints/vanilla_V_7_0.5041330446762449.pth'))
        print('loaded the pretrained model')
    dataset = VGGSound()
    len_train = int(len(dataset) * 0.8)
    len_test = len(dataset) - len_train
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len_train, len_test], generator=torch.Generator().manual_seed(42))
    train(model, train_dataset, test_dataset, test=args.test)