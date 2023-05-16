from vggsound import VGGSound
import numpy as np
import torch
import models
from models.dynamicvit_runtime import AVnet_Runtime
from models.dynamicvit_legacy import DynToken
from utils.losses import DistillDiffPruningLoss_dynamic
import time
import warnings
from tqdm import tqdm
import argparse
warnings.filterwarnings("ignore")

def train_step(model, input_data, optimizer, loss, label):
    # cumulative loss
    outputs = model(*input_data)
    optimizer.zero_grad()
    loss, _ = loss(input_data, outputs, label)
    loss.backward()
    optimizer.step()
    return loss.item()
def test_step(model, input_data, label):
    outputs = model(*input_data)
    output = outputs[0]
    r = model.ratio
    acc = (torch.argmax(output, dim=-1).cpu() == label).sum() / len(label)
    return acc.item(), r
def profile(model, test_loader):
    model.eval()
    token_ratio = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
    acc = []
    modality_ratio = []
    with torch.no_grad():
        for ratio in token_ratio:
            model.token_ratio = [ratio, ratio ** 2, ratio ** 3]
            torch.cuda.synchronize()
            tic1 = time.time()
            for batch in tqdm(test_loader):
                audio, image, text, _ = batch
                a, r = test_step(model, input_data=[audio.to(device), image.to(device)], label=text)
                acc.append(a)
                modality_ratio.append(r)
            torch.cuda.synchronize()
            tic2 = time.time()

            mean_acc = np.mean(acc)
            mean_ratio = np.mean(modality_ratio, axis=0)
            print('preserved ratio:', ratio)
            print('throughput:', len(test_loader) * batch_size / (tic2 - tic1))
            print('modality-1 balance:', mean_ratio[0], 'modality-2 balance:', mean_ratio[1])
            print('modality-wise ratio:', mean_ratio[2:])
            print('accuracy:', mean_acc)
def train(model, train_dataset, test_dataset, loss, test):
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, num_workers=workers, batch_size=batch_size, shuffle=True,
                                               drop_last=True, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, num_workers=workers, batch_size=1, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=.0001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)
    best_acc = 0
    if test:
        profile(model, test_dataset)
    else:
        for epoch in range(10):
            model.train()
            for idx, batch in enumerate(tqdm(train_loader)):
                audio, image, text = batch
                train_step(model, input_data=[audio.to(device), image.to(device)], optimizer=optimizer,
                            loss=loss, label=text.to(device))
            scheduler.step()
            model.eval()
            acc = []
            with torch.no_grad():
                for batch in tqdm(test_loader):
                    audio, image, text = batch
                    a, _ = test_step(model, input_data=[audio.to(device), image.to(device)], label=text)
                    acc.append(a)
            mean_acc = np.mean(acc)
            print('epoch', epoch)
            print('accuracy:', mean_acc)
            if mean_acc > best_acc:
                best_acc = mean_acc
                best_model = model.state_dict()
        torch.save(best_model, 'our_' + str(args.model) + '_' + str(args.scale) + '_' + str(best_acc) + '.pth')
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='MBT', type=str)
    parser.add_argument('-w', '--worker', default=4, type=int)
    parser.add_argument('-b', '--batch', default=4, type=int)
    parser.add_argument('-s', '--scale', default='base', type=str)
    parser.add_argument('-c', '--cuda', default=0, type=int)
    parser.add_argument('-test', action='store_true', default=False)
    args = parser.parse_args()
    workers = args.worker
    batch_size = args.batch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(args.cuda)

    pruning_loc = (3, 6, 9)
    base_rate = 0.7
    token_ratio = [base_rate, base_rate ** 2, base_rate ** 3]

    dataset = VGGSound()
    len_train = int(len(dataset) * 0.8)
    len_test = len(dataset) - len_train
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len_train, len_test], generator=torch.Generator().manual_seed(42))
    model = DynToken(pruning_loc=pruning_loc, token_ratio=token_ratio, distill=True, backbone=getattr(models, args.model), scale=args.scale, pretrained=False).to(device)
    model.load_state_dict(torch.load('MBT_base_0.6702001.pth'), strict=False)
    if args.test:
        model.load_state_dict(torch.load())

    teacher_model = DynToken(distill=True, backbone=getattr(models, args.model), scale=args.scale, pretrained=False).to(device)
    teacher_model.load_state_dict(torch.load('MBT_base_0.6702001.pth'))
    teacher_model.eval()
    loss = DistillDiffPruningLoss_dynamic(teacher_model, torch.nn.CrossEntropyLoss(), clf_weight=1.0,
            keep_ratio=token_ratio, mse_token=True, ratio_weight=2, distill_weight=0.5)
    train(model, train_dataset, test_dataset, loss, args.test)

