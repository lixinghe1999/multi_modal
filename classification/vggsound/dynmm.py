import time
from vggsound import VGGSound
import numpy as np
import torch
from model.dynmm import DynMM, Gate_MM, Gate_SM
import model 
from tqdm import tqdm
import argparse
import warnings
warnings.filterwarnings("ignore")

def train_step(model, input_data, optimizer, criteria, label, mode='dynamic'):
    audio, image = input_data
    # cumulative loss
    output_cache, output = model(audio, image, mode)
    optimizer.zero_grad()
    loss = criteria(output, label)
    loss.backward()
    optimizer.step()
    return loss.item()
def test_step(model, input_data, label, mode='gate'):
    audio, image = input_data
    output_cache, output = model(audio, image, mode)
    acc = (torch.argmax(output, dim=-1).cpu() == label).sum() / len(label)
    return acc.item(), len(output_cache['audio']) + len(output_cache['image'])
def profile(model, test_loader):
    model.eval()
    compress_level = []
    error = 0
    correct = 0
    with torch.no_grad():
        for batch in tqdm(test_loader):
            audio, image, text, _ = batch
            output_cache, output = model(audio.to(device), image.to(device), 'no_exit')
            gate_label = model.label(output_cache, text)
            gate_label = torch.argmax(gate_label[0], dim=-1, keepdim=True).cpu().numpy()
            if torch.argmax(output).cpu() == text:
                correct += 1
                compress_level.append(gate_label)
            elif gate_label[0] == 11 and gate_label[1] == 11:
                error += 1
            else:
                compress_level.append(gate_label)
    compress_level = np.concatenate(compress_level, axis=-1)
    compress_diff = np.abs(compress_level[0] - compress_level[1])
    compress_diff = np.bincount(compress_diff)
    compress_audio = np.bincount(compress_level[0])
    compress_image = np.bincount(compress_level[1])
    print("compression level difference:", compress_diff / len(test_loader))
    print("audio compression level:", compress_audio / len(test_loader))
    print("image compression level:", compress_image / len(test_loader))
    print("overall accuracy:", 1 - error / len(test_loader))
    print("final layer accuracy:", correct / len(test_loader))
def train(model, train_dataset, test_dataset, test=False):
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, num_workers=workers, batch_size=batch_size, shuffle=True,
                                               drop_last=True, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, num_workers=workers, batch_size=1, shuffle=False)
    optimizer = torch.optim.Adam(model.get_parameters(), lr=.0001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)
    best_acc = 0
    if test:
        profile(model, test_loader)
    for epoch in range(10):
        model.train()
        for idx, batch in enumerate(tqdm(train_loader)):
            audio, image, text = batch
            optimizer.zero_grad()
            output = model.gate_train(audio.to(device), image.to(device), text.to(device))
            if idx % 100 == 0 and idx > 0:
                print(output)
            optimizer.step()
        scheduler.step()
        model.eval()
        acc = [0] * 2 * model.len_blocks; count = [0] * 2 * model.len_blocks
        with torch.no_grad():
            for batch in tqdm(test_loader):
                audio, image, text = batch
                a, e = test_step(model, input_data=(audio.to(device), image.to(device)), label=text, mode='gate')
                acc[e-1] += a
                count[e-1] += 1
        mean_acc = []
        for i in range(len(acc)):
            if count[i] == 0:
                mean_acc.append(0)
            else:
                mean_acc.append(acc[i]/count[i])
        acc_all = np.round(mean_acc, 3)
        acc_avg = np.round(np.sum(acc) / np.sum(count), 3)
        comp_all = np.round(np.array(count) / np.sum(count), 3)
        comp_avg = np.mean(comp_all * np.linspace(1, 24, 24))
        print('epoch', epoch, 'trained gate exit')
        print('accuracy for early-exits:', acc_all.tolist())
        print('mean accuracy ', acc_avg)
        print('compression level distribution:', comp_all.tolist())
        print('mean compression level:', comp_avg )
        if acc_avg > best_acc:
            best_acc = acc_avg
            best_model = model.state_dict()
    torch.save(best_model, 'dynmm_' + str(args.modal) + '_' + str(acc_avg) + '_' + str(comp_avg) + '.pth')
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='MBT', type=str)
    parser.add_argument('-modal', '--modal', default='sm', type=str)
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
    model = DynMM(getattr(model, args.model), args.modal, args.scale, pretrained=True).to(device)
    model.load_state_dict(torch.load('MBT_base_0.6702001.pth'), strict=False)
    if args.test:
        model.load_state_dict(torch.load())

    dataset = VGGSound()
    len_train = int(len(dataset) * 0.8)
    len_test = len(dataset) - len_train
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len_train, len_test], generator=torch.Generator().manual_seed(42))
    train(model, train_dataset, test_dataset, args.test)


