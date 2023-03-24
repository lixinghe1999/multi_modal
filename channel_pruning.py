import torchvision.models
from utils.datasets.vggsound import VGGSound
import numpy as np
import torch
from model.slim_model import AVnet_Slim
import argparse
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")
# remove annoying librosa warning
def step(model, input_data, optimizer, criteria, label):
    audio, image = input_data
    # Track history only in training
    for mode in range(4):
        model.audio.set_mode(mode)
        model.image.set_mode(mode)
        output = model(audio, image)
    # Backward
    optimizer.zero_grad()
    loss = criteria(output, label)
    loss.backward()
    optimizer.step()
    return loss
def train(model, train_dataset, test_dataset):
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, num_workers=workers, batch_size=batch_size, shuffle=True,
                                               drop_last=True, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, num_workers=workers, batch_size=batch_size, shuffle=False)
    best_acc = 0
    if args.task == 'AV':
        optimizer = torch.optim.Adam(model.parameters(), lr=.0001, weight_decay=1e-4)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=.0001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)
    criteria = torch.nn.CrossEntropyLoss()
    for epoch in range(20):
        model.train()
        for idx, batch in enumerate(tqdm(train_loader)):
            audio, image, text, _ = batch
            step(model, input_data=(audio.to(device), image.to(device)), optimizer=optimizer,
                        criteria=criteria, label=text.to(device))
        scheduler.step()
        model.eval()
        acc = []
        with torch.no_grad():
            for batch in tqdm(test_loader):
                audio, image, text, _ = batch
                predict = model(audio.to(device), image.to(device))
                acc.append((torch.argmax(predict, dim=-1).cpu() == text).sum() / len(text))
        print('epoch', epoch, np.mean(acc))
        if np.mean(acc) > best_acc:
            best_acc = np.mean(acc)
            torch.save(model.state_dict(), 'vanilla_' + args.model + '_' + args.task + '_' +
                       str(epoch) + '_' + str(np.mean(acc)) + '.pth')
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='resnet')
    parser.add_argument('-t', '--task', default='distill')
    parser.add_argument('-w', '--worker', default=4, type=int)
    parser.add_argument('-b', '--batch', default=32, type=int)
    args = parser.parse_args()
    workers = args.worker
    batch_size = args.batch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(0)
    if args.model == 'resnet':
        model = AVnet_Slim().to('cuda')
        model.load_state_dict(torch.load('vanilla_resnet_AV_19_0.65678066.pth'), strict=False)
    dataset = VGGSound()
    len_train = int(len(dataset) * 0.8)
    len_test = len(dataset) - len_train
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len_train, len_test], generator=torch.Generator().manual_seed(42))
    train(model, train_dataset, test_dataset)

