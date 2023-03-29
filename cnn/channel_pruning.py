from utils.datasets.vggsound import VGGSound
import numpy as np
import torch
from model.slim_model import AVnet_Slim
import argparse
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")
# remove annoying librosa warning
class SoftTargetCrossEntropy(torch.nn.Module):
    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()
    def forward(self, x, target):
        loss = torch.sum(-target * torch.nn.functional.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()
def train_step(model, input_data, optimizer, criteria, soft_criteria, label):
    audio, image = input_data
    # Track history only in training
    losses = []
    max_length = model.channel_split - 1
    optimizer.zero_grad()
    if args.task == 'uniform':
        # outputs = []
        for mode in range(max_length, -1, -1):
            model.audio.set_mode(mode)
            model.image.set_mode(mode)
            output, _ = model(audio, image)
            # for j in range(len(outputs)):
            #     loss += soft_criteria(output, outputs[j])
            loss = criteria(output, label) * 1
            # loss += soft_criteria(output, output_distill)
            if mode < max_length:
                loss += torch.nn.functional.kl_div(
                        torch.nn.functional.log_softmax(output, dim=-1),
                        torch.nn.functional.log_softmax(output_distill, dim=-1),
                        reduction='batchmean',
                        log_target=True) * 0.5
            else:
                output_distill = output.detach()
            loss = loss * (mode + 1)/ (max_length+1)
            loss.backward()
            losses.append(loss.item())
    else:
        model.audio.set_mode('dynamic')
        model.image.set_mode('dynamic')
        output, _ = model(audio, image)

        loss = criteria(output, label)
        loss.backward()
        losses.append(loss.item())
    optimizer.step()

    return losses
def test_step(model, input_data, label):
    audio, image = input_data
    acc = []
    max_length = model.channel_split - 1
    if args.task == 'uniform':
        for mode in range(max_length, -1, -1):
            model.audio.set_mode(mode)
            model.image.set_mode(mode)
            output, comp = model(audio, image)
            acc.append((torch.argmax(output, dim=-1).cpu() == label).sum() / len(label))
    else:
        for i, mode in enumerate(['largest', 'random', 'random', 'smallest']):
            model.audio.set_mode(mode)
            model.image.set_mode(mode)
            output, comp = model(audio, image)
            acc.append((torch.argmax(output, dim=-1).cpu() == label).sum() / len(label))
    return acc, comp
def train(model, train_dataset, test_dataset):
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, num_workers=workers,
                           batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, num_workers=workers,
                                              batch_size=4, shuffle=False)
    best_acc = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=.0001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)
    criteria = torch.nn.CrossEntropyLoss()
    soft_criteria = SoftTargetCrossEntropy()
    for epoch in range(1):
        model.train()
        for idx, batch in enumerate(tqdm(train_loader)):
            audio, image, text, _ = batch
            losses = train_step(model, input_data=(audio.to(device), image.to(device)), optimizer=optimizer,
                        criteria=criteria, soft_criteria=soft_criteria, label=text.to(device))
            if idx % 200 == 0 and idx > 0:
                print('iteration:', str(idx), losses)
        scheduler.step()
        model.eval()
        acc = []
        comp = []
        with torch.no_grad():
            for batch in tqdm(test_loader):
                audio, image, text, _ = batch
                accuracy, computation = test_step(model, input_data=(audio.to(device), image.to(device)),
                                                  label=text)
                acc.append(accuracy)
                comp.append(computation)

        mean_acc = np.mean(acc, axis=0)
        print('epoch', epoch, mean_acc)
        avg_acc = np.mean(mean_acc)
        if avg_acc > best_acc:
            best_acc = avg_acc
            torch.save(model.state_dict(), 'slim_' + args.task + '_' +
                       str(epoch) + '_' + str(avg_acc) + '.pth')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='resnet')
    parser.add_argument('-t', '--task', default='random')
    parser.add_argument('-w', '--worker', default=4, type=int)
    parser.add_argument('-b', '--batch', default=32, type=int)
    args = parser.parse_args()
    workers = args.worker
    batch_size = args.batch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(0)
    model = AVnet_Slim(channel_split=8).to('cuda')
    model.load_state_dict(torch.load('slim_resnet_uniform_9_0.51721764.pth'), strict=False)
    # model_distill = AVnet(model='resnet', pretrained=False).to(device)
    # model.load_state_dict(torch.load('vanilla_resnet_AV_19_0.65678066.pth'), strict=False)
    # model_distill.load_state_dict(torch.load('slim_resnet_distill_9_0.5563375.pth'), strict=False)

    # model_distill.eval()
    dataset = VGGSound()
    len_train = int(len(dataset) * 0.8)
    len_test = len(dataset) - len_train
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len_train, len_test],
                                                                generator=torch.Generator().manual_seed(42))
    train(model, train_dataset, test_dataset)

