from audio_visual.vggsound import VGGSound
from har.adbox import ADBox
import numpy as np
import torch
from audio_visual.model import AVnet
from audio_visual.model import AudioTransformerDiffPruning, VisionTransformerDiffPruning
from har.model import HARnet
from har.model import AdaConvNeXt

import argparse
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")
# remove annoying librosa warning
def step(model, input_data, optimizer, criteria, label):
    output = model(*input_data)
    # Backward
    optimizer.zero_grad()
    if isinstance(output, tuple):
        output = output[0]
    loss = criteria(output, label)
    loss.backward()
    optimizer.step()
    return loss.item()
def train(model, train_dataset, test_dataset):
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, num_workers=workers, batch_size=batch_size, shuffle=True,
                                               drop_last=True, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, num_workers=workers, batch_size=batch_size, shuffle=False)
    best_acc = 0
    if args.task == 'AV':
        for param in model.audio.parameters():
            param.requires_grad = False
        for param in model.image.parameters():
            param.requires_grad = False
        optimizer = torch.optim.Adam(model.parameters(), lr=.0001, weight_decay=1e-4)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=.0001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)
    criteria = torch.nn.CrossEntropyLoss()
    for epoch in range(20):
        model.train()
        for idx, batch in enumerate(tqdm(train_loader)):
            break
            if args.task == 'vggsound':
                if args.modal == 'AV':
                    input_data = [batch[0].to(device), batch[1].to(device)]
                elif args.modal == 'A':
                    input_data = [batch[0].to(device)]
                else:
                    input_data = [batch[1].to(device)]
            else:
                if args.modal == 'D':
                    input_data = [batch[0].to(device)]
                else:
                    input_data = [batch[0].to(device), batch[1].to(device), batch[2].to(device)]
            step(model, input_data=input_data, optimizer=optimizer,
                        criteria=criteria, label=batch[-1].to(device))
        scheduler.step()
        model.eval()
        acc = []
        ratio = []
        with torch.no_grad():
            for batch in tqdm(test_loader):
                if args.task == 'vggsound':
                    if args.modal == 'AV':
                        input_data = [batch[0].to(device), batch[1].to(device)]
                    elif args.modal == 'A':
                        input_data = [batch[0].to(device)]
                    else:
                        input_data = [batch[1].to(device)]
                else:
                    if args.modal == 'D':
                        input_data = [batch[0].to(device)]
                    else:
                        input_data = [batch[0].to(device), batch[1].to(device), batch[2].to(device)]
                predict = model(*input_data)
                acc.append((torch.argmax(predict, dim=-1).cpu() == batch[-1]).sum() / len(batch[-1]))

                one_hot_label = torch.nn.functional.one_hot(batch[-1].to(device), num_classes=309)
                split_prediction = [(model.modality_weight[0] * one_hot_label).sum(dim=-1).abs(),
                                    (model.modality_weight[1] * one_hot_label).sum(dim=-1).abs()]
                print(split_prediction)
                ratio.append(r)
        import pickle
        file = open(r"modality_weight.pkl", "wb")
        pickle.dump(ratio, file)  # 保存list到文件
        file.close()

        print('epoch', epoch, np.mean(acc))
        if np.mean(acc) > best_acc:
            best_acc = np.mean(acc)
    torch.save(model.state_dict(), 'vanilla_' + args.task + '_' + args.modal + '_' + str(epoch) +
               '_' + str(np.mean(acc)) + '.pth')
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', default='vggsound')
    parser.add_argument('-m', '--modal', default='AV')
    parser.add_argument('-w', '--worker', default=4, type=int)
    parser.add_argument('-b', '--batch', default=4, type=int)
    args = parser.parse_args()
    workers = args.worker
    batch_size = args.batch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(0)
    if args.task == 'vggsound':
        config = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                      pruning_loc=())
        embed_dim = 768
        if args.modal == 'AV':
            model = AVnet().to(device)
            model.load_state_dict(torch.load('vanilla_vit_AV_8_0.6951769.pth'))
            # model.audio.load_state_dict(torch.load('vanilla_A_6_0.5303089942924621.pth'))
            # model.image.load_state_dict(torch.load('vanilla_V_7_0.5041330446762449.pth'))
        elif args.modal == 'A':
            model = AudioTransformerDiffPruning(config, imagenet_pretrain=True).to(device)
        else:
            model = VisionTransformerDiffPruning(**config).to(device)
            model.load_state_dict(torch.load('assets/deit_base_patch16_224.pth')['model'], strict=False)

        dataset = VGGSound()
        len_train = int(len(dataset) * 0.8)
        len_test = len(dataset) - len_train
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len_train, len_test], generator=torch.Generator().manual_seed(42))
        train(model, train_dataset, test_dataset)

    if args.task == 'adbox':
        if args.modal == 'D':
            model = AdaConvNeXt(in_chans=16, pruning_loc=[0], num_classes=14, depths=[3, 3, 27, 3]).to(device)
            # weight = torch.load('assets/convnext-s-0.7.pth')['model']
            # weight = {k: v for k, v in weight.items() if 'downsample_layers' not in k and 'head' not in k}
            # model.load_state_dict(weight, strict=False)
        else:
            model = HARnet(pretrained=True).to(device)

        train_dataset1 = ADBox('../dataset/adbox', split='train1')
        train_dataset2 = ADBox('../dataset/adbox', split='train2')
        train_dataset = torch.utils.data.ConcatDataset([train_dataset1, train_dataset2])
        test_dataset = ADBox('../dataset/adbox', split='test')
        train(model, train_dataset, test_dataset)

