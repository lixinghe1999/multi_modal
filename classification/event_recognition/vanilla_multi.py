import dataset
import numpy as np
import torch
import models

import argparse
import warnings
from tqdm import tqdm
import pandas as pd


warnings.filterwarnings("ignore")
# remove annoying librosa warning
def test_vggsound(model, test_loader):
    model.eval()
    acc = []; ratio = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_data = [batch[0].to(device), batch[1].to(device)]
            predict = model(*input_data)
            acc.append((torch.argmax(predict, dim=-1).cpu() == batch[-1]).sum() / len(batch[-1]))
            one_hot = torch.nn.functional.one_hot(batch[-1], num_classes=309)
            ratio1 = (torch.nn.functional.sigmoid(model.modality_weight[0]).cpu() * one_hot).sum(dim=-1).abs()
            ratio2 = (torch.nn.functional.sigmoid(model.modality_weight[1]).cpu() * one_hot).sum(dim=-1).abs()
            ratio.append((ratio1/ratio2).numpy())
    ratio = np.concatenate(ratio, axis=0)
    print('mean =', np.mean(ratio), 'variance =', np.var(ratio))
    return np.mean(acc)
def test_epickitchen(model, test_loader):
    model.eval()
    acc = {'verb':[], 'noun':[], 'action':[]}
    ratio =  {'verb':[], 'noun':[]}
    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_data = [batch[0].to(device), batch[1].to(device), batch[2].to(device)]
            predict = model(*input_data)

            predict_verb = (torch.argmax(predict['verb'], dim=-1).cpu() == batch[-1]['verb'])
            predict_noun = (torch.argmax(predict['noun'], dim=-1).cpu() == batch[-1]['noun'])
            predict_action = torch.logical_and(predict_verb, predict_noun)
            acc['verb'].append( predict_verb.sum() / len(batch[-1]['verb']))
            acc['noun'].append( predict_noun.sum() / len(batch[-1]['noun']))
            acc['action'].append( predict_action.sum() / len(batch[-1]['verb']))
            

            num_modal = len(model.modality_weight)//2
            one_hot = {'verb': torch.nn.functional.one_hot(batch[-1]['verb'], num_classes=97), 'noun': torch.nn.functional.one_hot(batch[-1]['noun'], num_classes=300)}
            for i, key in enumerate(['verb', 'noun']):
                r = []
                for j in range(num_modal):
                    r.append((torch.nn.functional.sigmoid(model.modality_weight[i * num_modal + j]).cpu() * one_hot[key]).sum(dim=-1).abs().numpy())
                r = np.column_stack(r)
                r = r/np.sum(r, axis=1, keepdims=True)
                ratio[key].append(r)
    ratio['verb'] = np.concatenate(ratio['verb'], axis=0)
    ratio['noun'] = np.concatenate(ratio['noun'], axis=0)
    print('mean verb =', np.mean(ratio['verb'], axis=0), 'variance verb =', np.var(ratio['noun'], axis=0))
    print('mean noun =', np.mean(ratio['noun'], axis=0), 'variance verb =', np.var(ratio['noun'], axis=0))
    print('verb =', np.mean(acc['verb']), 'noun =', np.mean(acc['noun']))
    return np.mean(acc['action']) 

def train_step(model, input_data, optimizer, criteria, label, device):
    output = model(*input_data)
    # Backward
    optimizer.zero_grad()
    if isinstance(label, dict):
        loss = 0
        for key in label:
            loss += criteria(output[key], label[key].to(device))
    else:
        loss = criteria(output, label.to(device))
    loss.backward()
    optimizer.step()
    return loss.item()
def train(model, train_dataset, test_dataset, test=False, test_epoch=test_vggsound):
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, num_workers=workers, batch_size=batch_size,  shuffle=True, drop_last=True, pin_memory=False)
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
            loss = 0
            for i, batch in enumerate(tqdm(train_loader)):
                input_data = [batch[0].to(device), batch[1].to(device), batch[2].to(device)]
                loss += train_step(model, input_data=input_data, optimizer=optimizer,
                            criteria=criteria, label=batch[-1], device=device)
                if i % 600 == 0 and i > 0:
                    print('loss =', loss/(i+1))
            scheduler.step()
            acc = test_epoch(model, test_loader)
            print('epoch', epoch, 'acc =', acc)
            if acc > best_acc:
                best_acc = acc
                best_model = model.state_dict()
        torch.save(best_model, args.model + '_'  + args.scale + '_' + str(best_acc) + '.pth')
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='MBT', type=str)
    parser.add_argument('-d', '--dataset', default='EPICKitchen', type=str) # VGGSound, EPICKitchen
    parser.add_argument('-w', '--worker', default=4, type=int)
    parser.add_argument('-b', '--batch', default=16, type=int)
    parser.add_argument('-s', '--scale', default='base', type=str)
    parser.add_argument('-c', '--cuda', default=0, type=int)
    parser.add_argument('-test', action='store_true', default=False)
    args = parser.parse_args()
    workers = args.worker
    batch_size = args.batch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(args.cuda)

    if args.dataset == 'VGGSound':
        full_dataset = getattr(dataset, args.dataset)()
        len_train = int(len(full_dataset) * 0.8)
        len_test = len(full_dataset) - len_train
        train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [len_train, len_test], generator=torch.Generator().manual_seed(42))
        model = getattr(models, args.model)(args.scale, pretrained=True, num_class=309).to(device)
        if args.test:
            model.load_state_dict(torch.load('MBT_base_0.6702001.pth'))
        train(model, train_dataset, test_dataset, args.test, test_vggsound)
    else:
        import h5py
        print('pre-load audio dict.....')
        audio_path = h5py.File('../split_EPIC_audio.hdf5', 'r')
        print('finish loading....')
        train_transform, val_transform = dataset.get_train_transform()
        train_dataset = getattr(dataset, args.dataset)(list_file=pd.read_pickle('EPIC_train.pkl'),               
                                                 transform=train_transform, mode='train', audio_path=audio_path)
        val_dataset = getattr(dataset, args.dataset)(list_file=pd.read_pickle('EPIC_val.pkl'),               
                                                 transform=val_transform, mode='val', audio_path=audio_path)
        model = getattr(models, args.model)(args.scale, pretrained=True, num_class=(97, 300)).to(device)
        model.audio.load_state_dict(torch.load('A_0.16073199.pth'), strict=False)
        model.image.load_state_dict(torch.load('V_0.30203858.pth'), strict=False)
        if args.test:
            model.load_state_dict(torch.load('MBT_small_0.33567417.pth'))
        train(model, train_dataset, val_dataset, args.test, test_epickitchen)
    

    
    