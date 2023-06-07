import dataset
import numpy as np
import torch
from models.early_exit import Early_Exit
import models
import warnings
from tqdm import tqdm
import argparse
import pandas as pd

warnings.filterwarnings("ignore")
# remove annoying librosa warning
def test_vggsound(model, test_loader):
    model.eval()
    acc = []; exits = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_data = [batch[0].to(device), batch[1].to(device)]
            predict = model(*input_data)
            early_acc, early_exit = test_step(predict, batch[-1])
            acc.append(early_acc)
            exits.append(early_exit)
    return np.mean(acc, axis=0), np.mean(exits, axis=0)
def test_epickitchen(model, test_loader):
    model.eval()
    acc = {'verb':[], 'noun':[], 'action':[]}
    exit = {'verb':[], 'noun':[], 'action':[]}
    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_data = [batch[0].to(device), batch[1].to(device)]
            predict = model(*input_data)
            predict_verb = [o['verb'] for o in predict]
            predict_noun = [o['noun'] for o in predict]
            verb_acc, verb_exit = test_step(predict_verb, batch[-1]['verb'])
            noun_acc, noun_exit = test_step(predict_noun, batch[-1]['noun'])
            action_acc = np.logical_and(verb_acc, noun_acc)
            acc['verb'].append(verb_acc)
            acc['noun'].append(noun_acc)
            acc['action'].append(action_acc)
            
            exit['verb'].append(verb_exit)
            exit['noun'].append(noun_exit)
            exit['action'].append(np.max(np.column_stack([verb_exit, noun_exit]), axis=1))
    print('verb: ', np.mean(acc['verb'], axis=0).tolist(), 'noun:', np.mean(acc['noun'], axis=0).tolist())
    return np.mean(acc['action'], axis=0), np.mean(exit['action'], axis=0)
def train_step(model, input_data, optimizer, criteria, label, device):
    outputs = model(*input_data)
    optimizer.zero_grad()
    loss = 0
    if isinstance(label, dict):
        loss = 0
        for i, output in enumerate(outputs):
            for key in label:
                loss += (i+1)/len(outputs) * criteria(output[key], label[key].to(device)) * 0.5
    else:
        for i, output in enumerate(outputs):
            loss += (i+1)/len(outputs)  * criteria(output, label.to(device))
    loss.backward()
    optimizer.step()
    return loss.item()
def test_step(outputs, label):
    thresholds = [0.6, 0.7, 0.8, 0.9]
    early_acc = np.zeros((len(thresholds))); early_exit = np.zeros((len(thresholds)))
    for j, thres in enumerate(thresholds):
        for i, output in enumerate(outputs):
            early_acc[j] = torch.argmax(output, dim=-1).cpu() == label
            early_exit[j] = (i + 1) / len(outputs)
            max_confidence = torch.max(torch.softmax(output, dim=-1))
            if max_confidence > thres:
                break 
    return early_acc, early_exit

def train(model, train_dataset, test_dataset, test=False, test_epoch=test_vggsound):
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, num_workers=workers, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, num_workers=1, batch_size=1, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=.0001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.2)
    criteria = torch.nn.CrossEntropyLoss()
    best_acc = 0
    if test:
        acc, exits = test_epoch(model, test_loader)
        print('accuracy:', acc.tolist(), 'computation:', exits.tolist())
    else:
        for epoch in range(10):
            model.train()
            for idx, batch in enumerate(tqdm(train_loader)):
                loss = train_step(model, input_data=(batch[0].to(device), batch[1].to(device)), optimizer=optimizer,
                        criteria=criteria, label=batch[-1], device=device)
                if idx % 100 == 0 and idx != 0:
                    print(loss)
            scheduler.step()
            acc, exits = test_epoch(model, test_loader)
            print('epoch:', epoch, 'accuracy:', acc.tolist(), 'computation:', exits.tolist())
            if np.mean(acc) > best_acc:
                best_acc = np.mean(acc)
                best_model = model.state_dict()
        torch.save(best_model, 'exit' + '_' + args.model + '_' + args.scale + '_' + str(best_acc) + '.pth')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='MBT', type=str)
    parser.add_argument('-d', '--dataset', default='EPICKitchen', type=str) # VGGSound, EPICKitchen
    parser.add_argument('-w', '--worker', default=4, type=int)
    parser.add_argument('-b', '--batch', default=32, type=int)
    parser.add_argument('-s', '--scale', default='base', type=str)
    parser.add_argument('-c', '--cuda', default=0, type=int)
    parser.add_argument('-test', action='store_true', default=False)
    args = parser.parse_args()
    workers = args.worker
    batch_size = args.batch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(args.cuda)
    
    if args.dataset == 'EPICKitchen':
        checkpoint_loc = 'checkpoints_epic_kitchen/'
    else:
        checkpoint_loc = 'checkpoints_vggsound/'

    if args.dataset == 'VGGSound':
        model = Early_Exit(getattr(models, args.model), args.scale, pretrained=True, num_class=309).to(device)
        if args.test:
            model.load_state_dict(torch.load())

        full_dataset = getattr(dataset, args.dataset)()
        len_train = int(len(full_dataset) * 0.8)
        len_test = len(full_dataset) - len_train
        train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [len_train, len_test], generator=torch.Generator().manual_seed(42))
        if args.test:
            model.load_state_dict(torch.load(checkpoint_loc + ''))
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
        model = Early_Exit(getattr(models, args.model), args.scale, pretrained=True, num_class=(97, 300)).to(device)
        model.audio.load_state_dict(torch.load(checkpoint_loc + 'A_0.16073199.pth'), strict=False)
        model.image.load_state_dict(torch.load(checkpoint_loc + 'V_0.30203858.pth'), strict=False)
        if args.test:
            model.load_state_dict(torch.load(checkpoint_loc + 'exit_MBT_base_0.367.pth'))
        train(model, train_dataset, val_dataset, args.test, test_epickitchen)

