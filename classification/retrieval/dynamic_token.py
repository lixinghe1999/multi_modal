import dataset
import numpy as np
import torch
import models
from models.dynamicvit_legacy import DynToken
from utils.losses import DistillDiffPruningLoss_dynamic
import warnings
from tqdm import tqdm
import argparse
import pandas as pd
warnings.filterwarnings("ignore")
def test_epickitchen(model, test_loader):
    model.eval()
    acc = {'verb':[], 'noun':[], 'action':[]}
    ratio = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_data = [batch[0].to(device), batch[1].to(device)]
            predict, _ = model(*input_data)

            predict_verb = (torch.argmax(predict['verb'], dim=-1).cpu() == batch[-1]['verb'])
            predict_noun = (torch.argmax(predict['noun'], dim=-1).cpu() == batch[-1]['noun'])
            predict_action = torch.logical_and(predict_verb, predict_noun)
            acc['verb'].append( predict_verb.sum() / len(batch[-1]['verb']))
            acc['noun'].append( predict_noun.sum() / len(batch[-1]['noun']))
            acc['action'].append( predict_action.sum() / len(batch[-1]['verb']))
            ratio.append(model.ratio)

    print('verb =', np.mean(acc['verb']), 'noun =', np.mean(acc['noun']))
    mean_ratio = np.mean(ratio, axis=0)
    print('modality-1 balance:', mean_ratio[0], 'modality-2 ratio:', mean_ratio[1], 'difference:', mean_ratio[2])
    return np.mean(acc['action']) 
def train_step(model, input_data, optimizer, loss, label):
    outputs = model(*input_data)
    optimizer.zero_grad()
    loss, _ = loss(input_data, outputs, label)
    loss.backward()
    optimizer.step()
    return loss.item()
def profile(model, test_loader, test_epoch):
    model.eval()
    token_ratio = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
    acc = []
    with torch.no_grad():
        for ratio in token_ratio:
            model.token_ratio = ratio
            acc = test_epoch(model, test_loader)
            mean_acc = np.mean(acc)
            print('preserved ratio:', ratio, 'accuracy:', mean_acc)
def train(model, train_dataset, test_dataset, loss, test, test_epoch=test_epickitchen):
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, num_workers=workers, batch_size=batch_size, shuffle=True,
                                               drop_last=True, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, num_workers=workers, batch_size=1, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=.0001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)
    best_acc = 0
    if test:
        profile(model, test_loader, test_epoch)
    else:
        for epoch in range(10):
            model.train()
            for idx, batch in enumerate(tqdm(train_loader)):
                train_step(model, input_data=[batch[0].to(device), batch[1].to(device)],
                            optimizer=optimizer, loss=loss,    label=batch[-1])
            scheduler.step()
            model.eval()
            acc = []
            with torch.no_grad():
                acc = test_epoch(model, test_loader)
            print('epoch', epoch, 'acc =', acc)
            if acc > best_acc:
                best_acc = acc
                best_model = model.state_dict()
        torch.save(best_model, checkpoint_loc + 'DynToken_' + str(args.model) + '_' + str(args.scale) + '_' + str(best_acc) + '.pth')
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

    pruning_loc = (3, 6, 9)
    token_ratio = 0.7
    if args.dataset == 'EPICKitchen':
        checkpoint_loc = 'checkpoints_epic_kitchen/'
    else:
        checkpoint_loc = 'checkpoints_vggsound/'

    import h5py
    print('pre-load audio dict.....')
    audio_path = h5py.File('../split_EPIC_audio.hdf5', 'r')
    print('finish loading....')
    train_transform, val_transform = dataset.get_train_transform()
    train_dataset = getattr(dataset, args.dataset)(list_file=pd.read_pickle('EPIC_train.pkl'),               
                                                transform=train_transform, mode='train', audio_path=audio_path)
    val_dataset = getattr(dataset, args.dataset)(list_file=pd.read_pickle('EPIC_val.pkl'),               
                                                transform=val_transform, mode='val', audio_path=audio_path)
    model = DynToken(pruning_loc=pruning_loc, token_ratio=token_ratio, distill=True, backbone=getattr(models, args.model), scale=args.scale, pretrained=True, num_class=(97, 300)).to(device)
    model.load_state_dict(torch.load(checkpoint_loc + 'MBT_base_0.3744411.pth'), strict=False)

    teacher_model = DynToken(distill=True, backbone=getattr(models, args.model), scale=args.scale, pretrained=False,    num_class=(97, 300)).to(device)

    teacher_model.load_state_dict(torch.load(checkpoint_loc + 'MBT_base_0.3744411.pth'))
    teacher_model.eval()
    loss = DistillDiffPruningLoss_dynamic(teacher_model, torch.nn.CrossEntropyLoss(), clf_weight=1.0,
            keep_ratio=token_ratio, mse_token=True, ratio_weight=2, distill_weight=0.5)
    if args.test:
        model.load_state_dict(torch.load(checkpoint_loc + 'DynToken_MBT_base_0.37599015.pth'), strict=False)
        model.apply_merge(r=12)
    train(model, train_dataset, val_dataset, loss, args.test, test_epickitchen)

