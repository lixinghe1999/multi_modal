import dataset
import numpy as np
import torch
import models

import argparse
import warnings
from tqdm import tqdm
import pandas as pd
from utils.cos_similar import sharded_cross_view_inner_product
from utils.losses import MaxMarginRankingLoss, loss_infoNCE
from utils.metric import t2v_metrics


warnings.filterwarnings("ignore")
# remove annoying librosa warning
def test_epickitchen_classification(model, test_loader):
    model.eval()
    acc = {'verb':[], 'noun':[], 'action':[]}
    ratio =  {'verb':[], 'noun':[]}
    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_data = [batch[0].to(device), batch[1].to(device)]
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
    print('ratio verb =', np.mean(ratio['verb'], axis=0), 'ratio noun =', np.mean(ratio['noun'], axis=0))
    print('verb =', np.mean(acc['verb']), 'noun =', np.mean(acc['noun']))
    return np.mean(acc['action']) 

def test_epickitchen_retrieval(model, test_loader):
    model.eval()
    metric_all = {'R1': [], 'R5': [], 'MeanR': []}
    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_data = [batch[0].to(device), batch[1].to(device)]
            output = model(*input_data)['feature']
            output = output @ batch[-1]['feature'].to(dtype=output.dtype, device=device).t()
            metric = t2v_metrics(output.cpu().numpy())
            for key in metric_all:
                metric_all[key].append(metric[key])
    for key in metric_all:
        metric_all[key] = np.mean(metric_all[key])
    print(metric_all)
    return metric_all['R1']

def train_step(model, input_data, optimizer, label, device):
    output = model(*input_data)
    # Backward
    optimizer.zero_grad()
    if isinstance(label, dict):
        loss = 0
        loss += torch.nn.functional.cross_entropy(output['verb'], label['verb'].to(device))
        loss += torch.nn.functional.cross_entropy(output['noun'], label['noun'].to(device))
        loss += loss_infoNCE(output['feature'], label['feature'].to(device))
    else:
        loss = torch.nn.functional.cross_entropy(output, label.to(device))
    loss.backward()
    optimizer.step()
    return loss.item()
def train(model, train_dataset, test_dataset, test=False):
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, num_workers=workers, batch_size=batch_size,  shuffle=True, drop_last=True, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, num_workers=workers, batch_size=batch_size, shuffle=False)
    best_acc = 0
    optimizer = torch.optim.Adam(model.head_feature.parameters(), lr=.00001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)
    if test:
        acc_cls = test_epickitchen_classification(model, test_loader)
        acc_ret = test_epickitchen_retrieval(model, test_loader)
        print('classification acc =', acc_cls, 'retrieval acc =', acc_ret)
    else:
        for epoch in range(10):
            model.train()
            for i, batch in enumerate(tqdm(train_loader)):
                input_data = [batch[0].to(device), batch[1].to(device)]
                loss = train_step(model, input_data=input_data, optimizer=optimizer, label=batch[-1], device=device)
                if i % 100 == 0 and i > 0:
                    print(loss)
            scheduler.step()
            acc_cls = test_epickitchen_classification(model, test_loader)
            acc_ret = test_epickitchen_retrieval(model, test_loader)
            print('epoch', epoch, 'classification acc =', acc_cls, 'retrieval acc =', acc_ret)
            if acc_cls > best_acc:
                best_acc = acc_cls
                best_model = model.state_dict()
        torch.save(best_model, args.model + '_'  + args.scale + '_' + str(best_acc) + '.pth')
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
    checkpoint_loc = 'checkpoints_epic_kitchen/'
    if args.dataset == 'EPICKitchen':
        import h5py
        print('pre-load audio dict.....')
        audio_path = h5py.File('../split_EPIC_audio.hdf5', 'r')
        print('finish loading....')
        train_transform, val_transform = dataset.get_train_transform()
        train_dataset = getattr(dataset, args.dataset)(list_file=pd.read_pickle('EPIC_train_100.pkl'),               
                                                 transform=train_transform, mode='train', audio_path=audio_path,
                                                 narration_embedding='train_vector.npy')
        val_dataset = getattr(dataset, args.dataset)(list_file=pd.read_pickle('EPIC_val_100.pkl'),               
                                                 transform=val_transform, mode='val', audio_path=audio_path,
                                                 narration_embedding='test_vector.npy')
        model = getattr(models, args.model)(args.scale, pretrained=True, num_class=(97, 300)).to(device)
        # model.audio.load_state_dict(torch.load(checkpoint_loc+'A_0.16073199.pth'), strict=False)
        # model.image.load_state_dict(torch.load(checkpoint_loc+'V_0.30203858.pth'), strict=False)
        # model.flow.load_state_dict(torch.load(checkpoint_loc+'F_0.02931882.pth'), strict=False)
        model.load_state_dict(torch.load(checkpoint_loc + 'MBT_base_0.3744411.pth'), strict=False)
        if args.test:
            model.load_state_dict(torch.load(checkpoint_loc + 'MBT_base_0.3744411.pth'), strict=False)
        train(model, train_dataset, val_dataset, args.test)
    

    
    