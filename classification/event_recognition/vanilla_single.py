import dataset
import numpy as np
import torch
from models import AudioTransformer, VisionTransformer
import pandas as pd
import argparse
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")
# remove annoying librosa warning
def test_vggsound(model, test_loader):
    model.eval()
    acc = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            if args.modal == 'A':
                predict = model(batch[0].to(device))
            elif args.modal == 'V':
                predict = model(batch[1].to(device))
            else:
                predict = model(batch[2].to(device))
            acc.append((torch.argmax(predict, dim=-1).cpu() == batch[-1]).sum() / len(batch[-1]))
    return np.mean(acc)
def test_epickitchen(model, test_loader):
    model.eval()
    acc = {'verb':[], 'noun':[], 'action':[]}
    with torch.no_grad():
        for batch in tqdm(test_loader):
            if args.modal == 'A':
                predict = model(batch[0].to(device))
            elif args.modal == 'V':
                predict = model(batch[1].to(device))
            else:
                predict = model(batch[2].to(device))

            predict_verb = (torch.argmax(predict['verb'], dim=-1).cpu() == batch[-1]['verb'])
            predict_noun = (torch.argmax(predict['noun'], dim=-1).cpu() == batch[-1]['noun'])
            predict_action = torch.logical_and(predict_verb, predict_noun)
            acc['verb'].append( predict_verb.sum() / len(batch[-1]['verb']))
            acc['noun'].append( predict_noun.sum() / len(batch[-1]['noun']))
            acc['action'].append( predict_action.sum() / len(batch[-1]['verb']))   
    print('verb =', np.mean(acc['verb']), 'noun =', np.mean(acc['noun']))
    return np.mean(acc['action']) 
def train_step(model, input_data, optimizer, criteria, label):
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
                elif args.modal == 'V':
                    input_data = [batch[1].to(device)]
                else:
                    input_data = [batch[2].to(device)]
                train_step(model, input_data=input_data, optimizer=optimizer,
                            criteria=criteria, label=batch[-1])
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
    parser.add_argument('-d', '--dataset', default='EPICKitchen', type=str) # VGGSound, EPICKitchen
    parser.add_argument('-w', '--worker', default=4, type=int)
    parser.add_argument('-b', '--batch', default=64, type=int)
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
        pretrained_weight = torch.load('pretrained/deit_base_patch16_224.pth')['model']
        config = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, num_classes=309, pretrained=pretrained_weight)
        embed_dim = 768
        
        if args.modal == 'A':
            model = AudioTransformer(config, input_fdim=128, input_tdim=384).to(device)
        else:
            model = VisionTransformer(**config).to(device)

        if args.test: 
            if args.modal == 'A':
                model.load_state_dict(torch.load(checkpoint_loc + 'A_0.5118207.pth'))
            else:
                model.load_state_dict(torch.load(checkpoint_loc + 'V_0.49591964.pth'))
            print('loaded the pretrained model')
        full_dataset = getattr(dataset, args.dataset)()
        len_train = int(len(dataset) * 0.8)
        len_test = len(dataset) - len_train
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len_train, len_test], generator=torch.Generator().manual_seed(42))
        train(model, train_dataset, test_dataset, args.test, test_vggsound)
    else:
        pretrained_weight = torch.load('pretrained/deit_base_patch16_224.pth')['model']
        config = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, num_classes=(97, 300), pretrained=pretrained_weight)
        embed_dim = 768
        if args.modal == 'A':
            model = AudioTransformer(config, input_fdim=256, input_tdim=256).to(device)
        elif args.modal == 'V':
            model = VisionTransformer(**config).to(device)
        else:
            model = VisionTransformer(**config, in_chans=2).to(device)
        import h5py
        print('pre-load audio dict.....')
        audio_path = h5py.File('../split_EPIC_audio.hdf5', 'r')
        print('finish loading....')
        train_transform, val_transform = dataset.get_train_transform()
        train_dataset = getattr(dataset, args.dataset)(list_file=pd.read_pickle('EPIC_train.pkl'),               
                                                 transform=train_transform, mode='train', audio_path=audio_path)
        val_dataset = getattr(dataset, args.dataset)(list_file=pd.read_pickle('EPIC_val.pkl'),               
                                                 transform=val_transform, mode='val', audio_path=audio_path)
        if args.test:
            if args.modal == 'A':
                model.load_state_dict(torch.load(checkpoint_loc + 'A_0.16073199.pth'))
            else:
                model.load_state_dict(torch.load(checkpoint_loc + 'V_0.30203858.pth'))
            print('loaded the pretrained model')
        train(model, train_dataset, val_dataset, args.test, test_epickitchen)