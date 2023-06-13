import dataset
import numpy as np
import torch
from models import AudioTransformer, VisionTransformer, IMUTransformer  
import pandas as pd
import argparse
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")
# remove annoying librosa warning

def test_epickitchen(model, test_loader):
    model.eval()
    acc = {'verb':[], 'noun':[], 'action':[]}
    with torch.no_grad():
        for batch in tqdm(test_loader):
            predict = model(batch[0].to(device))
            predict_verb = (torch.argmax(predict['verb'], dim=-1).cpu() == batch[-1]['verb'])
            predict_noun = (torch.argmax(predict['noun'], dim=-1).cpu() == batch[-1]['noun'])
            predict_action = torch.logical_and(predict_verb, predict_noun)
            acc['verb'].append( predict_verb.sum() / len(batch[-1]['verb']))
            acc['noun'].append( predict_noun.sum() / len(batch[-1]['noun']))
            acc['action'].append( predict_action.sum() / len(batch[-1]['verb']))   
    print('verb =', np.mean(acc['verb']), 'noun =', np.mean(acc['noun']))
    return np.mean(acc['action']) 
def train_step(model, input_data, optimizer, criteria, label):
    output = model(input_data)
    # Backward
    optimizer.zero_grad()
    if isinstance(label, dict):
        loss = 0
        loss += torch.nn.functional.cross_entropy(output['verb'], label['verb'].to(device))
        loss += torch.nn.functional.cross_entropy(output['noun'], label['noun'].to(device))
    else:
        loss = criteria(output, label.to(device))
    loss.backward()
    optimizer.step()
    return loss.item()
def train(model, train_dataset, test_dataset, test=False, test_epoch=test_epickitchen):
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
                input_data = batch[0].to(device)
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
    parser.add_argument('-m', '--modal', default='Spec', type=str) # RGB, Flow, Spec, IMU
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

    pretrained_weight = torch.load('pretrained/deit_base_patch16_224.pth')['model']
    config = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, num_classes=(97, 300), pretrained=pretrained_weight)
    embed_dim = 768
    if args.modal == 'Spec':
        model = AudioTransformer(config, input_fdim=256, input_tdim=256).to(device)
    elif args.modal == 'RGB':
        model = VisionTransformer(**config).to(device)
    elif args.modal == 'Flow':
        model = VisionTransformer(**config, in_chans=2).to(device)
    else:
        model = IMUTransformer(num_classes=(97, 300)).to(device)

    import h5py
    print('pre-load audio dict.....')
    audio_path = h5py.File('../split_EPIC_audio.hdf5', 'r')
    print('finish loading....')
    train_transform, val_transform = dataset.get_train_transform()
    train_dataset = getattr(dataset, args.dataset)(modality=[args.modal], list_file=pd.read_pickle('EPIC_train_100.pkl'),               
                                                transform=train_transform, mode='train', audio_path=audio_path)
    val_dataset = getattr(dataset, args.dataset)(modality=[args.modal], list_file=pd.read_pickle('EPIC_val_100.pkl'),               
                                                transform=val_transform, mode='val', audio_path=audio_path)
    if args.test:
        if args.modal == 'Spec':
            model.load_state_dict(torch.load(checkpoint_loc + 'A_0.16073199.pth'))
        elif args.modal == 'RGB':
            model.load_state_dict(torch.load(checkpoint_loc + 'V_0.30203858.pth'))
        print('loaded the pretrained model')
    train(model, train_dataset, val_dataset, args.test, test_epickitchen)