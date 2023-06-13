'''
Two-stage training
1. inference, get all the features and corresponding token locations after dynamic token (batch = 1)
2. train the model with the features and token locations
'''
import dataset
import numpy as np
import torch
import models
from models.dynamicvit_legacy import DynToken
import torchmetrics
import warnings
from tqdm import tqdm
import argparse
import pandas as pd
import utils
warnings.filterwarnings("ignore")
def test_epickitchen(model, test_loader):
    model.eval()
    acc = {'verb':[], 'noun':[], 'action':[]}
    ratio = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_data = [batch[0].to(device), batch[1].to(device)]
            predict, _, _= model(*input_data)

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

def test_compression(model, test_loader):
    psnr = torchmetrics.PeakSignalNoiseRatio()
    ssim = torchmetrics.StructuralSimilarityIndexMeasure()
    model.eval()
    decoder.eval()
    metrics = {'PSNR': [], 'SSIM': []}
    with torch.no_grad():
        for batch in tqdm(test_loader):
            outputs = model(batch[0].to(device), batch[1].to(device))
            pred = decoder.forward_decoder(outputs[1], outputs[2])
            reconstruct = decoder.unpatchify(pred).cpu()

            metrics['PSNR'].append(psnr(reconstruct, batch[1]).item())
            metrics['SSIM'].append(ssim(reconstruct, batch[1]).item())
            print(metrics)
    print('PSNR =', np.mean(metrics['PSNR']), 'SSIM =', np.mean(metrics['SSIM']))
    return np.mean(metrics['PSNR'])


def train_compression(model, train_dataset, test_dataset, loss_compress, loss_prun, test_epoch=test_compression):
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, num_workers=workers, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, num_workers=workers, batch_size=1, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=.0001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)
    best_acc = 0
    for epoch in range(10):
        model.train()
        for idx, batch in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            input_data = [batch[0].to(device), batch[1].to(device)]
            outputs = model(*input_data)

            loss_p, _ = loss_prun(input_data, outputs, batch[-1])
            loss_c = loss_compress(batch[1].to(device), outputs)
            loss = loss_c + loss_p
            if idx % 100 == 0:
                print(loss.item())
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        acc = []
        with torch.no_grad():
            acc = test_epoch(model, test_loader)
        print('epoch', epoch, 'acc =', acc)
        if acc > best_acc:
            best_acc = acc
            best_model = model.state_dict()
    torch.save(best_model, checkpoint_loc + 'Compression_' + str(args.model) + '_' + str(args.scale) + '_' + str(best_acc) + '.pth')

def save_compression(model, train_dataset, test_dataset, test_epoch=test_epickitchen):
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, num_workers=workers, batch_size=1, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, num_workers=workers, batch_size=1, shuffle=False)
    with torch.no_grad():
        print('first test the performance')
        acc = test_epoch(model, test_loader)
        mean_acc = np.mean(acc)
        print('preserved ratio:', model.token_ratio, 'accuracy:', mean_acc)

        compress_token, compress_feature = compression(train_loader)
        print(compress_token.shape, compress_feature.shape)
        np.savez_compressed('train_compressed_feature.npz', token =compress_token, feature=compress_feature)

        compress_token, compress_feature = compression(test_loader)
        print(compress_token.shape, compress_feature.shape)
        np.savez_compressed('test_compressed_feature.npz', token =compress_token, feature=compress_feature)
def compression(loader):
    compress_feature = []
    compress_token = []
    print('start saving the compressed feature')
    model.eval()
    for batch in tqdm(loader):
        input_data = [batch[0].to(device), batch[1].to(device)]
        _, features, remain_token= model(*input_data)
        # features = features[:, :25, :]
        # remain_token = remain_token[:, :25]
        compress_feature.append(features.cpu().numpy())
        compress_token.append(remain_token.cpu().numpy())
    compress_feature = np.concatenate(compress_feature, axis=0)
    compress_token = np.concatenate(compress_token, axis=0)
    print('finish saving the compressed feature')
    return compress_token, compress_feature
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='MBT', type=str)
    parser.add_argument('-d', '--dataset', default='EPICKitchen', type=str) # VGGSound, EPICKitchen
    parser.add_argument('-w', '--worker', default=4, type=int)
    parser.add_argument('-b', '--batch', default=32, type=int)
    parser.add_argument('-s', '--scale', default='base', type=str)
    parser.add_argument('-c', '--cuda', default=0, type=int)
    parser.add_argument('-compress', action='store_true', default=False)
    args = parser.parse_args()
    workers = args.worker
    batch_size = args.batch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(args.cuda)

    pruning_loc = (3, 6, 9)
    token_ratio = 0.6
    if args.dataset == 'EPICKitchen':
        checkpoint_loc = 'checkpoints_epic_kitchen/'
    else:
        checkpoint_loc = 'checkpoints_vggsound/'
    if args.compress:
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
        model.load_state_dict(torch.load(checkpoint_loc + 'DynToken_MBT_base_0.37599015.pth'))
        model.apply_merge(r=0)
        
        save_compression(model, train_dataset, val_dataset, test_epoch=test_epickitchen)
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
        # after compression, only train the reconstruction
        model = DynToken(pruning_loc=pruning_loc, token_ratio=token_ratio, distill=True, backbone=getattr(models, args.model), scale=args.scale, pretrained=True, num_class=(97, 300)).to(device)
        # model.apply_merge(r=0)

        decoder = models.MaskedAutoencoderViT().to(device)
        loss_compress = utils.Dynamic_compression(decoder, device)

        teacher_model = DynToken(distill=True, backbone=getattr(models, args.model), scale=args.scale, pretrained=False,    num_class=(97, 300)).to(device)
        teacher_model.load_state_dict(torch.load(checkpoint_loc + 'MBT_base_0.3744411.pth'))
        teacher_model.eval()
        # teacher_model.apply_merge(r=0)
        loss_prun = utils.DistillDiffPruningLoss_dynamic(teacher_model, torch.nn.CrossEntropyLoss(), clf_weight=1.0, keep_ratio=token_ratio, mse_token=True, ratio_weight=2, distill_weight=0.5)

        train_transform, val_transform = dataset.get_train_transform()
        train_dataset = getattr(dataset, args.dataset)(modality=['Spec', 'RGB'], list_file=pd.read_pickle('EPIC_train.pkl'), transform=train_transform, mode='train',audio_path=audio_path)
        val_dataset = getattr(dataset, args.dataset)(modality=['Spec', 'RGB'], list_file=pd.read_pickle('EPIC_val.pkl'), transform=val_transform, mode='val',audio_path=audio_path)
        train_compression(model, train_dataset, val_dataset, loss_compress, loss_prun, test_compression)

