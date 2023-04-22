# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from src.args import ArgumentParserRGBDSegmentation
from src.build_model import build_model
from src.confusion_matrix import ConfusionMatrixPytorch, miou_pytorch
from src.prepare_data import prepare_data


if __name__ == '__main__':
    # arguments
    parser = ArgumentParserRGBDSegmentation(
        description='Efficient RGBD Indoor Sematic Segmentation (Evaluation)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.set_common_args()
    parser.add_argument('--ckpt_path', type=str,
                        required=False,
                        help='Path to the checkpoint of the trained model.')
    args = parser.parse_args()

    # dataset
    args.pretrained_on_imagenet = False  # we are loading other weights anyway
    _, data_loader, *add_data_loader = prepare_data(args, with_input_orig=True)
    if args.valid_full_res:
        # cityscapes only -> use dataloader that returns full resolution images
        data_loader = add_data_loader[0]

    n_classes = data_loader.dataset.n_classes_without_void
    # model and checkpoint loading
    model, device = build_model(args, n_classes=n_classes)
    checkpoint = torch.load(args.ckpt_path,
                            map_location=lambda storage, loc: storage)
    # model.load_state_dict(checkpoint['state_dict'])
    print('Loaded checkpoint from {}'.format(args.ckpt_path))

    model.eval()
    model.to(device)

    n_samples = 0

    confusion_matrices = dict()

    cameras = data_loader.dataset.cameras
    # modality_weight = []
    for camera in cameras:
        confusion_matrices[camera] = dict()
        confusion_matrices[camera] = ConfusionMatrixPytorch(n_classes)
        torch_miou = miou_pytorch(confusion_matrices[camera])

        n_samples_total = len(data_loader.dataset)
        with data_loader.dataset.filter_camera(camera):
            for i, sample in enumerate(data_loader):
                n_samples += sample['image'].shape[0]
                print(f'\r{n_samples}/{n_samples_total}', end='')

                image = sample['image'].to(device)
                depth = sample['depth'].to(device)
                label_orig = sample['label_orig']

                _, image_h, image_w = label_orig.shape

                with torch.no_grad():
                    if args.modality == 'rgbd':
                        inputs = (image, depth)
                    elif args.modality == 'rgb':
                        inputs = (image,)
                    elif args.modality == 'depth':
                        inputs = (depth,)

                    pred = model(*inputs)

                    pred = F.interpolate(pred, (image_h, image_w),
                                         mode='bilinear',
                                         align_corners=False)
                    pred = torch.argmax(pred, dim=1)

                    # ignore void pixels
                    mask = label_orig > 0
                    label = torch.masked_select(label_orig, mask)
                    pred = torch.masked_select(pred, mask.to(device))

                    # In the label 0 is void but in the prediction 0 is wall.
                    # In order for the label and prediction indices to match
                    # we need to subtract 1 of the label.
                    label -= 1

                    # copy the prediction to cpu as tensorflow's confusion
                    # matrix is faster on cpu
                    pred = pred.cpu()

                    label = label.numpy()
                    pred = pred.numpy()

                    # confusion_matrices[camera].update_conf_matrix(label, pred)
                    confusion_matrices[camera].update(torch.from_numpy(label), torch.from_numpy(pred))

                # modality_weight.append(model.modality_weight)
                print(f'\r{i + 1}/{len(data_loader)}', end='')

        miou = torch_miou.compute().data.numpy()
        print(f'\rCamera: {camera} mIoU: {100*miou:0.2f}')
    # import pickle
    # file = open(r"modality_weight.pkl", "wb")
    # pickle.dump(modality_weight, file)  # 保存list到文件
    # file.close()

    confusion_matrices['all'] = ConfusionMatrixPytorch(n_classes)
    torch_miou = miou_pytorch(confusion_matrices['all'])
    # sum confusion matrices of all cameras
    for camera in cameras:
        # confusion_matrices['all'].overall_confusion_matrix += \
        #     confusion_matrices[camera].overall_confusion_matrix
        confusion_matrices['all'].confusion_matrix += \
            confusion_matrices[camera].confusion_matrix.numpy()
        confusion_matrices['all']._num_examples += confusion_matrices[camera]._num_examples
    # miou, _ = confusion_matrices['all'].compute_miou()
    miou = torch_miou.compute().data.numpy()
    print(f'All Cameras, mIoU: {100*miou:0.2f}')
