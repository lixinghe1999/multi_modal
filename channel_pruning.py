
from utils.datasets.vggsound import VGGSound
import numpy as np
import torch
from model.slim_model import AVnet_Slim
from model.vanilla_model import AVnet
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
    outputs = []
    optimizer.zero_grad()
    for mode in range(4):
        model.audio.set_mode(mode)
        model.image.set_mode(mode)
        outputs.append(model(audio, image))
    for mode in range(4):
        loss = 0
        for j in range(mode + 1, 4):
            loss += soft_criteria(outputs[mode], outputs[j])
        loss += criteria(outputs[mode], label)
        loss.backward(retain_graph=True)
    optimizer.step()
def test_step(model, input_data, label):
    audio, image = input_data
    acc = []
    output = model(audio, image)
    acc.append((torch.argmax(output, dim=-1).cpu() == label).sum() / len(label))
    # for mode in range(4):
    #     model.audio.set_mode(mode)
    #     model.image.set_mode(mode)
    #     output = model(audio, image)
    #     acc.append((torch.argmax(output, dim=-1).cpu() == label).sum() / len(label))
    return acc
def train(model, train_dataset, test_dataset):
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, num_workers=workers, batch_size=batch_size, shuffle=True,
                                               drop_last=True, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, num_workers=workers, batch_size=batch_size, shuffle=False)
    best_acc = 0
    if args.task == 'AV':
        optimizer = torch.optim.Adam(model.parameters(), lr=.0001, weight_decay=1e-4)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=.0001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)
    criteria = torch.nn.CrossEntropyLoss()
    soft_criteria = SoftTargetCrossEntropy()
    for epoch in range(20):
        model.train()
        # for idx, batch in enumerate(tqdm(train_loader)):
        #     audio, image, text, _ = batch
        #     train_step(model, input_data=(audio.to(device), image.to(device)), optimizer=optimizer,
        #                 criteria=criteria, soft_criteria=soft_criteria,label=text.to(device))
        # scheduler.step()
        model.eval()
        acc = []
        with torch.no_grad():
            for batch in tqdm(test_loader):
                audio, image, text, _ = batch
                acc.append(test_step(model, input_data=(audio.to(device), image.to(device)), label=text))
        mean_acc = np.mean(acc, axis=0)
        print('epoch', epoch, mean_acc)
        avg_acc = np.mean(mean_acc)
        if avg_acc > best_acc:
            best_acc = avg_acc
            torch.save(model.state_dict(), 'vanilla_' + args.model + '_' + args.task + '_' +
                       str(epoch) + '_' + str(avg_acc) + '.pth')

def load_batchnorm(model, state_dict):
    from collections import OrderedDict
    missing_keys = []
    unexpected_keys = []
    error_msgs = []

    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = OrderedDict(state_dict)
    if metadata is not None:
        # mypy isn't aware that "_metadata" exists in state_dict
        state_dict._metadata = metadata  # type: ignore[attr-defined]

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')
        print(missing_keys)
        print(unexpected_keys)
        # Note that the hook can modify missing_keys and unexpected_keys.
        # incompatible_keys = _IncompatibleKeys(missing_keys, unexpected_keys)
        # for hook in module._load_state_dict_post_hooks.values():
        #     out = hook(module, incompatible_keys)
        #     assert out is None, (
        #         "Hooks registered with ``register_load_state_dict_post_hook`` are not"
        #         "expected to return new values, if incompatible_keys need to be modified,"
        #         "it should be done inplace."
        #     )

    load(model)
    del load

    # if strict:
    #     if len(unexpected_keys) > 0:
    #         error_msgs.insert(
    #             0, 'Unexpected key(s) in state_dict: {}. '.format(
    #                 ', '.join('"{}"'.format(k) for k in unexpected_keys)))
    #     if len(missing_keys) > 0:
    #         error_msgs.insert(
    #             0, 'Missing key(s) in state_dict: {}. '.format(
    #                 ', '.join('"{}"'.format(k) for k in missing_keys)))
    #
    # if len(error_msgs) > 0:
    #     raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
    #         self.__class__.__name__, "\n\t".join(error_msgs)))
    # return _IncompatibleKeys(missing_keys, unexpected_keys)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='resnet')
    parser.add_argument('-t', '--task', default='distill')
    parser.add_argument('-w', '--worker', default=4, type=int)
    parser.add_argument('-b', '--batch', default=32, type=int)
    args = parser.parse_args()
    workers = args.worker
    batch_size = args.batch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(0)
    model = AVnet_Slim().to('cuda')
    weight = torch.load('vanilla_resnet_AV_19_0.65678066.pth')
    load_batchnorm(model, weight)
    # model = AVnet(model='resnet', pretrained=False).to(device)
    # model.load_state_dict(weight, strict=False)
    # dataset = VGGSound()
    # len_train = int(len(dataset) * 0.8)
    # len_test = len(dataset) - len_train
    # train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len_train, len_test], generator=torch.Generator().manual_seed(42))
    # train(model, train_dataset, test_dataset)

