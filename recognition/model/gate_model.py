'''
We implement multi-modal dynamic network here
We get three modes
1. Training main: randomly select exit (or bootstrapped?), without using gate network, exit = False
2. Training gate: without exit, only train the gate network, exit = False
3. Inference: use trained gate network to do inference, exit = True
'''
import torch.nn as nn
import torch
from torch.cuda.amp import autocast
from model.vit_model import AudioTransformerDiffPruning, VisionTransformerDiffPruning
def gumbel_softmax(logits, tau=1, hard=False, dim=1, training=True):
    """ See `torch.nn.functional.gumbel_softmax()` """
    # if training:
    # gumbels = -torch.empty_like(logits,
    #                             memory_format=torch.legacy_contiguous_format).exponential_().log()  # ~Gumbel(0,1)
    # gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    # # else:
    # #     gumbels = logits
    # y_soft = gumbels.softmax(dim)

    gumbels = -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)
    with torch.no_grad():
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        #  **test**
        # index = 0
        # y_hard = torch.Tensor([1, 0, 0, 0]).repeat(logits.shape[0], 1).cuda()
    ret = y_hard - y_soft.detach() + y_soft
    return y_soft, ret, index
class Gate_MM(nn.Module):
    '''
    output multi-modal decision
    '''
    def __init__(self, decision_space=12):
        super(Gate_MM, self).__init__()
        self.decision_space = decision_space
        self.bottle_neck = 768
        self.gate = nn.Linear(self.bottle_neck * 2, self.decision_space)

    def forward(self, output_cache):
        gate_input = torch.cat([output_cache['audio'][0], output_cache['image'][0]], dim=-1)
        logits = self.gate(gate_input)
        logits_audio = logits[:, :12]
        logits_image = logits[:, 12:]
        y_soft, ret_audio, index = gumbel_softmax(logits_audio)
        y_soft, ret_image, index = gumbel_softmax(logits_image)

        if len(output_cache['audio']) == 1:
            return ret_audio, ret_image
        else:
            audio = torch.cat(output_cache['audio'], dim=-1)
            audio = (audio.reshape(-1, self.decision_space, self.bottle_neck) * ret_audio.unsqueeze(2)).mean(dim=1)
            image = torch.cat(output_cache['image'], dim=-1)
            image = (image.reshape(-1, self.decision_space, self.bottle_neck) * ret_image.unsqueeze(2)).mean(dim=1)
            return torch.cat([audio, image], dim=-1), ret_audio, ret_image

class Gate_SM(nn.Module):
    '''
    output single-modal decision -> one modality is always full
    '''
    def __init__(self, decision_space=12):
        super(Gate_SM, self).__init__()
        self.decision_space = decision_space
        self.bottle_neck = 768
        self.gate = nn.Linear(self.bottle_neck * 2, self.decision_space)

    def forward(self, output_cache):
        gate_input = torch.cat([output_cache['audio'][0], output_cache['image'][0]], dim=-1)
        logits = self.gate(gate_input)
        y_soft, ret, index = gumbel_softmax(logits)
        hard_decision = torch.zeros((gate_input.shape[0], self.decision_space)).to(device=gate_input.device)
        hard_decision[:, -1] = 1
        ret_audio = hard_decision
        ret_image = ret
        if len(output_cache['audio']) == 1:
            # real inference
            return ret_audio, ret_image
        else:
            audio = torch.cat(output_cache['audio'], dim=-1)
            audio = (audio.reshape(-1, self.decision_space, self.bottle_neck) * ret_audio.unsqueeze(2)).mean(dim=1)
            image = torch.cat(output_cache['image'], dim=-1)
            image = (image.reshape(-1, self.decision_space, self.bottle_neck) * ret_image.unsqueeze(2)).mean(dim=1)
            return torch.cat([audio, image], dim=-1), ret_audio, ret_image
class AVnet_Gate(nn.Module):
    def __init__(self, gate_network=None, pretrained=True):
        '''
        :param exit: True - with exit, normally for testing, False - no exit, normally for training
        :param gate_network: extra gate network
        :param num_cls: number of class
        '''
        super(AVnet_Gate, self).__init__()
        self.gate = gate_network
        config = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                      pruning_loc=())
        embed_dim = 768
        self.audio = AudioTransformerDiffPruning(config, imagenet_pretrain=pretrained)
        self.image = VisionTransformerDiffPruning(**config)
        if pretrained:
            self.image.load_state_dict(torch.load('assets/deit_base_patch16_224.pth')['model'], strict=False)
        self.head = nn.Sequential(nn.Linear(embed_dim*2, 309))
    def get_parameters(self):
        parameter = [{'params': self.gate.parameters()},
                     {'params': self.projection.parameters()}]
        return parameter

    def gate_train(self, audio, image, label, teacher_model):
        '''
        We get three loss:
        computation loss1: absolute computation
        computation loss2: computation difference
        recognition loss: cross_entropy
        distillation loss1: KL divergence of logits
        distillation loss2: MSE of features
        '''
        with torch.no_grad():
            output_cache_distill, output_distill = teacher_model(audio, image, 'no_exit')
            feature_distill = torch.cat([output_cache_distill['audio'][-1], output_cache_distill['image'][-1]], dim=-1)

        output_cache, output = self.forward(audio, image, 'no_exit')
        feature, gate_a, gate_i = self.gate(output_cache)
        output = self.head(feature)

        computation_penalty = torch.range(1, 12).to('cuda')/12
        loss_c = (((gate_a * computation_penalty + gate_i * computation_penalty).sum(1)) ** 2).mean()
        loss_c += (((gate_a * computation_penalty).sum(1) - (gate_i * computation_penalty).sum(1)) ** 2).mean()

        loss_r = nn.functional.cross_entropy(output, label)# recognition-level loss

        loss_kd = nn.functional.kl_div(
                nn.functional.log_softmax(output, dim=-1), nn.functional.log_softmax(output_distill, dim=-1),
                reduction='batchmean', log_target=True)
        loss_kd += torch.pow(feature - feature_distill, 2).mean()

        compress = [(torch.argmax(gate_a, dim=-1).float().mean() + 1).item()/12 ,
              (torch.argmax(gate_i, dim=-1).float().mean() + 1).item()/12,
            (torch.argmax(gate_a, dim=-1) - torch.argmax(gate_i, dim=-1)).float().abs().mean().item()/12]
        acc = (torch.argmax(output, dim=-1) == label).sum().item() / len(label)

        loss = loss_c * 1 + loss_r * 1 + loss_kd * 0.5
        loss.backward()
        return [loss_c.item(), loss_r.item(), loss_kd.item(), compress, acc]

    @autocast()
    def forward(self, audio, image, mode='dynamic'):
        output_cache = {'audio': [], 'image': []}

        B, audio = self.audio.preprocess(audio.unsqueeze(1))
        B, image = self.image.preprocess(image)

        # first block
        audio = self.audio.blocks[0](audio)
        audio_norm = self.audio.norm(audio)[:, 0]
        output_cache['audio'].append(audio_norm)

        image = self.image.blocks[0](image)
        image_norm = self.image.norm(image)[:, 0]
        output_cache['image'].append(image_norm)

        if mode == 'dynamic':
            self.exit = torch.randint(12, (2, 1))
        elif mode == 'no_exit':
            # by default, no exit
            self.exit = torch.tensor([11, 11])
        elif mode == 'gate':
            # not implemented yet
            gate_a, gate_i = self.gate(output_cache)
            self.exit = torch.argmax(torch.cat([gate_a, gate_i], dim=0), dim=-1)
        else: # directly get the exit
            self.exit = mode

        for i, (blk_a, blk_i) in enumerate(zip(self.audio.blocks[1:], self.image.blocks[1:])):
            if i < self.exit[0].item():
                audio = blk_a(audio)
                audio_norm = self.audio.norm(audio)[:, 0]
                output_cache['audio'].append(audio_norm)
            if i < self.exit[1].item():
                image = blk_i(image)
                image_norm = self.image.norm(image)[:, 0]
                output_cache['image'].append(image_norm)

        audio = output_cache['audio'][-1]
        image = output_cache['image'][-1]
        output = torch.cat([audio, image], dim=1)
        output = torch.flatten(output, start_dim=1)
        output = self.head(output)
        return output_cache, output