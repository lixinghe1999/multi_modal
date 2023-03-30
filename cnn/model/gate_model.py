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
from model.resnet_model import resnet50
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
class Gate(nn.Module):
    def __init__(self):
        super(Gate, self).__init__()
        self.feature = 256
        self.reduced_feature = 64
        self.level = 4
        self.gate1 = nn.Sequential(*[nn.AdaptiveAvgPool2d(1), nn.Conv2d(self.feature, self.reduced_feature, kernel_size=1, bias=True),
                                     nn.Sigmoid(), nn.Conv2d(self.reduced_feature, self.feature, kernel_size=1, bias=True) ])
        self.gate2 = nn.Sequential(*[nn.AdaptiveAvgPool2d(1), nn.Conv2d(self.feature, self.reduced_feature, kernel_size=1, bias=True),
                                     nn.Sigmoid(), nn.Conv2d(self.reduced_feature, self.feature, kernel_size=1, bias=True) ])
        self.gate = nn.Linear(self.feature * 2, self.level * 2)
    def forward(self, audio, image):
        '''
        :param audio, image: raw data
        :param output_cache: dict: ['audio', 'image'] list -> 12 (for example) * [batch, bottle_neck]
         Or [batch, raw_data_shape] -> [batch, 3, 224, 224]
        :return: Gumbel_softmax decision
        '''
        audio_pool = self.gate1(audio)
        image_pool = self.gate2(image)
        gate_input = torch.cat([audio_pool, image_pool], dim=-1)
        logits = self.gate(gate_input)
        logits_audio = logits[:, :self.level]
        logits_image = logits[:, self.level:]

        y_soft, ret_audio, index = gumbel_softmax(logits_audio)
        y_soft, ret_image, index = gumbel_softmax(logits_image)
        return ret_audio, ret_image


class AVnet_Gate(nn.Module):
    def __init__(self, gate_network=None, pretrained=True):
        '''
        :param exit: True - with exit, normally for testing, False - no exit, normally for training
        :param gate_network: extra gate network
        :param num_cls: number of class
        '''
        super(AVnet_Gate, self).__init__()
        self.gate = gate_network
        self.audio = resnet50(pretrained=pretrained)
        self.image = resnet50(pretrained=pretrained)
        embed_dim = 512 * 4
        self.head = nn.ModuleList([nn.Linear(embed_dim * 2, 309) for _ in range(4)])
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
        gate_a, gate_i = self.gate(audio, image, output_cache)

        audio = torch.cat(output_cache['audio'], dim=-1)
        audio = (audio.reshape(-1, self.level, self.bottle_neck) * gate_a.unsqueeze(2)).mean(dim=1)
        image = torch.cat(output_cache['image'], dim=-1)
        image = (image.reshape(-1, self.level, self.bottle_neck) * gate_i.unsqueeze(2)).mean(dim=1)
        feature = torch.cat([audio, image], dim=1)
        output = self.head(feature)

        computation_penalty = torch.range(1, 12).to('cuda')/12
        loss_c = (((gate_a * computation_penalty + gate_i * computation_penalty).sum(1)) ** 2).mean()
        loss_c += (((gate_a * computation_penalty).sum(1) - (gate_i * computation_penalty).sum(1)) ** 2).mean()

        loss_r = nn.functional.cross_entropy(output, label) # recognition-level loss

        loss_kd = nn.functional.kl_div(
                nn.functional.log_softmax(output, dim=-1), nn.functional.log_softmax(output_distill, dim=-1),
                reduction='batchmean', log_target=True)
        loss_kd += torch.pow(feature - feature_distill, 2).mean()

        compress = [(torch.argmax(gate_a, dim=-1).float().mean() + 1).item()/12 ,
              (torch.argmax(gate_i, dim=-1).float().mean() + 1).item()/12,
            (torch.argmax(gate_a, dim=-1) - torch.argmax(gate_i, dim=-1)).float().abs().mean().item()/12]
        acc = (torch.argmax(output, dim=-1) == label).sum().item() / len(label)

        loss = loss_c * 0.5 + loss_r * 1 + loss_kd * 0.5
        loss.backward()
        return [loss_c.item(), loss_r.item(), loss_kd.item(), compress, acc]

    @autocast()
    def forward(self, audio, image, mode='dynamic'):
        output_cache = []

        audio = self.audio.preprocess(audio)
        image = self.image.preprocess(image)

        if mode == 'dynamic':
            self.exit = torch.randint(12, (2, 1))
        elif mode == 'no_exit':
            # by default, no exit
            self.exit = torch.tensor([11, 11])
        elif mode == 'gate':
            # not implemented yet
            gate_a, gate_i = self.gate(audio, image)
            self.exit = torch.argmax(torch.cat([gate_a, gate_i], dim=0), dim=-1)
        else:  # directly get the exit
            self.exit = mode

        for i, (blk_a, blk_i) in enumerate(zip(self.audio.blocks, self.image.blocks)):
            audio = blk_a(audio)
            image = blk_i(image)

            audio_pool = torch.flatten(self.audio.avgpool(audio), 1)
            image_pool = torch.flatten(self.image.avgpool(image), 1)
            x = self.head[i](torch.cat([audio_pool, image_pool], dim=1))
            output_cache.append(x)

        return output_cache, x


