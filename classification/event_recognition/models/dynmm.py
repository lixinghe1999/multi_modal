'''
Baseline
We implement multi-modal dynamic network here
We get three variants
1. Training main: randomly select exit (or bootstrapped?), without using gate network, exit = False
2. Training gate: without exit, only train the gate network, exit = False
3. Inference: use trained gate network to do inference, exit = True
'''
import torch.nn as nn
import torch
from torch.cuda.amp import autocast
class Gate_MM(nn.Module):
    '''
    output multi_modal decision choose one of the modality
    '''
    def __init__(self, decision_space=12, embed_dim=768):
        super(Gate_SM, self).__init__()
        self.decision_space = decision_space
        self.embed_dim = embed_dim
        self.gate = nn.Linear(self.embed_dim * 2, 2)

    def forward(self, output_cache):
        gate_input = torch.cat([output_cache['audio'][0], output_cache['image'][0]], dim=-1)
        logits = self.gate(gate_input)
        decision_modal = torch.nn.functional.gumbel_softmax(logits, hard=True)
        decision = torch.zeros((gate_input.shape[0], self.decision_space, 2)).to(device=gate_input.device)
        decision[:, -1] = decision_modal
        if len(output_cache['audio']) == 1:
            # real inference
            return decision[:, :, 0], decision[:, :, 1]
        else:
            audio = output_cache['audio'][-1] * decision_modal[:, 0]
            image = output_cache['image'][-1] * decision_modal[:, 1]
            return torch.cat([audio, image], dim=-1), decision[:, :, 0], decision[:, :, 1]


class Gate_SM(nn.Module):
    '''
    output single-modal decision -> one modality is always full
    '''
    def __init__(self, decision_space=12, embed_dim=768):
        super(Gate_SM, self).__init__()
        self.decision_space = decision_space
        self.embed_dim = embed_dim
        self.gate = nn.Linear(self.embed_dim * 2, self.decision_space)

    def forward(self, output_cache):
        gate_input = torch.cat([output_cache['audio'][0], output_cache['image'][0]], dim=-1)
        logits = self.gate(gate_input)
        decision = torch.nn.functional.gumbel_softmax(logits, hard=True)

        fixed_decision = torch.zeros((gate_input.shape[0], self.decision_space)).to(device=gate_input.device)
        fixed_decision[:, -1] = 1
        if len(output_cache['audio']) == 1:
            # real inference
            return decision, fixed_decision
        else:
            audio = torch.cat(output_cache['audio'], dim=-1)
            audio = (audio.reshape(-1, self.decision_space, self.embed_dim) * decision.unsqueeze(2)).mean(dim=1)
            image = output_cache['image'][-1]
            return torch.cat([audio, image], dim=-1), decision, fixed_decision

class DynMM(nn.Module):
    def __init__(self, backbone, gate='sm', scale='base', pretrained=True):
        super(DynMM, self).__init__()
        
        backbone = backbone(scale, pretrained)
        module_list = ['audio', 'image', 'embed_dim']
        for m in module_list:
            setattr(self, m, getattr(backbone, m))
        self.len_blocks = len(self.audio.blocks)
        if gate == 'sm':
            self.gate = Gate_SM(decision_space=self.len_blocks, embed_dim=self.embed_dim)
        else:
            self.gate = Gate_MM(decision_space=self.len_blocks, embed_dim=self.embed_dim)
        self.head = nn.Linear(self.embed_dim*2, 309)
    def get_parameters(self):
        parameter = [{'params': self.gate.parameters()},
                     {'params': self.head.parameters()}]
        return parameter

    def gate_train(self, audio, image, label):
        '''
        We get three loss:
        computation loss1: absolute computation
        computation loss2: computation difference
        recognition loss: cross_entropy
        '''
        output_cache, output = self.forward(audio, image, 'no_exit')
        feature, gate_a, gate_i = self.gate(output_cache)
        output = self.head(feature)

        computation_penalty = torch.range(1, self.len_blocks).to('cuda')/self.len_blocks
        loss_c = (((gate_a * computation_penalty + gate_i * computation_penalty).sum(1)) ** 2).mean()
        loss_c += (((gate_a * computation_penalty).sum(1) - (gate_i * computation_penalty).sum(1)) ** 2).mean()

        loss_r = nn.functional.cross_entropy(output, label)# recognition-level loss

        compress = [(torch.argmax(gate_a, dim=-1).float().mean() + 1).item()/self.len_blocks ,
              (torch.argmax(gate_i, dim=-1).float().mean() + 1).item()/self.len_blocks,
            (torch.argmax(gate_a, dim=-1) - torch.argmax(gate_i, dim=-1)).float().abs().mean().item()/self.len_blocks]
        acc = (torch.argmax(output, dim=-1) == label).sum().item() / len(label)

        loss = loss_c * 0.1 + loss_r * 1
        loss.backward()
        return [loss_c.item(), loss_r.item(), compress, acc]

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
            self.exit = torch.randint(self.len_blocks, (2, 1))
        elif mode == 'no_exit':
            # by default, no exit
            self.exit = torch.tensor([self.len_blocks-1, self.len_blocks-1])
        elif mode == 'gate':
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