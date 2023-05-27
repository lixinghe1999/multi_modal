from models.vit_model import PredictorLG, batch_index_select
from torch.cuda.amp import autocast
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class DynToken(nn.Module):
    def __init__(self, distill=False, pruning_loc=(), token_ratio=(), backbone='base', scale='base', pretrained=False, num_class=309):
        super(DynToken, self).__init__()

        backbone = backbone(scale, pretrained, num_class)
        module_list = ['audio', 'image', 'embed_dim', 'head', 'multi_head', 'head_verb', 'head_noun']
        for m in module_list:
            try:
                setattr(self, m, getattr(backbone, m))
            except:
                print('Careful, do not have', m)
                pass
        self.len_blocks = len(self.audio.blocks)
        
        if isinstance(num_class, int):
            self.num_patches = self.audio.num_patches + 14 * 14
        else:
            self.num_patches = (self.audio.num_patches + 14 * 14)

        if len(pruning_loc) > 0:
            predictor_list = [PredictorLG(self.embed_dim) for _ in range(len(pruning_loc))]
            self.score_predictor = nn.ModuleList(predictor_list)

        self.distill = distill
        self.pruning_loc = pruning_loc
        self.token_ratio = token_ratio


    def output(self, audio, image, B):
        audio = self.audio.norm(audio)
        image = self.image.norm(image)
        if self.multi_head:
            features = torch.cat([audio[:, 1:], image[:, 1:]], dim=1)
            x = torch.cat([audio[:, 0], image[:, 0]], dim=1)
            verb = self.head_verb(x)
            noun = self.head_noun(x)
            return {'verb': verb, 'noun': noun}, features
        else:
            features = torch.cat([audio[:, 1:], image[:, 1:]], dim=1)
            x = torch.cat([audio[:, 0], image[:, 0]], dim=1)
            x = torch.flatten(x, start_dim=1)
            x = self.head(x)
        return x, features

    @autocast()
    def forward(self, audio, image):
        B = audio.shape[0]
        if self.multi_head:
            audio = audio.view(-1, 1, 256, 256)
            image = image.view(-1, 3, 224, 224)
        _, audio = self.audio.preprocess(audio)
        _, image = self.image.preprocess(image) 
        audio = audio.view(B, -1, self.audio.embed_dim)
        image = image.view(B, -1, self.image.embed_dim)
        p_count = 0
        out_pred_prob = []
        early_output = []
        prev_decision = torch.ones(B, self.num_patches, 1, dtype=audio.dtype, device=audio.device)
        policy = torch.ones(B, self.num_patches + 2, 1, dtype=audio.dtype, device=audio.device)
        for i, (blk_a, blk_i) in enumerate(zip(self.audio.blocks, self.image.blocks)):
            if i in self.pruning_loc:
                spatial_x = torch.cat([audio[:, 1:], image[:, 1:]], dim=1)
                token_len_audio = audio.shape[1] - 1
                pred_score = self.score_predictor[p_count](spatial_x, prev_decision).reshape(B, -1, 2)
                if self.training:
                    hard_keep_decision = F.gumbel_softmax(pred_score, hard=True)[:, :, 0:1] * prev_decision
                    out_pred_prob.append(hard_keep_decision.reshape(B, self.num_patches))
                    decision_audio = hard_keep_decision[:, :token_len_audio]
                    decision_image = hard_keep_decision[:, token_len_audio:]

                    cls_policy = torch.ones(B, 1, 1, dtype=hard_keep_decision.dtype, device=hard_keep_decision.device)
                    policy_a = torch.cat([cls_policy, decision_audio], dim=1)
                    audio = blk_a(audio, policy=policy_a)

                    cls_policy = torch.ones(B, 1, 1, dtype=hard_keep_decision.dtype, device=hard_keep_decision.device)
                    policy_i = torch.cat([cls_policy, decision_image], dim=1)
                    image = blk_i(image, policy=policy_i)
                    prev_decision = hard_keep_decision
                    policy = torch.cat([policy_a, policy_i], dim=1)
                    early_output.append(self.output(audio, image, B)[0])
                else:
                    score = pred_score[:, :, 0]
                    _, indices = torch.sort(score, dim=1, descending=False)
                    # TopK selection
                    num_keep_node = int(self.num_patches * self.token_ratio[p_count])
                    keep_policy = indices[:, -num_keep_node:]

                    prev_decision = batch_index_select(prev_decision, keep_policy)

                    keep_audio = keep_policy < token_len_audio
                    keep_image = keep_policy >= token_len_audio

                    keep_audio = torch.masked_select(keep_policy, mask=keep_audio).unsqueeze(0)
                    keep_image = torch.masked_select(keep_policy, mask=keep_image).unsqueeze(0) - token_len_audio

                    cls_policy = torch.zeros(B, 1, dtype=keep_policy.dtype, device=keep_policy.device)
                    now_policy = torch.cat([cls_policy, keep_audio + 1], dim=1)
                    audio = batch_index_select(audio, now_policy)
                    audio = blk_a(audio)

                    cls_policy = torch.zeros(B, 1, dtype=keep_policy.dtype, device=keep_policy.device)
                    now_policy = torch.cat([cls_policy, keep_image + 1], dim=1)
                    image = batch_index_select(image, now_policy)
                    image = blk_i(image)

                p_count += 1
            else:
                if self.training:
                    policy_a = policy[:, :self.audio.num_patches + 1]
                    policy_i = policy[:, self.audio.num_patches + 1:]
                    audio = blk_a(audio, policy=policy_a)
                    image = blk_i(image, policy=policy_i)
                else:
                    audio = blk_a(audio)
                    image = blk_i(image)
        
        x, features = self.output(audio, image, B)
        if self.training:
            if self.distill:
                return x, features, prev_decision.detach(), out_pred_prob, early_output
            else:
                return x, out_pred_prob
        else:
            r = (audio.shape[1] / (audio.shape[1] + image.shape[1]))
            self.ratio = [r, 1 - r, abs(2 * r - 1)]
            if self.distill:
                return x, features
            else:
                return x

