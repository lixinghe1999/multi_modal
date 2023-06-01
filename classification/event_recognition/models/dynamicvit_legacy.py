from .vit_model import PredictorLG, batch_index_select, Block, Attention
from torch.cuda.amp import autocast
import torch
import torch.nn as nn
import torch.nn.functional as F
from .merge import bipartite_soft_matching, merge_wavg 
from .apply_merge import ToMeBlock, ToMeAttention, make_tome_class


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
        
        self.num_patches = self.audio.num_patches + self.image.num_patches

        if len(pruning_loc) > 0:
            predictor_list = [PredictorLG(self.embed_dim) for _ in range(len(pruning_loc))]
            self.score_predictor = nn.ModuleList(predictor_list)

        self.distill = distill
        self.pruning_loc = pruning_loc
        self.token_ratio = token_ratio
        self.merge = False
    def apply_merge(self, r=0, trace_source: bool = False, prop_attn: bool = False):
        """
        Applies ToMe to this transformer. Afterward, set r using model.r.

        If you want to know the source of each token (e.g., for visualization), set trace_source = true.
        The sources will be available at model._tome_info["source"] afterward.

        For proportional attention, set prop_attn to True. This is only necessary when evaluating models off
        the shelf. For trianing and for evaluating MAE models off the self set this to be False.
        """
        if r > 0:
            self.merge = True

        for modal in [self.audio, self.image]:
            ToMeVisionTransformer = make_tome_class(modal.__class__)

            modal.__class__ = ToMeVisionTransformer
            modal.r = r
            modal._tome_info = {
                "r": modal.r,
                "size": None,
                "source": None,
                "trace_source": trace_source,
                "prop_attn": prop_attn,
                "class_token": True,
                "distill_token": False,
            }
            for name, module in modal.named_modules():
                if isinstance(module, Block):
                    if int(name.split(".")[1]) in self.pruning_loc:
                        continue
                    else:
                        module.__class__ = ToMeBlock
                        module._tome_info = modal._tome_info
                elif isinstance(module, Attention):
                    if int(name.split(".")[1]) in self.pruning_loc:
                        continue
                    else:
                        module.__class__ = ToMeAttention

    def output(self, audio, image):
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
            x = self.head(x)
            return x, features

    @autocast()
    def forward(self, audio, image):
        if self.merge:
            self.audio._tome_info["size"] = None
            self.image._tome_info["size"] = None
        B = audio.shape[0]
        audio = audio.view(-1, 1, 256, 256)
        image = image.view(-1, 3, 224, 224)
        _, audio = self.audio.preprocess(audio)
        _, image = self.image.preprocess(image) 
        p_count = 0
        out_pred_prob = []
        early_output = []
        prev_decision = torch.ones(B, self.num_patches, 1, dtype=audio.dtype, device=audio.device)
        policy = torch.ones(B, self.num_patches + 2, 1, dtype=audio.dtype, device=audio.device)
        self.distribution = []
        for i, (blk_a, blk_i) in enumerate(zip(self.audio.blocks, self.image.blocks)):
            if i in self.pruning_loc:
                spatial_x = torch.cat([audio[:, 1:], image[:, 1:]], dim=1)
                token_len_audio = audio.shape[1] - 1
                if self.training:
                    pred_score = self.score_predictor[p_count](spatial_x, prev_decision).reshape(B, -1, 2)
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
                    early_output.append(self.output(audio, image)[0])
                else:
                    # L2 activation magnitude
                    score_norm = torch.norm(spatial_x, dim=-1, keepdim=False)
                    # score_predict = self.score_predictor[p_count](spatial_x, None).reshape(B, -1, 2)[:, :, 0]
                    # self.distribution.append(torch.cat([score_norm, score_predict], dim=0))
                    values, indices = torch.sort(score_norm, dim=1, descending=True)
                    # TopK selection
                    num_keep_node = int(score_norm.shape[1] * self.token_ratio)
                    keep_policy = indices[:, :num_keep_node]
                    # repeated prunning
                    # values = values[:, :num_keep_node]
                    # num_keep_node = int(keep_policy.shape[1]) - 16
                    # indices_indices = torch.sort(torch.diff(values, dim=1), dim=1, descending=False)[1] + 1
                    # indices_indices = torch.cat([torch.zeros(B, 1, dtype=indices_indices.dtype, device=indices_indices.device), indices_indices], dim=1)
                    # keep_policy = keep_policy[0, indices_indices[:, :num_keep_node]]


                    prev_decision = torch.ones(B, keep_policy.shape[1], 1, dtype=audio.dtype, device=audio.device)
                    keep_audio = keep_policy < token_len_audio
                    keep_image = keep_policy >= token_len_audio

                    keep_audio = torch.masked_select(keep_policy, mask=keep_audio).unsqueeze(0)
                    keep_image = torch.masked_select(keep_policy, mask=keep_image).unsqueeze(0) - token_len_audio

                    cls_policy = torch.zeros(B, 1, dtype=keep_policy.dtype, device=keep_policy.device)
                    now_policy = torch.cat([cls_policy, keep_audio + 1], dim=1)
                    audio = batch_index_select(audio, now_policy)
                    audio = blk_a(audio)
                    if self.merge:
                        self.audio._tome_info["size"] = batch_index_select(self.audio._tome_info["size"], now_policy)

                    cls_policy = torch.zeros(B, 1, dtype=keep_policy.dtype, device=keep_policy.device)
                    now_policy = torch.cat([cls_policy, keep_image + 1], dim=1)
                    image = batch_index_select(image, now_policy)
                    image = blk_i(image)
                    if self.merge:
                        self.image._tome_info["size"] = batch_index_select(self.image._tome_info["size"], now_policy)

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
        
        x, features = self.output(audio, image)
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

