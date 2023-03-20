'''
Inference-only model
'''
from model.vit_model import AudioTransformerDiffPruning, VisionTransformerDiffPruning, PredictorLG,\
    batch_index_select
from torch.cuda.amp import autocast
import torch
import torch.nn as nn
import time
from torch.nn.utils.rnn import pad_sequence
class AVnet_Runtime(nn.Module):
    def __init__(self, real_batch=2, \
                 pruning_loc=[3, 6, 9], token_ratio=[0.7, 0.7**2, 0.7**3], pretrained=True):
        super(AVnet_Runtime, self).__init__()
        config = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                      pruning_loc=pruning_loc, token_ratio=token_ratio)
        embed_dim = 768
        self.audio = AudioTransformerDiffPruning(config, imagenet_pretrain=pretrained)
        self.image = VisionTransformerDiffPruning(**config)
        if pretrained:
            self.image.load_state_dict(torch.load('assets/deit_base_patch16_224.pth')['model'], strict=False)

        self.num_patches = self.audio.num_patches + 14 * 14

        self.head = nn.Sequential(nn.LayerNorm(embed_dim * 2), nn.Linear(embed_dim * 2, 309))
        if len(pruning_loc) > 0:
            predictor_list = [PredictorLG(embed_dim) for _ in range(len(pruning_loc))]
            self.score_predictor = nn.ModuleList(predictor_list)
        self.real_batch = real_batch
        self.pruning_loc = pruning_loc
        self.token_ratio = token_ratio
    def output(self, audio, image):
        audio = self.audio.norm(audio)
        image = self.image.norm(image)
        features = torch.cat([audio[:, 1:], image[:, 1:]], dim=1)
        x = torch.cat([audio[:, 0], image[:, 0]], dim=1)
        x = torch.flatten(x, start_dim=1)
        x = self.head(x)
        return x, features

    def cluster_inference(self, audio, image, prev_decision, B):
        spatial_x = torch.cat([audio[:, 1:], image[:, 1:]], dim=1)
        token_len_audio = audio.shape[1] - 1
        pred_score = self.score_predictor[0](spatial_x, prev_decision).reshape(B, -1, 2)
        score = pred_score[:, :, 0]
        num_keep_node = int(self.num_patches * self.token_ratio[0])
        keep_policy = torch.argsort(score, dim=1, descending=True)[:, :num_keep_node]

        keep_audio = keep_policy < token_len_audio
        audio_token = torch.sum(keep_audio, dim=1)

        sorted_batch = torch.argsort(audio_token)
        output = torch.empty(sorted_batch.shape[0], 309, dtype=audio.dtype, device=audio.device)
        for b in range(0, B, self.real_batch):
            batch_audio = audio[sorted_batch[b * self.real_batch: (b + 1) * self.real_batch]]
            batch_image = image[sorted_batch[b * self.real_batch: (b + 1) * self.real_batch]]

            prev_decision = torch.ones(self.real_batch, self.num_patches, 1,
                                       dtype=audio.dtype, device=audio.device)
            print(batch_image.shape, batch_audio.shape)
            batch_output, feature = self.shared_inference(batch_audio, batch_image, prev_decision, self.real_batch)
            print(batch_output.shape, output.shape)
            output[sorted_batch[b * self.real_batch: (b + 1) * self.real_batch]] = batch_output
        return output

    def shared_inference(self, audio, image, prev_decision, B):
        for i in range(len(self.pruning_loc)):
            spatial_x = torch.cat([audio[:, 1:], image[:, 1:]], dim=1)
            token_len_audio = audio.shape[1] - 1
            print(spatial_x.shape, prev_decision.shape)
            pred_score = self.score_predictor[i](spatial_x, prev_decision).reshape(B, -1, 2)

            score = pred_score[:, :, 0]
            num_keep_node = int(self.num_patches * self.token_ratio[i])
            keep_policy = torch.argsort(score, dim=1, descending=True)[:, :num_keep_node]

            keep_audio = keep_policy < token_len_audio
            audio_token = torch.sum(keep_audio, dim=1)
            mask = torch.arange(audio_token.max(), device=keep_policy.device).unsqueeze(0).expand(B, -1)
            policy_a = mask < audio_token.unsqueeze(1).expand(-1, audio_token.max())

            keep_image = keep_policy >= token_len_audio
            image_token = torch.sum(keep_image, dim=1)
            mask = torch.arange(image_token.max(), device=keep_policy.device).unsqueeze(0).expand(B, -1)
            policy_i = mask < image_token.unsqueeze(1).expand(-1, image_token.max())

            cls_mask = torch.ones(B, 1, dtype=keep_policy.dtype, device=keep_policy.device)
            policy_a = torch.cat([cls_mask, policy_a], dim=1).unsqueeze(2)
            policy_i = torch.cat([cls_mask, policy_i], dim=1).unsqueeze(2)

            keep_audio = pad_sequence([keep_policy[b, keep_audio[b]] for b in range(B)], padding_value=0).T
            keep_image = pad_sequence([keep_policy[b, keep_image[b]] for b in range(B)],
                                      padding_value=token_len_audio).T
            keep_image = keep_image - token_len_audio

            cls_policy = torch.zeros(B, 1, dtype=keep_policy.dtype, device=keep_policy.device)
            now_policy = torch.cat([cls_policy, keep_audio + 1], dim=1)
            audio = batch_index_select(audio, now_policy)

            cls_policy = torch.zeros(B, 1, dtype=keep_policy.dtype, device=keep_policy.device)
            now_policy = torch.cat([cls_policy, keep_image + 1], dim=1)
            image = batch_index_select(image, now_policy)

            prev_decision = torch.cat([policy_a[:, 1:], policy_i[:, 1:]], dim=1)
            # register
            # r = (audio_token.max() / (audio_token.max() + image_token.max())).item()
            # ratio.append([torch.mean(audio_token / audio_token.max()).item(),
            #               torch.mean(image_token / image_token.max()).item(),
            #               r, 1 - r, abs(2 * r - 1)])
            for j in range(self.pruning_loc[i], self.pruning_loc[i] + 3):
                blk_a = self.audio.blocks[i]
                blk_i = self.image.blocks[i]

                audio = blk_a(audio, policy=policy_a)
                image = blk_i(image, policy=policy_i)
        x, features = self.output(audio, image)
        return x, features

    def forward(self, audio, image):
        B, audio = self.audio.preprocess(audio.unsqueeze(1))
        B, image = self.image.preprocess(image)

        prev_decision = torch.ones(B, self.num_patches, 1, dtype=audio.dtype, device=audio.device)
        policy_a = torch.ones(B, audio.shape[1], 1, dtype=audio.dtype, device=audio.device)
        policy_i = torch.ones(B, image.shape[1], 1, dtype=audio.dtype, device=audio.device)
        ratio = []
        for i in range(0, self.pruning_loc[0]):
            blk_a = self.audio.blocks[i]
            blk_i = self.image.blocks[i]
            audio = blk_a(audio, policy=policy_a)
            image = blk_i(image, policy=policy_i)
        x = self.cluster_inference(audio, image, prev_decision, B)
        # x, feature = self.shared_inference(audio, image, prev_decision, B)
        return x, ratio
if __name__ == "__main__":
    device = 'cpu'
    base_rate = 0.5
    pruning_loc = [3, 6, 9]
    token_ratio = [base_rate, base_rate ** 2, base_rate ** 3]
    config = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                  pruning_loc=pruning_loc, token_ratio=token_ratio)

    model_image = VisionTransformerDiffPruning(**config).to(device)

    model_audio = AudioTransformerDiffPruning(config, imagenet_pretrain=True).to(device)

    model_teacher = AVnet_Dynamic(pruning_loc=()).to(device)

    multi_modal_model = AVnet_Dynamic().to(device)

    model_image.eval()
    model_audio.eval()
    model_teacher.eval()
    multi_modal_model.eval()

    audio = torch.zeros(1, 384, 128).to(device)
    image = torch.zeros(1, 3, 224, 224).to(device)

    num_iterations = 10
    with torch.no_grad():
        for i in range(num_iterations):
            if i == 1:
                t_start = time.time()
            multi_modal_model(audio, image)
        print((time.time() - t_start) / (num_iterations - 1))

    with torch.no_grad():
        for i in range(num_iterations):
            if i == 1:
                t_start = time.time()
            model_image(image)
        print((time.time() - t_start) / (num_iterations - 1))

    with torch.no_grad():
        for i in range(num_iterations):
            if i == 1:
                t_start = time.time()
            model_audio(audio)
        print((time.time() - t_start) / (num_iterations - 1))

    with torch.no_grad():
        for i in range(num_iterations):
            if i == 1:
                t_start = time.time()
            model_teacher(audio, image)
        print((time.time() - t_start) / (num_iterations-1))
