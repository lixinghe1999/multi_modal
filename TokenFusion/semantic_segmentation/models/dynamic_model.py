import torch.nn as nn
from . import dynamic_segformer
import torch.nn.functional as F
from mmcv.cnn import ConvModule
import torch

class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

class SegFormerHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, feature_strides=None, in_channels=128, embedding_dim=256, num_classes=20, **kwargs):
        super(SegFormerHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        #decoder_params = kwargs['decoder_params']
        #embedding_dim = decoder_params['embed_dim']

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)
        self.dropout = nn.Dropout2d(0.1)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='BN', requires_grad=True)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, x):
        c1, c2, c3, c4 = x

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x

class Dynamic_Model(nn.Module):
    def __init__(self, backbone, num_classes=20, embedding_dim=256, pretrained=True,
                 pruning_loc=[1, 2, 3], token_ratio=[0.9, 0.7, 0.5]):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.feature_strides = [4, 8, 16, 32]
        self.num_parallel = 2
        self.pruning_loc = pruning_loc
        self.token_ratio = token_ratio
        self.encoder_rgb = getattr(dynamic_segformer, backbone)()
        self.encoder_depth = getattr(dynamic_segformer, backbone)()
        self.in_channels = self.encoder_rgb.embed_dims

        if pretrained:
            state_dict = torch.load('pretrained/' + backbone + '.pth')
            state_dict.pop('head.weight')
            state_dict.pop('head.bias')
            self.encoder_rgb.load_state_dict(state_dict, strict=True)
            self.encoder_depth.load_state_dict(state_dict, strict=True)

        if len(pruning_loc) > 0:
            predictor_list = [dynamic_segformer.PredictorLG(self.in_channels[i]) for i in range(len(pruning_loc))]
            self.score_predictor = nn.ModuleList(predictor_list)

        self.decoder = SegFormerHead(feature_strides=self.feature_strides, in_channels=self.in_channels,
                                     embedding_dim=self.embedding_dim, num_classes=self.num_classes)

        self.alpha = nn.Parameter(torch.ones(self.num_parallel, requires_grad=True))
        self.register_parameter('alpha', self.alpha)

    def get_param_groups(self):
        param_groups = [[], [], []]
        for name, param in list(self.encoder.named_parameters()):
            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)
        for param in list(self.decoder.parameters()):
            param_groups[2].append(param)
        return param_groups

    def forward(self, x):
        rgb, depth = x
        B = rgb.shape[0]
        outs = []
        p_count = 0
        prev_decision = torch.ones(B, self.num_patches, 1, dtype=rgb.dtype, device=rgb.device)
        policy = torch.ones(B, self.num_patches + 2, 1, dtype=rgb.dtype, device=rgb.device)

        # stage 1
        rgb, H, W = self.patch_embed1(rgb)
        depth, H, W = self.patch_embed1(depth)
        for i, (blk_r, blk_d) in enumerate(zip(self.encoder_rgb.block1, self.encoder_depth.block1)):
            rgb = blk_r(rgb, H, W)
            depth = blk_d(depth, H, W)
        rgb, depth = self.encoder_rgb.norm1(rgb), self.encoder_depth.norm1(depth)
        rgb = rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        depth = rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append([rgb, depth])

        # stage 2
        rgb, H, W = self.patch_embed2(rgb)
        depth, H, W = self.patch_embed2(depth)
        spatial_x = torch.cat([rgb, depth], dim=1)
        pred_score = self.score_predictor[p_count](spatial_x, prev_decision).reshape(B, -1, 2)
        token_len = rgb.shape[-1]
        if self.training:
            rgb = [rgb, rgb]
            depth = [depth, depth]
            hard_keep_decision = F.gumbel_softmax(pred_score, hard=True)[:, :, 0:1] * prev_decision
            decision_rgb = hard_keep_decision[:, :token_len]
            decision_depth = hard_keep_decision[:, token_len:]
        else:
            score = pred_score[:, :, 0]
            values, indices = torch.sort(score, dim=1, descending=False)
            num_keep_node = int(self.num_patches * self.token_ratio[p_count])
            keep_policy = indices[:, -num_keep_node:]
            # prev_decision = dynamic_segformer.batch_index_select(prev_decision, keep_policy)
            keep_rgb = keep_policy < token_len
            keep_depth = keep_policy >= token_len
            decision_rgb = torch.masked_select(keep_policy, mask=keep_audio).unsqueeze(0)
            decision_depth = torch.masked_select(keep_policy, mask=keep_image).unsqueeze(0) - token_len
        for i, (blk_r, blk_d) in enumerate(zip(self.encoder_rgb.block2, self.encoder_depth.block2)):
            rgb = blk_r(rgb, H, W, decision_rgb)
            depth = blk_d(depth, H, W, decision_depth)
        p_count += 1
        rgb, depth = self.encoder_rgb.norm2(rgb), self.encoder_depth.norm2(depth)
        rgb = rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        depth = rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append([rgb, depth])

        # stage 3
        rgb, H, W = self.patch_embed3(rgb)
        depth, H, W = self.patch_embed3(depth)
        for i, (blk_r, blk_d) in enumerate(zip(self.encoder_rgb.block3, self.encoder_depth.block3)):
            rgb = blk_r(rgb, H, W)
            depth = blk_d(depth, H, W)
        rgb, depth = self.encoder_rgb.norm3(rgb), self.encoder_depth.norm3(depth)
        rgb = rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        depth = rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append([rgb, depth])

        # stage 4
        rgb, H, W = self.patch_embed4(rgb)
        depth, H, W = self.patch_embed4(depth)
        for i, (blk_r, blk_d) in enumerate(zip(self.encoder_rgb.block4, self.encoder_depth.block4)):
            rgb = blk_r(rgb, H, W)
            depth = blk_d(depth, H, W)
        rgb, depth = self.encoder_rgb.norm4(rgb), self.encoder_depth.norm4(depth)
        rgb = rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        depth = rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append([rgb, depth])

        x = [self.decoder(rgb), self.decoder(depth)]
        ens = 0
        alpha_soft = F.softmax(self.alpha, dim=-1)
        for l in range(self.num_parallel):
            ens += alpha_soft[l] * x[l].detach()
        x.append(ens)
        return x