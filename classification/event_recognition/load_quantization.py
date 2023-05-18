from models.vit_model import VisionTransformerDiffPruning
import torch
if __name__ == "__main__":
    config = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                  pruning_loc=())
    model = VisionTransformerDiffPruning(**config)
    # model.load_state_dict(torch.load('assets/deit_base_patch16_224.pth')['model'])

    # model.load_state_dict(torch.load('assets/quantized_deit_base_patch16_224.pth'))
    # model.load_state_dict(torch.load('assets/deit_base_patch16_224-b5f2ef4d.pth')['model'])