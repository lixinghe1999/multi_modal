import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))
from torchvision.models.vision_transformer import (
    vit_b_16,
    # vit_b_32,
    # vit_l_16,
    # vit_l_32,
    # vit_h_14,
)
# from models.mbt import MBT
from models.mbt_timm import MBT
if __name__ == "__main__":

    entries = globals().copy()

    import torch
    import torch.nn as nn
    import torch_pruning as tp
    import random

    def my_prune(model, example_inputs, output_transform, model_name):
        
        from torchvision.models.vision_transformer import VisionTransformer
        from torchvision.models.convnext import CNBlock, ConvNeXt

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        ori_size = tp.utils.count_params(model)
        model.cpu().eval()
        ignored_layers = []
        for p in model.parameters():
            p.requires_grad_(True)
        #########################################
        # Ignore unprunable modules
        #########################################
        for name, m in model.named_modules():
            if isinstance(m, nn.Linear) and m.out_features == 309:
                print(name, m)
                ignored_layers.append(m)
            #elif isinstance(m, nn.modules.linear.NonDynamicallyQuantizableLinear):
            #    ignored_layers.append(m) # this module is used in Self-Attention
        # For ViT: Rounding the number of channels to the nearest multiple of num_heads
        round_to = None
        if isinstance( model, VisionTransformer): round_to = model.encoder.layers[0].num_heads

        #########################################
        # (Optional) Register unwrapped nn.Parameters 
        # TP will automatically detect unwrapped parameters and prune the last dim for you by default.
        # If you want to prune other dims, you can register them here.
        #########################################
        unwrapped_parameters = None
        #if model_name=='ssd300_vgg16':
        #    unwrapped_parameters=[ (model.backbone.scale_weight, 0) ] # pruning the 0-th dim of scale_weight
        #if isinstance( model, VisionTransformer):
        #    unwrapped_parameters = [ (model.class_token, 0), (model.encoder.pos_embedding, 0)]
        #elif isinstance(model, ConvNeXt):
        #    unwrapped_parameters = []
        #    for m in model.modules():
        #        if isinstance(m, CNBlock):
        #            unwrapped_parameters.append( (m.layer_scale, 0) )

        #########################################
        # Build network pruners
        #########################################
        importance = tp.importance.MagnitudeImportance(p=1)
        pruner = tp.pruner.MagnitudePruner(
            model,
            example_inputs=example_inputs,
            importance=importance,
            iterative_steps=1,
            ch_sparsity=0.5,
            round_to=round_to,
            unwrapped_parameters=unwrapped_parameters,
            ignored_layers=ignored_layers,
        )

        # DG = tp.DependencyGraph()
        # DG.build_dependency(model, example_inputs=example_inputs)

        # get a pruning group according to the dependency graph. idxs is the indices of pruned filters.
        # pruning_group = DG.get_pruning_group(
        #     model.image, tp.prune_linear_out_channels, idxs=[0, 2, 4]
        # )
        # execute this group (prune the model)
        


        #########################################
        # Pruning 
        #########################################
        print("==============Before pruning=================")
        print("Model Name: {}".format(model_name))
        # print(model)
        pruner.step()
        # Torchvision relies on the hidden_dim variable for forwarding, so we have to modify this varaible after pruning
        model.image.hidden_dim = model.image.conv_proj.out_channels
        #print(model.image.class_token.shape, model.image.encoder.pos_embedding.shape)
        model.audio.hidden_dim = model.audio.conv_proj.out_channels
        # print(model.audio.class_token.shape, model.audio.encoder.pos_embedding.shape)
        print("==============After pruning=================")
        # print(model)

        #########################################
        # Testing 
        #########################################
        with torch.no_grad():
            if isinstance(example_inputs, dict):
                out = model(**example_inputs)
            else:
                out = model(example_inputs)
            if output_transform:
                out = output_transform(out)
            print("{} Pruning: ".format(model_name))
            print("  Params: %s => %s" % (ori_size, tp.utils.count_params(model)))
            if isinstance(out, (dict,list,tuple)):
                print("  Output:")
                for o in tp.utils.flatten_as_list(out):
                    print(o.shape)
            else:
                print("  Output:", out.shape)
            print("------------------------------------------------------\n")

    successful = []
    unsuccessful = []
    model_name = 'mbt'
    # entry = vit_b_16
    entry = MBT

    # example_inputs = torch.randn(1, 3, 224, 224)
    example_inputs= {'audio': torch.randn(1, 256, 256), 'image': torch.randn(1, 3, 224, 224)}
    
    model = entry()

    output_transform = None

    # try:
    my_prune(
        model, example_inputs=example_inputs, output_transform=output_transform, model_name=model_name
    )
    successful.append(model_name)
    # except Exception as e:
    #     print(e)
    #     unsuccessful.append(model_name)
    print("Successful Pruning: %d Models\n"%(len(successful)), successful)
    print("")
    print("Unsuccessful Pruning: %d Models\n"%(len(unsuccessful)), unsuccessful)
    sys.stdout.flush()