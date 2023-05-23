import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))
from torchvision.models.vision_transformer import (
    vit_b_16,
    # vit_b_32,
    # vit_l_16,
    # vit_l_32,
    # vit_h_14,
)

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
        for m in model.modules():
            if isinstance(m, nn.Linear) and m.out_features == 1000:
                ignored_layers.append(m)
            #elif isinstance(m, nn.modules.linear.NonDynamicallyQuantizableLinear):
            #    ignored_layers.append(m) # this module is used in Self-Attention
        if 'ssd' in model_name:
            ignored_layers.append(model.head)
        if model_name=='raft_large':
            ignored_layers.extend(
                [model.corr_block, model.update_block, model.mask_predictor]
            )
        if 'fasterrcnn' in model_name:
            ignored_layers.extend([ 
                 model.rpn.head.cls_logits, model.rpn.head.bbox_pred, model.backbone.fpn, model.roi_heads
            ])
        if model_name=='fcos_resnet50_fpn':
            ignored_layers.extend([model.head.classification_head.cls_logits, model.head.regression_head.bbox_reg, model.head.regression_head.bbox_ctrness])
        if model_name=='keypointrcnn_resnet50_fpn':
            ignored_layers.extend([model.rpn.head.cls_logits, model.backbone.fpn.layer_blocks, model.rpn.head.bbox_pred, model.roi_heads.box_head, model.roi_heads.box_predictor, model.roi_heads.keypoint_predictor])
        if model_name=='maskrcnn_resnet50_fpn_v2':
            ignored_layers.extend([model.rpn.head.cls_logits, model.rpn.head.bbox_pred, model.roi_heads.box_predictor, model.roi_heads.mask_predictor])
        if model_name=='retinanet_resnet50_fpn_v2':
            ignored_layers.extend([model.head.classification_head.cls_logits, model.head.regression_head.bbox_reg])
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


        #########################################
        # Pruning 
        #########################################
        print("==============Before pruning=================")
        print("Model Name: {}".format(model_name))
        print(model)
        pruner.step()
        if isinstance(
            model, VisionTransformer
        ):  # Torchvision relies on the hidden_dim variable for forwarding, so we have to modify this varaible after pruning
            model.hidden_dim = model.conv_proj.out_channels
            print(model.class_token.shape, model.encoder.pos_embedding.shape)
        print("==============After pruning=================")
        print(model)

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
    for model_name, entry in entries.items():
        print(model_name)
        if 'vit' in model_name.lower():
            pass
        else:
            continue

        if not callable(entry):
            continue
   
        example_inputs = torch.randn(1, 3, 224, 224)
       
        model = entry()

        output_transform = None

        try:
            my_prune(
                model, example_inputs=example_inputs, output_transform=output_transform, model_name=model_name
            )
            successful.append(model_name)
        except Exception as e:
            print(e)
            unsuccessful.append(model_name)
        print("Successful Pruning: %d Models\n"%(len(successful)), successful)
        print("")
        print("Unsuccessful Pruning: %d Models\n"%(len(unsuccessful)), unsuccessful)
        sys.stdout.flush()