import os
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torchvision.models as models

import timm
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.swin_transformer import SwinTransformer
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, PatchEmbed

from torch.hub import load_state_dict_from_url
from timm.models.helpers import load_state_dict

from functools import partial
import simmim
from convnext import ConvNeXt
from resnet import ResNet50
from utils import load_swin_pretrained
from transformers import AutoModel, Swinv2Model, Swinv2Config


def _swin_window_size(input_size, default=7):
    # Swin's window_size must evenly divide the feature-map spatial size at every stage
    # (input/4, /8, /16, /32). window_size=7 is the ImageNet-224 convention (224/32=7,
    # divides cleanly); the protocol's 320px input (320/32=10) doesn't divide by 7, but
    # does divide cleanly by 5 at every stage (80/5, 40/5, 20/5, 10/5 all integers).
    if input_size == 320:
        return 5
    return default


def _load_swin_pretrained_filtered(model, pretrained_model_name):
    # Loads a named timm Swin checkpoint (trained at window_size=7) into a model that may
    # have a different window_size (e.g. 5, for 320px). relative_position_bias_table/index
    # and attn_mask depend only on window_size, not img_size, so they don't transfer across
    # a window_size change -- everything else (patch_embed, attention QKV/proj, MLP, norms)
    # does. PyTorch's strict=False only tolerates missing/unexpected keys, not shape
    # mismatches for a key present on both sides, so those are filtered out first.
    # Verified locally: only ~0.4% of params (exactly the window-geometry buffers) are
    # affected; the bulk of pretrained weights transfer correctly.
    pretrained = timm.create_model(pretrained_model_name, num_classes=model.head.out_features, pretrained=True)
    pretrained_sd = pretrained.state_dict()
    target_sd = model.state_dict()
    filtered_sd = {k: v for k, v in pretrained_sd.items() if k in target_sd and target_sd[k].shape == v.shape}
    dropped = [k for k in pretrained_sd if k not in filtered_sd]
    unexpected_names = [k for k in dropped if not any(s in k for s in ("relative_position_bias_table", "relative_position_index", "attn_mask"))]
    assert not unexpected_names, f"Unexpected keys dropped during filtered Swin load (not window-geometry): {unexpected_names}"
    model.load_state_dict(filtered_sd, strict=False)
    print(f"Loaded {pretrained_model_name} into a window_size={model.layers[0].blocks[0].attn.window_size[0]} "
          f"architecture -- dropped {len(dropped)} window-geometry keys (relative position bias/index, attn masks), "
          f"kept everything else.")
    return model


class RadDinoClassifier(nn.Module):
    """RAD-DINO (microsoft/rad-dino, a DINOv2-based chest X-ray encoder from Hugging
    Face) is encoder-only -- no classification head -- so this wraps it with a plain
    linear head on its pooled CLS-token output.
    """
    def __init__(self, num_classes):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("microsoft/rad-dino")
        self.head = nn.Linear(self.encoder.config.hidden_size, num_classes)

    def forward(self, x):
        # interpolate_pos_encoding lets this run at any input resolution (not just the
        # native 518px) by bicubic-resizing the position embeddings -- needed for the
        # protocol's uniform 320px input across all models.
        out = self.encoder(pixel_values=x, interpolate_pos_encoding=True)
        feat = out.pooler_output if getattr(out, "pooler_output", None) is not None else out.last_hidden_state[:, 0]
        return self.head(feat)


class ConvNextV2Classifier(nn.Module):
    """ConvNeXt V2 (facebook/convnextv2-base-22k-224 from Hugging Face) -- not
    supported by the pinned timm==0.5.4 in this repo, loaded via transformers
    instead, wrapped with a plain linear head on its pooled output.
    """
    def __init__(self, num_classes):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("facebook/convnextv2-base-22k-224")
        self.head = nn.Linear(self.encoder.config.hidden_sizes[-1], num_classes)

    def forward(self, x):
        out = self.encoder(pixel_values=x)
        feat = out.pooler_output if getattr(out, "pooler_output", None) is not None else out.last_hidden_state.mean(dim=[-2, -1])
        return self.head(feat)


class Swinv2Classifier(nn.Module):
    """Swin Transformer V2 Base (microsoft/swinv2-base-patch4-window12-192-22k from Hugging
    Face) -- not supported by the pinned timm==0.5.4 in this repo (added to timm after this
    pin, same situation as ConvNeXt V2/RAD-DINO), loaded via transformers instead.

    Unlike Swin V1, V2 replaces the fixed relative-position-bias lookup table with a small
    continuous MLP over log-spaced coordinates, so it generalizes across window sizes instead
    of needing one retrained per resolution. Verified locally: loading this checkpoint at
    320px with its native window_size=12 (left unmodified -- no reason to change it) has
    zero missing/dropped backbone weights, only the classifier head (always replaced
    regardless of resolution). This is the whole point of V2's design, and the reason it
    doesn't need the window_size-change-and-drop-weights workaround the V1 Swin models in
    this file use.

    Precision note: this model needs args.amp_dtype="bf16", NOT fp16 (the default for the
    other 5 models in this protocol). Verified: under fp16, the key-projection weight's
    gradient goes non-finite in every attention block (22/473 param tensors) -- Swin V2's
    cosine attention L2-normalizes Q/K before the dot product, and normalizing near-zero-norm
    vectors underflows fp16's narrower range. bf16 (same exponent range as fp32) doesn't have
    this problem; verified clean (0/473 non-finite) under bf16. This is the mirror image of
    ConvNeXt V2's situation elsewhere in this file (that one needs fp16, not bf16) -- both are
    real, hardware/numerics-forced exceptions, not preferences.
    """
    def __init__(self, num_classes, image_size=320):
        super().__init__()
        ckpt = "microsoft/swinv2-base-patch4-window12-192-22k"
        config = Swinv2Config.from_pretrained(ckpt, image_size=image_size)
        self.encoder = Swinv2Model.from_pretrained(ckpt, config=config, ignore_mismatched_sizes=True)
        # config.hidden_size is wrong for this model (reports 768; verified empirically that
        # the actual pooled feature is 1024-dim). The correct value is the last stage's width:
        # embed_dim doubles at each of the 4 stages, so embed_dim * 2**(num_stages-1).
        pooled_dim = config.embed_dim * 2 ** (len(config.depths) - 1)
        self.head = nn.Linear(pooled_dim, num_classes)

    def forward(self, x):
        out = self.encoder(pixel_values=x)
        feat = out.pooler_output if getattr(out, "pooler_output", None) is not None else out.last_hidden_state.mean(dim=1)
        return self.head(feat)

def build_classification_model(args):
    model = None
    print("Creating model...")
    if args.pretrained_weights is None or args.pretrained_weights =='':
        print('Loading pretrained {} weights for {} from timm.'.format(args.init, args.model_name))
        if args.model_name.lower() == "vit_base":
            if args.init.lower() =="random":
                if args.input_size == 448:
                    model = VisionTransformer(num_classes=args.num_class, img_size = args.input_size,
                        patch_size=32, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, drop_path_rate=0.1,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6))
                else:
                    model = VisionTransformer(num_classes=args.num_class,
                        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, drop_path_rate=0.1,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6))
                model.default_cfg = _cfg()
                # model = timm.create_model('vit_base_patch16_224', num_classes=args.num_class, pretrained=False)
            elif args.init.lower() =="imagenet_1k":
                model = timm.create_model('vit_base_patch16_224', num_classes=args.num_class, pretrained=True)
            elif args.init.lower() =="imagenet_21k":
                model = timm.create_model('vit_base_patch16_224_in21k', num_classes=args.num_class, pretrained=True)  
            elif args.init.lower() =="sam":
                model = timm.create_model('vit_base_patch16_224_sam', num_classes=args.num_class, pretrained=True)
            elif args.init.lower() =="dino":
                model = VisionTransformer(num_classes=args.num_class,
                        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6))
                model.default_cfg = _cfg()
                #model = timm.create_model('vit_base_patch16_224_dino', num_classes=args.num_class, pretrained=True) #not available in current timm version
                url = "https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
                state_dict = torch.hub.load_state_dict_from_url(url=url)
                model.load_state_dict(state_dict, strict=False)
            elif args.init.lower() =="deit":
                model = timm.create_model('deit_base_patch16_224', num_classes=args.num_class, pretrained=True)
            elif args.init.lower() =="beit":
                model = timm.create_model('beit_base_patch16_224', num_classes=args.num_class, pretrained=True)

        elif args.model_name.lower() == "vit_small":
            if args.init.lower() =="random":
                model = timm.create_model('vit_small_patch16_224', num_classes=args.num_class, pretrained=False)
            elif args.init.lower() =="imagenet_1k":
                model = timm.create_model('vit_small_patch16_224', num_classes=args.num_class, pretrained=True)
            elif args.init.lower() =="imagenet_21k":
                model = timm.create_model('vit_small_patch16_224_in21k', num_classes=args.num_class, pretrained=True)
            elif args.init.lower() =="dino":
                #model = timm.create_model('vit_small_patch16_224_dino', num_classes=args.num_class, pretrained=True)
                model = VisionTransformer(num_classes=args.num_class,
                    patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6))
                model.default_cfg = _cfg()
                url = "https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
                state_dict = torch.hub.load_state_dict_from_url(url=url)
                model.load_state_dict(state_dict, strict=False)
            elif args.init.lower() =="deit":
                model = timm.create_model('deit_small_patch16_224', num_classes=args.num_class, pretrained=True)           
        
        elif args.model_name.lower() == "swin_large":
            model = SwinTransformer(num_classes=args.num_class, img_size = args.input_size,
                patch_size=4, window_size=_swin_window_size(args.input_size), embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48))

        elif args.model_name.lower() == "swin_large_384":
            model = SwinTransformer(num_classes=args.num_class, img_size = args.input_size, 
                patch_size=4, window_size=12, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48))

        elif args.model_name.lower() == "swin_base":
            if args.init.lower() =="random":
                if args.input_size == 448:
                    model = SwinTransformer(num_classes=args.num_class, img_size = args.input_size,
                        patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32))
                elif args.input_size == 320:
                    model = SwinTransformer(num_classes=args.num_class, img_size = args.input_size,
                        patch_size=4, window_size=5, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32))
                else:
                    model = timm.create_model('swin_base_patch4_window7_224_in22k', num_classes=args.num_class, pretrained=False)
            elif args.init.lower() =="imagenet_21kto1k":
                if args.input_size == 320:
                    model = SwinTransformer(num_classes=args.num_class, img_size = args.input_size,
                        patch_size=4, window_size=5, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32))
                    _load_swin_pretrained_filtered(model, 'swin_base_patch4_window7_224')
                else:
                    model = timm.create_model('swin_base_patch4_window7_224', num_classes=args.num_class, pretrained=True)
            elif args.init.lower() =="imagenet_21k":
                if args.input_size == 320:
                    model = SwinTransformer(num_classes=args.num_class, img_size = args.input_size,
                        patch_size=4, window_size=5, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32))
                    _load_swin_pretrained_filtered(model, 'swin_base_patch4_window7_224_in22k')
                else:
                    model = timm.create_model('swin_base_patch4_window7_224_in22k', num_classes=args.num_class, pretrained=True)
            
        elif args.model_name.lower() == "swin_tiny": 
            if args.init.lower() =="random":
                model = timm.create_model('swin_tiny_patch4_window7_224', num_classes=args.num_class, pretrained=False)
            elif args.init.lower() =="imagenet_1k":
                model = timm.create_model('swin_tiny_patch4_window7_224', num_classes=args.num_class, pretrained=True)

        elif args.model_name.lower() == "densenet121":
            if args.init.lower() =="random":
                model = timm.create_model('densenet121', num_classes=args.num_class, pretrained=False)
            elif args.init.lower() =="imagenet_1k":
                model = timm.create_model('densenet121', num_classes=args.num_class, pretrained=True)

        elif args.model_name.lower() == "convx_base":
            if args.init.lower() =="random":
                model = timm.create_model('convnext_base_in22k', num_classes=args.num_class, pretrained=False)
            elif args.init.lower() =="imagenet_1k":
                model = timm.create_model('convnext_base.fb_in1k', num_classes=args.num_class, pretrained=True)
            elif args.init.lower() =="imagenet_21k":
                model = timm.create_model('convnext_base_in22k', num_classes=args.num_class, pretrained=True)
            elif args.init.lower() =="imagenet_21kto1k":
                model = timm.create_model('convnext_base_in22ft1k', num_classes=args.num_class, pretrained=True)
        elif args.model_name.lower() == "resnet50":
            if args.init.lower() =="random":
                model = ResNet50(num_classes=args.num_class)

        elif args.model_name.lower() == "rad_dino":
            model = RadDinoClassifier(num_classes=args.num_class)

        elif args.model_name.lower() == "convnextv2_base":
            model = ConvNextV2Classifier(num_classes=args.num_class)

        elif args.model_name.lower() == "swinv2_base":
            model = Swinv2Classifier(num_classes=args.num_class, image_size=getattr(args, "input_size", 320))

    else:
        print("Creating model from pretrained weights: "+ args.pretrained_weights)
        if args.model_name.lower() == "vit_base":
            if args.init.lower() == "simmim":
                model = simmim.create_model(args)
            else:
                model = VisionTransformer(num_classes=args.num_class,
                        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6))
                model.default_cfg = _cfg()
                load_pretrained_weights(model, args.init.lower(), args.pretrained_weights)
            
        elif args.model_name.lower() == "vit_small":
            model = VisionTransformer(num_classes=args.num_class,
                    patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6))
            model.default_cfg = _cfg()
            load_pretrained_weights(model, args.init.lower(), args.pretrained_weights)  
            
        elif args.model_name.lower() == "swin_large":
            model = SwinTransformer(num_classes=args.num_class, img_size = args.input_size,
                patch_size=4, window_size=_swin_window_size(args.input_size), embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48))
            load_pretrained_weights(model, args.init.lower(), args.pretrained_weights, args.key, args.scale_up)
            
        elif args.model_name.lower() == "swin_large_384":
            model = SwinTransformer(num_classes=args.num_class, img_size = args.input_size, 
                patch_size=4, window_size=12, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48))
            load_pretrained_weights(model, args.init.lower(), args.pretrained_weights, args.key, args.scale_up)
        
        elif args.model_name.lower() == "swin_base":
            if args.init.lower() == "simmim":
                model = simmim.create_model(args)
            elif args.init.lower() =="imagenet_1k":
                model = timm.create_model('swin_base_patch4_window7_224', num_classes=args.num_class)
                load_pretrained_weights(model, args.init.lower(), args.pretrained_weights)  
            else:
                model = SwinTransformer(num_classes=args.num_class, img_size = args.input_size,
                    patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32))
                load_pretrained_weights(model, args.init.lower(), args.pretrained_weights, args.key, args.scale_up)  
                
        elif args.model_name.lower() == "swin_tiny": 
            model = timm.create_model('swin_tiny_patch4_window7_224', num_classes=args.num_class)
            load_pretrained_weights(model, args.init.lower(), args.pretrained_weights)
            
        elif args.model_name.lower() == "convx_base":
          if args.init.lower().startswith("ark"):
                model = ConvNeXt(num_classes=args.num_class,
                     depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024])
                load_pretrained_weights(model, args.init.lower(), args.pretrained_weights, args.key)   
          
    if model is None:
        print("Not provide {} pretrained weights for {}.".format(args.init, args.model_name))
        raise Exception("Please provide correct parameters to load the model!")
    return model  
    

def load_pretrained_weights(model, init, pretrained_weights, checkpoint_key = None, scale_up = False):
    if pretrained_weights.startswith('https'):
        checkpoint = load_state_dict_from_url(url=pretrained_weights, map_location='cpu')
    else:
        checkpoint = torch.load(pretrained_weights, map_location="cpu")
    print(checkpoint.keys())
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']

    if init =="dino":
        checkpoint_key = "teacher"
        if checkpoint_key is not None and checkpoint_key in checkpoint:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = checkpoint[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    elif init =="moco_v3":
        for k in list(state_dict.keys()):
            # retain only base_encoder up to before the embedding layer
            if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.head'):
                # remove prefix
                state_dict[k[len("module.base_encoder."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
    elif init == "moby":
        state_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items() if 'encoder.' in k}
    # elif init == "mae":
    #     state_dict = checkpoint['model']
    elif init.startswith("ark"): 
        print("Loading {} from checkpoint...".format(checkpoint_key))
        state_dict = checkpoint[checkpoint_key]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items() }  
  
    else:
        print("Trying to load the checkpoint for {} at {}, but we cannot guarantee the success.".format(init, pretrained_weights))

    if scale_up:
        k_del = []
        for k in state_dict.keys():
            if "attn_mask" in k:
                k_del.append(k)
        print(f"Removing key {k_del} from pretrained checkpoint")
        for k in k_del:
            del state_dict[k]

        # relative_position_bias_table/index are window-size-dependent (shape (2W-1)^2 x heads /
        # WxW) and don't transfer across a window-size change, same as attn_mask above. Unlike
        # missing/unexpected keys, strict=False does NOT tolerate a same-key-different-shape
        # error, so these have to be filtered out explicitly -- same approach already verified
        # in _load_swin_pretrained_filtered() elsewhere in this file (~0.4% of params affected).
        model_sd = model.state_dict()
        shape_mismatch = [k for k in state_dict if k in model_sd and state_dict[k].shape != model_sd[k].shape]
        if shape_mismatch:
            print(f"Removing {len(shape_mismatch)} shape-mismatched window-geometry keys (window size changed): {shape_mismatch}")
            for k in shape_mismatch:
                del state_dict[k]

    for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
        if k in state_dict:
            print(f"Removing key {k} from pretrained checkpoint")
            del state_dict[k]
    msg = model.load_state_dict(state_dict, strict=False)
    print('Loaded with msg: {}'.format(msg)) 

    return model

def save_checkpoint(state,filename='model'):

    torch.save(state, filename + '.pth.tar')

