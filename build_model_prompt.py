import scipy.io as matio
import torch
import torch.utils.data
from torch import nn
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import optim
import ipdb
import math
from torch.optim.lr_scheduler import MultiStepLR
from warmup_scheduler import GradualWarmupScheduler
# Modelm
import timm
import math
import torch
import torch.nn as nn

from prompt_model import *
def build_promptmodel(num_classes=2, img_size=224, model_idx='ViT', patch_size=16, base_model='vit_base_patch16_224_in21k',
                      Prompt_Token_num=10, VPT_type="Deep",opt = None):
    # VPT_type = "Deep" / "Shallow"

    if model_idx[0:3] == 'ViT':
        # ViT_Prompt
        import timm
        basic_model = timm.create_model('vit_base_patch16_224_in21k', pretrained=True,num_classes=10,pretrained_cfg_overlay=dict(file='/root/.cache/huggingface/hub/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz'))

        base_state_dict = basic_model.state_dict()
        del base_state_dict['head.weight']
        del base_state_dict['head.bias']
        model = VPT_ViT(num_classes=num_classes, img_size=img_size, patch_size=patch_size, Prompt_Token_num=Prompt_Token_num,VPT_type=VPT_type,opt = opt)

        model.load_state_dict(base_state_dict, False)
        #model.New_CLS_head(num_classes)
    else:
        print("The model is not difined in the Prompt script")
        return -1
    return model



def build_ccim_model_lprompt(num_classes=2, img_size=224, model_idx='ViT', patch_size=16, base_model='vit_base_patch16_224_in21k',Prompt_Token_num='',VPT_type='',ccim=None,confounder=None,confounder2 =None, prob=None,opt=''):

    if model_idx[0:3] == 'ViT':
        # ViT_Prompt
        import timm
        basic_model = timm.create_model('vit_base_patch16_224_in21k', pretrained=True,num_classes=10,pretrained_cfg_overlay=dict(file='/root/.cache/huggingface/hub/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz'))

        base_state_dict = basic_model.state_dict()
        del base_state_dict['head.weight']
        del base_state_dict['head.bias']
        
        model = VPT_ViT_cprompt(num_classes=num_classes, img_size=img_size, patch_size=patch_size,Prompt_Token_num=Prompt_Token_num,VPT_type=VPT_type,ccim=ccim,confounder=confounder,confounder2=confounder2, prob=prob,opt=opt)
        model.load_state_dict(base_state_dict, False)

    else:
        print("The model is not difined in the Prompt script")
        return -1
    

    return model
