import timm
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer, PatchEmbed
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)

        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def compute_rollout_attention(all_layer_matrices, start_layer=0):
    # adding residual consideration
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    # all_layer_matrices = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
    #                       for i in range(len(all_layer_matrices))]
    joint_attention = all_layer_matrices[start_layer]
    for i in range(start_layer+1, len(all_layer_matrices)):
        joint_attention = all_layer_matrices[i].bmm(joint_attention)
    return joint_attention

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class AdaBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, reduction_dim=32):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=3072, act_layer=act_layer, drop=drop)

        #self.adapter_down = nn.Linear(dim, reduction_dim)
        #self.adapter_act = nn.GELU()
        #self.adapter_up = nn.Linear(reduction_dim, dim)
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        #adapt = self.adapter_down(x)
        #adapt = self.adapter_act(adapt)
        #adapt = self.adapter_up(adapt)
        #x = x + adapt
        return x

class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        cosine = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        #out = x.mm(self.weight)
        return cosine

class PromptLearner(nn.Module):
    def __init__(self, num_classes, prompt_length, prompt_depth, prompt_channels,opt):
        super().__init__()
        self.Prompt_Tokens = nn.Parameter(torch.zeros(prompt_depth, prompt_length, prompt_channels))
        
        self.head = nn.Linear(prompt_channels, num_classes)
#         self.head = Causal_Norm_Classifier(num_classes=num_classes,feat_dim=prompt_channels,num_head=opt.num_head, tau=opt.tau, alpha=opt.alpha, gamma=opt.gamma)
        trunc_normal_(self.Prompt_Tokens, std=.02)
        trunc_normal_(self.head.weight, std=.02)
    def forward(self, x):
#         if not self.training:

#             output,_ = self.head(x, embed)
#             return output
#         else:
#             _, output = self.head(x, embed)
#             return output
        return self.head(x)

class PromptLearner_TDE(nn.Module):
    def __init__(self, num_classes, prompt_length, prompt_depth, prompt_channels,opt):
        super().__init__()
        self.Prompt_Tokens = nn.Parameter(torch.zeros(prompt_depth, prompt_length, prompt_channels))
        
#         self.head = nn.Linear(prompt_channels, num_classes)
        self.head = Causal_Norm_Classifier(num_classes=num_classes,feat_dim=prompt_channels,num_head=opt.num_head, tau=opt.tau, alpha=opt.alpha, gamma=opt.gamma)
        trunc_normal_(self.Prompt_Tokens, std=.02)
        trunc_normal_(self.head.weight, std=.02)
    def forward(self, x,embed):
        if not self.training:

            _,output = self.head(x, embed)
        else:
            _, output = self.head(x, embed)
        return output
#         return self.head(x)
class PromptLearner_ccim(nn.Module):
    def __init__(self, num_classes, prompt_length, prompt_depth, prompt_channels,opt,ccim,confounder,prob):
        super().__init__()
        self.Prompt_Tokens = nn.Parameter(torch.zeros(prompt_depth, prompt_length, prompt_channels))
        self.ccim = ccim
        self.confounder  = confounder
        self.prob   = prob
        self.head = nn.Linear(prompt_channels, num_classes)
        

        trunc_normal_(self.Prompt_Tokens, std=.02)
        trunc_normal_(self.head.weight, std=.02)
    def forward(self, x):

        output = self.ccim(x, self.confounder, self.prob)
        feat = x + output.to(torch.float32)
        return self.head(feat)



class VPT_ViT(VisionTransformer):
    def __init__(self,img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', Prompt_Token_num=1, VPT_type="Deep", opt=None,**kwargs):

        super(VPT_ViT,self).__init__(img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True)
        self.VPT_type = VPT_type
        self.embed_mean = torch.zeros(768).numpy()
        if VPT_type == "Deep":
            self.Prompt_Tokens = nn.Parameter(torch.zeros(depth, Prompt_Token_num, embed_dim))
        else:  # "Shallow"
            self.Prompt_Tokens = nn.Parameter(torch.zeros(1, Prompt_Token_num, embed_dim))
        trunc_normal_(self.Prompt_Tokens, std=.02)
        self.dropout = nn.Dropout(0.1)
#         self.prompt_learner = PromptLearner(num_classes, Prompt_Token_num, depth, embed_dim,opt)
        self.prompt_learner = PromptLearner_TDE(num_classes, Prompt_Token_num, depth, embed_dim,opt)
        self.avgpooling = nn.AdaptiveAvgPool2d((1,1))
        self.remethod = opt.remethod
        self.embed_mean = torch.zeros(768).numpy()
        self.head = nn.Identity()
        self.pre_logits = nn.Identity()

    # def New_CLS_head(self, new_classes=15):
    #     self.head = nn.Linear(self.embed_dim, new_classes)
    #     trunc_normal_(self.head.weight, std=.02)

    def Freeze(self):
        for param in self.parameters():
            param.requires_grad_(False)

        # self.Prompt_Tokens.requires_grad_(True)
        for param in self.prompt_learner.parameters():
            param.requires_grad_(True)

    def obtain_prompt(self):
        prompt_state_dict = {'head': self.head.state_dict(),
                             'Prompt_Tokens': self.Prompt_Tokens}
        # print(prompt_state_dict)
        return prompt_state_dict

    def load_prompt(self, prompt_state_dict):
        self.head.load_state_dict(prompt_state_dict['head'])
        self.Prompt_Tokens = prompt_state_dict['Prompt_Tokens']

    def forward_features(self, x):
        x = self.patch_embed(x)
        # print(x.shape,self.pos_embed.shape)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)


        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        if self.VPT_type == "Deep":

            Prompt_Token_num = self.prompt_learner.Prompt_Tokens.shape[1]

            for i in range(len(self.blocks)):
                # concatenate Prompt_Tokens
                Prompt_Tokens = self.prompt_learner.Prompt_Tokens[i].unsqueeze(0)
                x = torch.cat((x, Prompt_Tokens.expand(x.shape[0], -1, -1)), dim=1)
                num_tokens = x.shape[1]
                x = self.blocks[i](x)[:, :num_tokens - Prompt_Token_num]

        else:  # self.VPT_type == "Shallow"
            # concatenate Prompt_Tokens
            Prompt_Tokens = self.Prompt_Tokens.expand(x.shape[0], -1, -1)
            x = torch.cat((x, Prompt_Tokens), dim=1)
            # Sequntially procees
            x = self.blocks(x)
#         return x
        feat = x[:,1:]
        feat = feat.transpose(2,1)
        feature_map = feat.reshape(feat.shape[0], feat.shape[1], 14, 14)
        feat = self.avgpooling(feature_map)
        feat = feat.view(feat.shape[0],feat.shape[1])
        x = self.norm(x)
        return feat,self.pre_logits(x[:, 0]) # use cls token for cls head

    def forward(self, x):

        feat,x = self.forward_features(x)
        self.embed_mean = 0.8 * self.embed_mean + feat.detach().mean(0).view(-1).cpu().numpy()
        x = self.dropout(x)
#         x = self.prompt_learner(x)
        x = self.prompt_learner(x,self.embed_mean)
        return x

    
class VPT_ViT_cprompt(VisionTransformer):
    def __init__(self,img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', Prompt_Token_num=10, VPT_type="Deep",ccim=None,confounder=None,confounder2 =None, prob=None,opt=None,**kwargs):

        super(VPT_ViT_cprompt,self).__init__(img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True)
        self.VPT_type = VPT_type

        self.ccim = ccim
        self.confounder = confounder
        self.prob = prob
        self.avgpooling = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(0.1)
        self.remethod = opt.remethod
        self.embed_mean = torch.zeros(768).numpy()
        self.pre_logits = nn.Identity()
        self.relu = nn.ReLU()
        self.head = nn.Linear(embed_dim, num_classes)
        self.fc = Causal_Norm_Classifier(num_classes=num_classes,feat_dim=768,num_head=opt.num_head, tau=opt.tau, alpha=opt.alpha, gamma=opt.gamma)
        self.key = nn.Linear(768, 768)
        self.key1 = nn.Linear(768, 768)
        self.key2 = nn.Linear(768, 768)
        self.key3 = nn.Linear(768, 768)
        self.key4 = nn.Linear(768, 768)        
        self.key5 = nn.Linear(768, 768)
        self.key6 = nn.Linear(768, 768)
        self.key7 = nn.Linear(768, 768)
        self.key8 = nn.Linear(768, 768)
        self.key9 = nn.Linear(768, 768)
        self.key10 = nn.Linear(768, 768)
        self.key11 = nn.Linear(768, 768)
        self.key.apply(self._init_weights)
        self.key1.apply(self._init_weights)
        self.key2.apply(self._init_weights)
        self.key3.apply(self._init_weights)
        self.key4.apply(self._init_weights)
        self.key5.apply(self._init_weights)
        self.key6.apply(self._init_weights)
        self.key7.apply(self._init_weights)
        self.key8.apply(self._init_weights)
        self.key9.apply(self._init_weights)
        self.key10.apply(self._init_weights)
        self.key11.apply(self._init_weights)

        self.kkk =[self.key, self.key1, self.key2, self.key3, \
                      self.key4, self.key5, self.key6, self.key7, \
                      self.key8, self.key9, self.key10, self.key11]
    def Freeze(self):
        for param in self.parameters():
            param.requires_grad_(False)

        # self.Prompt_Tokens.requires_grad_(True)
        for param in self.prompt_learner.parameters():
            param.requires_grad_(True)

    def obtain_prompt(self):
        prompt_state_dict = {'head': self.head.state_dict(),
                             'Prompt_Tokens': self.Prompt_Tokens}
        # print(prompt_state_dict)
        return prompt_state_dict

    def load_prompt(self, prompt_state_dict):
        self.head.load_state_dict(prompt_state_dict['head'])
        self.Prompt_Tokens = prompt_state_dict['Prompt_Tokens']
    def get_embed(self):
        return self.embed_mean
    def set_embed(self,embed_mean):
        self.embed_mean = embed_mean
        
    def forward_features(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)

        # concatenate CLS token
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for block in self.blocks:
            x = block(x)

        feat = x[:,1:]
        feat = feat.transpose(2,1)
        feature_map = feat.reshape(feat.shape[0], feat.shape[1], 14, 14)
        feat = self.avgpooling(feature_map)
        feat = feat.view(feat.shape[0],feat.shape[1])
        x = self.norm(x)

        return feat,self.pre_logits(x[:, 0]) # use cls token for cls head
    
    def forward_kcausal_features(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)

        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        if self.VPT_type == "Deep":                
            for i in range(len(self.blocks)):
                num_samples = x.size(0)
                num_vectors_per_sample = 10
                indices = torch.randint(low=0, high=self.confounder.size(0), size=(num_samples, num_vectors_per_sample), device=self.confounder.device)
                global_dict = self.confounder[indices] 

                global_dict = self.kkk[i](global_dict)
                x = torch.cat((x, global_dict),dim=1)
                num_tokens = x.shape[1]
                x = self.blocks[i](x)[:, :num_tokens-10]
        x = self.norm(x)
        return feat, self.pre_logits(x[:, 0]) # use cls token for cls head

    def forward_causal_features(self, x):

        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        num_samples = x.size(0)
        num_vectors_per_sample = 20
        indices = torch.randint(low=0, high=self.confounder.size(0), size=(num_samples, num_vectors_per_sample), device=self.confounder.device)
        global_dict = self.confounder[indices] 
        global_dict = self.kkk[0](global_dict)
        x = torch.cat((x, global_dict),dim=1)
        x = self.blocks(x)
        x = self.norm(x)
        return self.pre_logits(x[:, 0]) # use cls token for cls head
    def forward(self, x):
        x = self.forward_kcausal_features(x)  #多个中间层加入confounder
        output = self.ccim(feat, self.confounder, self.prob)
        feat = feat + output.to(torch.float32)
        x = self.fc(feat)
        return x