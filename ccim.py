import torch.nn as nn
from torch.nn import Parameter
import torch
import math
import torch.nn.functional as F
# from models.vit import VisionTransformer

class CCIM(nn.Module):
    def __init__(self, num_joint_feature, num_gz, strategy):
        super(CCIM, self).__init__()
        self.num_joint_feature = num_joint_feature
        self.num_gz = num_gz
        if strategy == 'dp_cause':
            self.causal_intervention = dot_product_intervention(num_gz, num_joint_feature)
        elif strategy == 'ad_cause':
            self.causal_intervention = additive_intervention(num_gz, num_joint_feature)
        else:
            raise ValueError("Do Not Exist This Strategy.")

    def forward(self, joint_feature, confounder_dictionary, prior):
    
        g_z = self.causal_intervention(confounder_dictionary, joint_feature, prior)   # (64, 768)
        return g_z


class dot_product_intervention(nn.Module):
    def __init__(self, con_size, fuse_size):
        super(dot_product_intervention, self).__init__()
        self.con_size = con_size
        self.fuse_size = fuse_size
        self.query = nn.Linear(self.fuse_size, 256, bias=False)
        self.key = nn.Linear(self.con_size, 256, bias=False)

    def forward(self, confounder_set, fuse_rep, probabilities):

        query = self.query(fuse_rep)        # (64, 768) -> (64, 256)
        key = self.key(confounder_set)      # (172, 300) -> (172, 256)
        mid = torch.matmul(query, key.transpose(0, 1)) / math.sqrt(self.con_size)     # (64, 172) / scalar=300
        attention = F.softmax(mid, dim=-1)     # (64, 172)
        attention = attention.unsqueeze(2)     # (64, 172, 1) * (172, 300) * (172,)
        fin = (attention * confounder_set * probabilities).sum(1)    # (64, 172, 300) -> sum1:  (64, 300)
        
        return fin


class additive_intervention(nn.Module):
    def __init__(self, con_size, fuse_size):
        super(additive_intervention, self).__init__()
        self.con_size = con_size
        self.fuse_size = fuse_size
        self.Tan = nn.Tanh()
        self.query = nn.Linear(self.fuse_size, 256, bias=False)
        self.key = nn.Linear(self.con_size, 256, bias=False)
        self.w_t = nn.Linear(256, 1, bias=False)

    def forward(self, confounder_set, fuse_rep, probabilities):
        query = self.query(fuse_rep)

        key = self.key(confounder_set)

        query_expand = query.unsqueeze(1)
        fuse = query_expand + key
        fuse = self.Tan(fuse)
        attention = self.w_t(fuse)
        attention = F.softmax(attention, dim=1)
        fin = (attention * confounder_set * probabilities).sum(1)

        return fin


# class CausalVit(nn.Module):
#     def __init__(self, ccim, vit=None, confounder=None, prob=None, emb_dim=768, num_cls=172):
#         super(CausalVit, self).__init__()
#         if vit:
#             self.vit = vit    # load pretrained weights
#         self.vit_new = VisionTransformer(num_classes=172,
#                                       patch_size=(16, 16), image_size=(224, 224)) # train from start
#         self.ccim = ccim
#         self.classifier = nn.Linear(emb_dim, num_cls)
#         self.confounder = confounder
#         self.prob = prob

#     def forward(self, x, main_body=None):
#         if main_body is not None:
#             _, feat = self.vit(x)        # pretrained vit to handle the whole image input
#             feat = feat + self.vit_new(main_body)[1]
#         else:
#             _, feat = self.vit_new(x)
#         output = self.ccim(feat, self.confounder, self.prob)
#         feat = feat + output.to(torch.float32)
#         # classifier
#         return self.classifier(feat)
    
    
