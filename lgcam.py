import torch
import torch.nn as nn
import ipdb
import math

class Causal_Norm_Classifier(nn.Module):   
    def __init__(self, num_classes=100, feat_dim=768, use_effect=True, num_head=2, tau=16.0, alpha=0.15, gamma=0.03125, *args):
        super(Causal_Norm_Classifier, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(num_classes, feat_dim).cuda(), requires_grad=True)
        self.scale = tau / num_head   # 16.0 / num_head
        self.norm_scale = gamma       # 1.0 / 32.0
        self.alpha = alpha            # 3.0
        self.num_head = num_head
        self.head_dim = feat_dim // num_head
        self.use_effect = use_effect
        self.reset_parameters(self.weight)
        self.relu = nn.ReLU(inplace=True)
        
    def reset_parameters(self, weight):
        stdv = 1. / math.sqrt(weight.size(1))
        weight.data.uniform_(-stdv, stdv)

    def forward(self, x, embed):
        # calculate capsule normalized feature vector and predict
        
        normed_w = self.multi_head_call(self.causal_norm, self.weight, weight=self.norm_scale)
        normed_x = self.multi_head_call(self.l2_norm, x)
        y_TE = torch.mm(normed_x * self.scale, normed_w.t())
        y_TDE = y_TE.clone()

        # remove the effect of confounder c during test
        if (not self.training) and self.use_effect:
            self.embed = torch.from_numpy(embed).view(1, -1).to(x.device)
            normed_c = self.multi_head_call(self.l2_norm, self.embed)
            head_dim = x.shape[1] // self.num_head
            x_list = torch.split(normed_x, head_dim, dim=1)
            c_list = torch.split(normed_c, head_dim, dim=1)
            w_list = torch.split(normed_w, head_dim, dim=1)
            output = []

            for nx, nc, nw in zip(x_list, c_list, w_list):
                cos_val, sin_val = self.get_cos_sin(nx, nc)
                y0 = torch.mm((nx -  cos_val * self.alpha * nc) * self.scale, nw.t())
                output.append(y0)
            y_TDE = sum(output) 
        return y_TDE, y_TE


    def get_cos_sin(self, x, y):
        cos_val = (x * y).sum(-1, keepdim=True) / torch.norm(x, 2, 1, keepdim=True) / torch.norm(y, 2, 1, keepdim=True)
        sin_val = (1 - cos_val * cos_val).sqrt()
        return cos_val, sin_val

    def multi_head_call(self, func, x, weight=None):
        assert len(x.shape) == 2
        x_list = torch.split(x, self.head_dim, dim=1)
        if weight:
            y_list = [func(item, weight) for item in x_list]
        else:
            y_list = [func(item) for item in x_list]
        assert len(x_list) == self.num_head
        assert len(y_list) == self.num_head
        return torch.cat(y_list, dim=1)

    def l2_norm(self, x):
        normed_x = x / torch.norm(x, 2, 1, keepdim=True)
        return normed_x

    def capsule_norm(self, x):
        norm= torch.norm(x.clone(), 2, 1, keepdim=True)
        normed_x = (norm / (1 + norm)) * (x / norm)
        return normed_x

    def causal_norm(self, x, weight):
        norm= torch.norm(x, 2, 1, keepdim=True)
        normed_x = x / (norm + weight)
        return normed_x




class LGCAM(nn.Module):
    def __init__(self, backbone, global_dict=None, num_classes=100,opt=''):
        super(LGCAM, self).__init__()
        self.backbone = backbone
        self.feat_dim = 768
        self.remethod = opt.remethod
        if global_dict is not None:
            self.global_dict = global_dict
            self.classifier = nn.Linear(2 * self.feat_dim, num_classes)
            self.feat_ll = FeatLL(self.feat_dim)
            self.feat_lg = FeatLG(self.feat_dim)
        else:
            self.classifier = nn.Linear(self.feat_dim, num_classes)
#         self.embed_mean = torch.zeros(2048*2).numpy()
#         self.classifier = Causal_Norm_Classifier(num_classes=num_classes,feat_dim=2048*2,num_head=opt.num_head, tau=opt.tau, alpha=opt.alpha, gamma=opt.gamma)

    def forward(self, x):
        feats = self.backbone(x)

        if hasattr(self, 'global_dict'):
            ll = self.feat_ll(feats)
            # random choice
            indices = torch.randperm(self.global_dict.size(0))[:x.size(0)].cuda()
            
            global_dict = torch.index_select(self.global_dict, 0, indices)
            lg = self.feat_lg(feats, global_dict)
            fusion_feat = torch.cat((ll, lg), dim=-1)
            out= self.classifier(fusion_feat)
#             out = self.classifier(fusion_feat)
            return fusion_feat,out
        else:
            return feats,self.classifier(feats)

        
class LGCAM_GPaco(nn.Module):
    def __init__(self, backbone, global_dict=None, num_classes=100,opt=''):
        super(LGCAM_GPaco, self).__init__()
        self.backbone = backbone
        self.feat_dim = 2048
        self.remethod = opt.remethod
        if global_dict is not None:
            self.global_dict = global_dict
            self.classifier = nn.Linear(2 * self.feat_dim, num_classes)
            self.feat_ll = FeatLL(self.feat_dim)
            self.feat_lg = FeatLG(self.feat_dim)
        else:
            self.classifier = nn.Linear(self.feat_dim, num_classes)
        self.embed_mean = torch.zeros(2048*2).numpy()

#         self.classifier = Causal_Norm_Classifier(num_classes=num_classes,feat_dim=768*2,num_head=opt.num_head, tau=opt.tau, alpha=opt.alpha, gamma=opt.gamma)

    def forward(self, im_q, im_k=None, labels=None):
        if self.train:
            features, target, logits = self.backbone(im_q, im_k, labels)
            feats = features[0:im_q.size(0)*2]
        elif self.test:
            feats = self.backbone(im_q)
        
        if hasattr(self, 'global_dict'):
            ll = self.feat_ll(feats)
            # random choice
            indices = torch.randperm(self.global_dict.size(0))[:im_q.size(0)*2].cuda()
            
            global_dict = torch.index_select(self.global_dict, 0, indices)
            lg = self.feat_lg(feats, global_dict)
            fusion_feat = torch.cat((ll, lg), dim=-1)
            out = self.classifier(fusion_feat)
#             self.embed_mean = 0.8 * self.embed_mean + fusion_feat.detach().mean(0).view(-1).cpu().numpy()

#             if not self.training:
#                 if self.remethod == 'xERM':
#                     TDE, TE = self.classifier(fusion_feat, self.embed_mean)
#                     return TDE, TE
#                 else:
#                     output,_ = self.classifier(fusion_feat, self.embed_mean)
#                     return output
#             else:
#                 _, output = self.classifier(fusion_feat, self.embed_mean)
#                 return output

            return out
        else:
            if self.train:
                features, target, logits = self.backbone(im_q, im_k, labels)
                feats = features[0:im_q.size(0)]
            elif self.test:
                feats = self.backbone(im_q)
        
            return self.classifier(feats)


class FeatLL(nn.Module):
    def __init__(self, feat_dim=768):
        super(FeatLL, self).__init__()
        self.dropout = nn.Dropout(0.15)
        self.w_q = nn.Linear(feat_dim, feat_dim, bias=False)
        self.w_kv = nn.Linear(feat_dim, feat_dim, bias=False)
        self.linear1 = nn.Linear(2 * feat_dim, feat_dim, bias=True)
        self.linear2 = nn.Linear(feat_dim, feat_dim, bias=True)

    def forward(self, x):
        x = self.dropout(x)
        q_ = self.w_q(x)
        kv_ = self.w_kv(x)
        h = torch.cat((kv_, q_ * kv_), dim=-1)
        h_ = nn.GELU()(self.linear1(h))
        attn_weights = self.linear2(h_)
        return attn_weights * x


class FeatLG(nn.Module):
    def __init__(self, feat_dim=768):
        super(FeatLG, self).__init__()
        self.dropout = nn.Dropout(0.15)
        self.w_q = nn.Linear(feat_dim, feat_dim, bias=False)
        self.w_kv = nn.Linear(feat_dim, feat_dim, bias=False)
        self.linear1 = nn.Linear(2 * feat_dim, feat_dim, bias=True)
        self.linear2 = nn.Linear(feat_dim, feat_dim, bias=True)

    def forward(self, x, global_feat):
        global_feat = self.dropout(global_feat)
        q_ = self.w_q(x)
        kv_ = self.w_kv(global_feat)
        h = torch.cat((kv_, q_ * kv_), dim=-1)
        h_ = nn.GELU()(self.linear1(h))
        attn_weights = self.linear2(h_)
        return attn_weights * global_feat


if __name__ == '__main__':
    from vit_backbone import VisionTransformer
    backbone = VisionTransformer()

    global_dict = torch.randn((512, 768))
    # random choice
    input = torch.randn((64, 3, 224, 224))

    model = LGCAM(backbone, global_dict=global_dict, num_classes=172)
    out = model(input)
    print(out.shape)
