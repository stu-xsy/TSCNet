import scipy.io as matio
import torch
import torch.utils.data
from torch import nn
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import optim
import ipdb
import torchvision.models as models
import math
from torch.optim.lr_scheduler import MultiStepLR,StepLR
from warmup_scheduler import GradualWarmupScheduler
from lgcam import *
from ccim import *
# Modelm
import timm

class VisualNet(nn.Module):
    def __init__(self, img_encoder, net_type, method, num_cls, dim_feature, opt):
        super(VisualNet, self).__init__()
        self.net_type = net_type
        self.method = method
        self.linear_align = nn.Linear(dim_feature, dim_feature)
        self.embed_mean = torch.zeros(dim_feature).numpy()
        self.remethod = opt.remethod
        self.avgpooling = nn.AdaptiveAvgPool2d((1,1))
        if self.method == 'TDE':
            self.classifier = Causal_Norm_Classifier(num_classes=num_cls,feat_dim=dim_feature,num_head=opt.num_head, tau=opt.tau, alpha=opt.alpha, gamma=opt.gamma)
        else:
            self.classifier = nn.Linear(dim_feature, num_cls)
#         confounder = np.load(f'backdoor/resnet/cifar_ratio0.02_global_dict_64.npy', allow_pickle=True)
#         confounder = torch.Tensor(confounder)
#         confounder = torch.stack([x for x in confounder])
#         prob = np.load(f'backdoor/resnet/backdoor_labels_64.npy')
#         prob = torch.from_numpy(prob).reshape(-1, 1)
#         confounder = confounder.cuda()
#         prob = prob.cuda()
#         self.confounder = confounder
#         self.prob = prob
        # utilities
        self._initialize_weights()
        self.relu = nn.LeakyReLU()
        self.encoder_v = img_encoder
        # pretrain
        if self.net_type in ['wrn','wiser']:
            self.load_weight()
    
    def forward(self,x,x_bg=None,text =None):  # x:image
        if self.net_type  == 'CLIP':
#             image_features = self.encoder_v.encode_image(x)
#             x_latent = image_features / image_features.norm(dim=1, keepdim=True)
                x_latent = self.encoder_v.encode_image(x)
        else:
            output = self.encoder_v(x)
            output = output.permute(0, 3, 1, 2)
            global_avg_pool = nn.AdaptiveAvgPool2d(1)
            pooled_output = global_avg_pool(output)  # 输出：[64, 100, 1, 1]

            # 展平输出，并使用线性层将维度变为[64, 100]
            output = pooled_output.view(pooled_output.size(0), -1)  # 输出：[64, 100]
#             import ipdb
#             ipdb.set_trace()
#             x = self.avgpooling(x)  # (1x1)
#             x = x.view(x.shape[0], x.shape[1])
            return output
        x_latent = x_latent.float()
        self.embed_mean = 0.8 * self.embed_mean + x_latent.detach().mean(0).view(-1).cpu().numpy()
        if self.method=='archd':
            output_word = self.linear_i2w(x_latent)
            return output, output_word
        
        if self.method == 'TDE':
            if not self.training:
                if self.remethod == 'xERM':
                    TDE, TE = self.classifier(x_latent, self.embed_mean)
                    return TDE, TE
                else:
                    output,_= self.classifier(x_latent, self.embed_mean)
                    return output
            else:
                _, output = self.classifier(x_latent, self.embed_mean)
                return output
        if self.method == 'Tau':
            if not self.training:
                return x_latent

        output = self.classifier(x_latent)
        return x_latent,output
    def set_method(self,name):
        self.method = name
        
    def load_weight(self):
        path_wrn_pretrain = 'model_pretrain/wide-resnet-50-2-export-5ae25d50.pth'
        pretrained_dict = torch.load(path_wrn_pretrain, map_location='cpu')
        model_dict = self.encoder_v.state_dict()
        shared_dict = {k[:7] + k[12:]: v for k, v in pretrained_dict.items() if k[:7] + k[12:] in model_dict}
        cov0_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

        model_dict.update(shared_dict)
        model_dict.update(cov0_dict)
        self.encoder_v.load_state_dict(model_dict)

    def get_feat(self):
        return

    def get_embed(self):
        return self.embed_mean

    def set_embed(self, embed_mean):
        self.embed_mean = embed_mean
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)


class VisualNet_CCIM(nn.Module):
    def __init__(self, img_encoder, net_type, method, num_cls, dim_feature, opt):
        super(VisualNet_CCIM, self).__init__()
        self.net_type = net_type
        self.method = method
        self.linear_align = nn.Linear(dim_feature, dim_feature)
        self.embed_mean = torch.zeros(dim_feature).numpy()
        self.remethod = opt.remethod


#         self.fc = nn.Linear(dim_feature, num_cls)
        self.classifier  = Causal_Norm_Classifier(num_classes=num_cls,feat_dim=dim_feature,num_head=opt.num_head, tau=opt.tau, alpha=opt.alpha, gamma=opt.gamma)
        confounder = np.load(f'backdoor/resnet/base/cifar_ratio0.02_global_dict_256.npy', allow_pickle=True)
        confounder = torch.Tensor(confounder)
        confounder = torch.stack([x for x in confounder])
        prob = np.load(f'backdoor/resnet/base/backdoor_labels_256.npy')
        prob = torch.from_numpy(prob).reshape(-1, 1)

        self.ccim = CCIM(num_joint_feature=2048, num_gz=2048, strategy='dp_cause').cuda()
        self.confounder = confounder.cuda()
        self.prob = prob.cuda()
        self._initialize_weights()
        self.relu = nn.LeakyReLU()
        self.encoder_v = img_encoder
        if self.net_type in ['wrn','wiser']:
            self.load_weight()
    
    def forward(self, x):  # x:image
        x_latent = self.encoder_v(x)
        feat = x_latent.float()
        output = self.ccim(feat, self.confounder, self.prob)
        feat = feat + output.to(torch.float32)
        self.embed_mean = 0.8 * self.embed_mean + feat.detach().mean(0).view(-1).cpu().numpy()
#         output = self.fc(feat)
#         return output
        if self.method == 'TDE':
            if not self.training:
                if self.remethod == 'xERM':
                    TDE, TE = self.classifier(feat, self.embed_mean)
                    return TDE, TE
                else:
                    output,_= self.classifier(feat, self.embed_mean)
                    return output
            else:
                _, output = self.classifier(feat, self.embed_mean)
                return output
        output = self.fc(feat)
        return feat,output
        return feat,output
#         if not self.training:
#                 if self.remethod == 'xERM':
#                     TDE, TE = self.classifier(feat, self.embed_mean)
#                     return TDE, TE
#         else:
#                 _, output = self.classifier(feat, self.embed_mean)
#                 return x_latent,output
        return output
    def set_method(self,name):
        self.method = name
        
    def load_weight(self):
        path_wrn_pretrain = 'model_pretrain/wide-resnet-50-2-export-5ae25d50.pth'
        pretrained_dict = torch.load(path_wrn_pretrain, map_location='cpu')
        model_dict = self.encoder_v.state_dict()
        shared_dict = {k[:7] + k[12:]: v for k, v in pretrained_dict.items() if k[:7] + k[12:] in model_dict}
        cov0_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

        model_dict.update(shared_dict)
        model_dict.update(cov0_dict)
        self.encoder_v.load_state_dict(model_dict)

    def get_feat(self):
        return

    def get_embed(self):
        return self.embed_mean

    def set_embed(self, embed_mean):
        self.embed_mean = embed_mean
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                
                
class VisualNetH2T(nn.Module):
    def __init__(self, img_encoder, net_type, method, num_cls, dim_feature, opt):
        super(VisualNetH2T, self).__init__()
        self.net_type = net_type
        self.method = method
        self.linear_align = nn.Linear(dim_feature, dim_feature)
        self.embed_mean = torch.zeros(dim_feature).numpy()
        self.remethod = opt.remethod

        self.classifier = nn.Linear(dim_feature, num_cls)
        self.avgpooling = nn.AdaptiveAvgPool2d((1,1))
        # utilities
        self._initialize_weights()
        self.relu = nn.LeakyReLU()
        self.encoder_v = img_encoder
        # pretrain
        if self.net_type in ['wrn','wiser']:
            self.load_weight()
    
    def forward(self, x1,x2):  # x:image

        feature1 = self.encoder_v(x1)
        feature2 = self.encoder_v(x2)
        feature = self.H2T(feature1,feature2)
        x = self.avgpooling(feature)  # (1x1)
        x = x.view(x.shape[0], x.shape[1])
        output = self.classifier(x)
        return output
    def H2T(self, x1, x2, rho=0.3):
        if len(x1.shape) == 4:
            fea_num = x1.shape[1]
            index = torch.randperm(fea_num).cuda()
            slt_num = int(rho * fea_num)
            index = index[:slt_num]
            
            # 使用克隆避免原地操作
            x1_clone = x1.clone()
            x1_clone[:, index, :, :] = x2[:, index, :, :]
            x1 = x1_clone
        else:
            for i in range(len(x1)):
                fea_num = x1[i].shape[1]
                index = torch.randperm(fea_num).cuda()
                slt_num = int(rho * fea_num)
                index = index[:slt_num]
                
                # 使用克隆避免原地操作
                x1_clone = x1[i].clone()
                x1_clone[:, index, :, :] = x2[i][:, index, :, :]
                x1[i] = x1_clone
        return x1
    def set_method(self,name):
        self.method = name
        
    def load_weight(self):
        path_wrn_pretrain = 'model_pretrain/wide-resnet-50-2-export-5ae25d50.pth'
        pretrained_dict = torch.load(path_wrn_pretrain, map_location='cpu')
        model_dict = self.encoder_v.state_dict()
        shared_dict = {k[:7] + k[12:]: v for k, v in pretrained_dict.items() if k[:7] + k[12:] in model_dict}
        cov0_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

        model_dict.update(shared_dict)
        model_dict.update(cov0_dict)
        self.encoder_v.load_state_dict(model_dict)

    def get_feat(self):
        return

    def get_embed(self):
        return self.embed_mean

    def set_embed(self, embed_mean):
        self.embed_mean = embed_mean
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)       
    
class FDC5_resnet(nn.Module):
    def __init__(self, hidden_dim=2048, cat_num=65, drop_xp=False, drop_xp_ratio=0.5, middle_hidden=384):
        super(FDC5_resnet, self).__init__()
        self.drop_xp = drop_xp

        self.conv1 = nn.Conv2d(3, 32, kernel_size=2, stride=4, padding=0,
                               bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=1, stride=2, padding=0,
                               bias=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=1, stride=2, padding=0,
                               bias=False)
        # self.conv4 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0,
        #                        bias=False)
        # self.conv5 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0,
        #                        bias=False)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        # self.bn4 = nn.BatchNorm2d(256)
        # self.bn5 = nn.BatchNorm2d(512)
        # self.fc_z2 = nn.Linear(sub_in_dim, 64)
        self.dropout = nn.Dropout(p=drop_xp_ratio)
        # self.dropout2 = nn.Dropout(p=0.5)

        self.fc_c1 = nn.Linear(hidden_dim + 128, middle_hidden)
        # self.bn = nn.BatchNorm1d(1024)
        self.fc_c2 = nn.Linear(middle_hidden, cat_num)

    def forward(self, z1, z2, test=True, random_detach=False, noise_inside=False, scale=0.01):
        z2 = F.relu(self.bn1(self.conv1(z2)))
        z2 = F.relu(self.bn2(self.conv2(z2)))
        # z2 = self.dropout(z2)
        z2 = F.relu(self.bn3(self.conv3(z2)))
        # z2 = F.relu(self.bn4(self.conv4(z2)))
        # z2 = F.relu(self.bn5(self.conv5(z2)))
        z2 = F.avg_pool2d(z2, 14)
        z2 = z2.reshape(z2.size(0), -1)
        if random_detach:
            z2 = z2.detach()
        if self.drop_xp:
            z2 = self.dropout(z2)
        # z2 = z2.detach()

        if test:
            # import pdb;
            # pdb.set_trace()
            # Pairwise combination betwen feature from Image and feature from random image.
            # with larger batch size for z2, you get better randomized

            bs1=z1.size(0)
            bs2=z2.size(0)
            zlen_1=z1.size(1)
            z1 = z1.unsqueeze(1)
            z1 = z1.repeat(1, bs2, 1)

            if noise_inside:
                z1 = z1 + torch.normal(mean=torch.zeros((bs1, bs2, zlen_1)), std=torch.ones((bs1, bs2, zlen_1))).cuda()*scale
            z2 = z2.unsqueeze(0)
            z2 = z2.repeat(bs1, 1, 1)
            # z2 = torch.zeros_like(z2).cuda()
            # print(z1.shape, z2.shape)
            hh = torch.cat((z1, z2), dim=2)

            hh = hh.view(bs1*bs2, -1)
            # import pdb;
            # pdb.set_trace()
            # print('h', hh.size())
            hh = F.relu(self.fc_c1(hh))
            out = self.fc_c2(hh)
            out = out.view(bs1, bs2, -1)

            return torch.sum(out, dim=1)
        else:
            # z2 = torch.zeros_like(z2).cuda()
            hh = torch.cat((z1, z2), dim=1)

            hh = F.relu(self.fc_c1(hh))
            out = self.fc_c2(hh)
            return out

class FDC5_vit(nn.Module):
    def __init__(self, hidden_dim=2048, cat_num=65, drop_xp=False, drop_xp_ratio=0.5, encoder_v=None):
        super(FDC5_vit, self).__init__()
        sub_in_dim=64
        print("FDC 5")
        self.encoder = encoder_v
        self.fc_c2 = nn.Linear(hidden_dim + 64, cat_num)
        self.fc_c1 = nn.Linear(hidden_dim, 64)  # 添加一个全连接层
        self.relu = nn.ReLU()

    def forward(self, z1, z2, test=True, random_detach=False, noise_inside=False, scale=0.01):

        z2,_ = self.encoder(z2)
        z2 = F.relu(self.fc_c1(z2))

        if test:

            bs1=z1.size(0)
            bs2=z2.size(0)
            zlen_1=z1.size(1)
            z1 = z1.unsqueeze(1)
            z1 = z1.repeat(1, bs2, 1)

            if noise_inside:
                z1 = z1 + torch.normal(mean=torch.zeros((bs1, bs2, zlen_1)), std=torch.ones((bs1, bs2, zlen_1))).cuda()*scale
            z2 = z2.unsqueeze(0)
            z2 = z2.repeat(bs1, 1, 1)
            hh = torch.cat((z1, z2), dim=2)
            hh = hh.view(bs1*bs2, -1)

            out = self.fc_c2(hh)
            out = out.view(bs1, bs2, -1)

            return torch.sum(out, dim=1)
        else:

            hh = torch.cat((z1, z2), dim=1)
            out = self.fc_c2(hh)
            return out
        
class Causal_Norm_Classifier_adaptive(nn.Module):   
    def __init__(self, num_classes=100, feat_dim=2048, use_effect=True, num_head=2, tau=16.0, alpha=0.15, gamma=0.03125, *args):
        super(Causal_Norm_Classifier_adaptive, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(num_classes, feat_dim).cuda(), requires_grad=True)
#         self.scale_fc = nn.Linear(feat_dim, 1)
#         self.norm_scale_fc = nn.Linear(feat_dim, 1)
        self.scale = nn.Parameter(torch.Tensor([tau/num_head]).cuda(), requires_grad=True)
        self.norm_scale = gamma
        self.alpha = alpha            # 3.0
        
        self.num_head = num_head
        self.head_dim = feat_dim // num_head
        self.use_effect = use_effect
        self.reset_parameters(self.weight)
        self.relu = nn.ReLU(inplace=True)
        nn.init.constant_(self.scale, 8)
    def reset_parameters(self, weight):
        stdv = 1. / math.sqrt(weight.size(1))
        weight.data.uniform_(-stdv, stdv)

    def forward(self, x, embed):
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

    
class Causal_Norm_Classifier(nn.Module):   
    def __init__(self, num_classes=100, feat_dim=2048, use_effect=True, num_head=2, tau=16.0, alpha=0.15, gamma=0.03125, *args):
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

def flatten(t):
    return t.reshape(t.shape[0], -1)  
    
class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=768, K=65536, m=0.999, T=0.2, mlp=True, feat_dim=768, num_classes=100):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder
        self.linear = nn.Linear(feat_dim, num_classes)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(inplace=True), nn.Linear(dim_mlp, dim_mlp), nn.ReLU(inplace=True), self.encoder_q.fc)


        # create the queue
        self.register_buffer("queue", torch.randn(K, dim))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_l", torch.randint(0, num_classes, (K,)))

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # cross_entropy
        self.layer = -2 
        self.feat_after_avg_q = None
        self._register_hook()
        self.normalize = False 

    
    def _find_layer(self, module):
        if type(self.layer) == str:
            modules = dict([*module.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*module.children()]

            return children[self.layer]
        return None

    def _hook_q(self, _, __, output):

        self.feat_after_avg_q = flatten(output)
        if self.normalize:
            self.feat_after_avg_q = nn.functional.normalize(self.feat_after_avg_q, dim=1)

    def _register_hook(self):
        layer_q = self._find_layer(self.encoder_q)

        assert layer_q is not None, f'hidden layer ({self.layer}) not found'
        handle = layer_q.register_forward_hook(self._hook_q)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels):
        # gather keys before updating queue
        keys = keys
        labels = labels

        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
 
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[ptr:ptr + batch_size,:] = keys
        self.queue_l[ptr:ptr + batch_size] = labels
        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr[0] = ptr


    def _train(self, im_q, im_k, labels):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        
        q = nn.functional.normalize(q, dim=1)
        import ipdb
        ipdb.set_trace()
        logits_q = self.linear(self.feat_after_avg_q)

        # compute key features
        k = self.encoder_q(im_k)  # keys: NxC
        k = nn.functional.normalize(k, dim=1)
        logits_k = self.linear(self.feat_after_avg_q)

        # compute logits
        features = torch.cat((q, k, self.queue.clone().detach()), dim=0)
        target = torch.cat((labels, labels, self.queue_l.clone().detach()), dim=0)
        self._dequeue_and_enqueue(k, labels)

        # compute logits 
        logits = torch.cat((logits_q, logits_k), dim=0)

        return features, target, logits

    def _inference(self, image):
        q = self.encoder_q(image)
        encoder_q_logits = self.linear(self.feat_after_avg_q)
        return encoder_q_logits

    def forward(self, im_q, im_k=None, labels=None):
        if self.training:
            return self._train(im_q, im_k, labels) 
        else:
            return self._inference(im_q)


class MoCo_Causal(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=768, K=65536, m=0.999, T=0.2, mlp=True, feat_dim=768, num_classes=100, opt=None):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo_Causal, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder
        self.linear = Causal_Norm_Classifier(num_classes=num_classes,feat_dim=feat_dim,num_head=opt.num_head, tau=opt.tau, alpha=opt.alpha, gamma=opt.gamma)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(inplace=True), nn.Linear(dim_mlp, dim_mlp), nn.ReLU(inplace=True), self.encoder_q.fc)


        # create the queue
        self.register_buffer("queue", torch.randn(K, dim))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_l", torch.randint(0, num_classes, (K,)))

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # cross_entropy
        self.layer = -16
        self.feat_after_avg_q = None
        self._register_hook()
        self.normalize = False 

    
    def _find_layer(self, module):
        if type(self.layer) == str:
            modules = dict([*module.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*module.children()]


            return children[self.layer]
        return None

    def _hook_q(self, _, __, output):

        self.feat_after_avg_q = flatten(output)
        if self.normalize:
            self.feat_after_avg_q = nn.functional.normalize(self.feat_after_avg_q, dim=1)

    def _register_hook(self):
        layer_q = self._find_layer(self.encoder_q)

        assert layer_q is not None, f'hidden layer ({self.layer}) not found'
        handle = layer_q.register_forward_hook(self._hook_q)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels):
        # gather keys before updating queue
        keys = keys
        labels = labels

        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
 
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[ptr:ptr + batch_size,:] = keys
        self.queue_l[ptr:ptr + batch_size] = labels
        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr[0] = ptr


    def _train(self, im_q, im_k, labels):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        
        q = nn.functional.normalize(q, dim=1)
 
        logits_q = self.linear(self.feat_after_avg_q)
        # compute key features
        k = self.encoder_q(im_k)  # keys: NxC
        k = nn.functional.normalize(k, dim=1)
        logits_k = self.linear(self.feat_after_avg_q)

        # compute logits
        features = torch.cat((q, k, self.queue.clone().detach()), dim=0)
        target = torch.cat((labels, labels, self.queue_l.clone().detach()), dim=0)
        self._dequeue_and_enqueue(k, labels)

        # compute logits 
        logits = torch.cat((logits_q, logits_k), dim=0)

        return features, target, logits

    def _inference(self, image):
        q = self.encoder_q(image)
        encoder_q_logits = self.linear(self.feat_after_avg_q)
        return encoder_q_logits

    def forward(self, im_q, im_k=None, labels=None):
        if self.training:
            return self._train(im_q, im_k, labels) 
        else:
            return self._inference(im_q)

        
        
# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output           
class SemanticNet(nn.Module):
    def __init__(self, word_encoder, dim_latent, num_cls):
        super(SemanticNet, self).__init__()

        # ingre channel network
        self.encoder_t = word_encoder        
        self.classifier = nn.Linear(dim_latent, num_cls)
        self._initialize_weights()
    def forward(self, x):
        x_latent = self.encoder_t(x)
        output = self.classifier(x_latent)
        return output
       
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)


class V2S_Align(nn.Module):
    def __init__(self, word_encoder, img_encoder, dim_latent, dim_feature, dim_align, num_cls):
        super(V2S_Align, self).__init__()
        # ingre channel network
        self.align_s = nn.Linear(dim_latent, dim_align)
        self.align_v = nn.Linear(dim_feature, dim_align)
        self.classifier = nn.Linear(dim_align, num_cls)
        self._initialize_weights()
        self.encoder_s = word_encoder
        self.encoder_v = img_encoder
    def forward(self, img, word):  #y:ingredients
        y_s = self.encoder_s(word)
        y_v,_ = self.encoder_v(img)
        y_s_align = self.align_s(y_s)
        y_v_align = self.align_v(y_v)
        pre_s = self.classifier(y_s_align)
        pre_v = self.classifier(y_v_align)
        return y_s_align, y_v_align, pre_s, pre_v
       
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)

class V2S_ATNet(nn.Module):
    def __init__(self, word_encoder, img_encoder, dim_latent, dim_feature, pht_partial, num_cls):
        super(V2S_ATNet, self).__init__()
        # ingre channel network
        self.relu = nn.LeakyReLU()
        self.pht_partial = pht_partial
        self.dim_partial_v = int(self.pht_partial * dim_feature)
        self.dim_partial_s = int(self.pht_partial * dim_latent)
        self.dim_align = self.dim_partial_s if self.dim_partial_s<self.dim_partial_v else self.dim_partial_v
        self.align_s = nn.Linear(self.dim_partial_s, self.dim_align)
        self.align_v = nn.Linear(self.dim_partial_v, self.dim_align)
#         self.align_s_2 = nn.Linear(self.dim_align, self.dim_align)
#         self.align_v_2 = nn.Linear(self.dim_align, self.dim_align)
        self.classifier = nn.Linear(self.dim_align, num_cls)
        self._initialize_weights()
        
        self.encoder_s = word_encoder
        self.encoder_v = img_encoder
        
    def forward(self, img, word):  #y:ingredients
        y_s = self.encoder_s(word)
        y_v = self.encoder_v(img)

        y_s_partial = y_s[:,:self.dim_partial_s]
        y_v_partial = y_v[:,:self.dim_partial_v]
        y_s_align = self.align_s(y_s_partial)
        y_v_align = self.align_v(y_v_partial)
        pre_s = self.classifier(y_s_align)
        pre_v = self.classifier(y_v_align)
        return y_s_align, y_v_align, pre_s, pre_v
    
    def forward_kl_ln_l2(self, img, word):
        y_s = self.encoder_s(word)
        y_v = self.encoder_v(img)

        y_s_partial = self.relu(y_s[:,:self.dim_partial_s])
        y_v_partial = self.relu(y_v[:,:self.dim_partial_v])
        # align stage 1: 
        y_s_align = self.relu(self.align_s(y_s_partial))
        y_v_align = self.relu(self.align_v(y_v_partial))
        # align stage 2:
        y_v_align_2 = self.align_v_2(y_v_align)
        y_s_align_2 = self.align_s_2(y_s_align)

        pre_s = self.classifier(y_s_align_2)
        pre_v = self.classifier(y_v_align_2)
        return y_s_align, y_v_align, y_s_align_2, y_v_align_2, pre_s, pre_v
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)

class S2S_recon(nn.Module):
    def __init__(self, net_s, word_encoder, word_decoder, dim_latent, num_word):
        super(S2S_recon, self).__init__()
        # ingre channel network
        self.net_s = net_s
        self.decoder_s = word_decoder
        self.encoder_s = word_encoder
        self.classifier = nn.Linear(dim_latent, num_word)
        
    def forward(self, word, max_seq):  #y:ingredients
        if self.net_s=='nn':
            latent_vector = self.encoder_s(word)
            word_recon = self.decoder_s(latent_vector)
            return word_recon
        elif self.net_s=='gru':
            latent_vector = self.encoder_s(word)
            word_recon, label = self.decoder_s(latent_vector, word, max_seq)
            word_recon = self.classifier(word_recon)
            return word_recon, label
       
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                
class Img2word(nn.Module):
    def __init__(self, encoder_v, net_s, dim_feature, dim_latent, dim_align, num_cls, type_net_s):
        super(Img2word, self).__init__()        
        # networks for partial heterogeneous transfer
        self.relu = nn.LeakyReLU()
        self.align_v1 = nn.Linear(dim_feature,dim_latent)
        self.classifier = nn.Linear(dim_latent,num_cls)
        self._initialize_weights()
        self.net_s = type_net_s
        [encoder_s, decoder_s] = net_s
        # network for image channel
        self.encoder_v = encoder_v
        self.decoder_s = decoder_s
        self.modality = 'mm'
        
        # network for word channel
        self.encoder_s = encoder_s

    def forward(self, img, words, word_vector, max_seq):
        
        # feature extraction
        feature_v = self.encoder_v(img)
        feature_v2s = self.relu(self.align_v1(feature_v))
        if self.net_s=='nn':
            feature_s = self.encoder_s(words)
        elif self.net_s=='gru':
            feature_s = self.encoder_s(word_vector)
        word_embed, label_recon = self.decoder_s(feature_v2s, word_vector, max_seq)
        word_recon = self.classifier(word_embed)
        return feature_v2s, feature_s, word_recon, label_recon
    
    def forward_gru_cis(self, img, max_seq):
        # feature extraction
        feature_v = self.encoder_v(img)
        feature_v2s = self.relu(self.align_v1(feature_v))
#         feature_v2s = self.align_v2(feature_v2s)
        word_embed = self.decoder_s.forward_cis(feature_v2s, max_seq)
        word_recon = self.classifier(word_embed)
        word_recon = word_recon.reshape(img.shape[0], max_seq, -1)
        if self.modality == 'mm':
            return feature_v,word_recon
        else:
            return word_recon
    
    def forward_wide(self, img, words):
        
        # feature extraction
        feature_v = self.encoder_v(img)
        feature_v2s = self.relu(self.align_v1(feature_v))
        feature_v2s = self.align_v2(feature_v2s)

        feature_s = self.encoder_s(words)
        word_recon = self.decoder_s(feature_v2s)
        
        return feature_v2s, feature_s, word_recon
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                
                
class V2S_CIS(nn.Module):
    def __init__(self, model_img2word, dim_v, dim_latent, dim_embed, num_cls, num_words, method_adj, net_s):
        super(V2S_CIS, self).__init__()
        self.net_s = net_s
        self.num_words = num_words
        self.embedding = nn.Embedding(num_words+1, dim_latent)
        self.method_adj = method_adj
        if self.method_adj.startswith('sl'):
            self.weight_sl = nn.Linear(num_words, num_cls)
        self.rate_drop = 0.5
        
        self.linear_gcn = nn.Linear(dim_embed*self.num_words, dim_embed)
        self.linear_gcn_mm = nn.Linear((dim_embed+dim_v)*self.num_words, dim_embed)
        self.linear_weight = nn.Linear(dim_latent,dim_embed)
        self.linear_weight_mm = nn.Linear(dim_latent+dim_v,dim_embed)
        self.classifier_gcn = nn.Linear(dim_embed, num_cls)
        self.relu = nn.LeakyReLU()
        self._initialize_weights()
        self.softmax = nn.Softmax(dim=1)

        self.img2word = model_img2word
        
    def forward_gru(self, img, words, indexVectors, k, max_seq):  #y:ingredients
        word_recon = self.img2word.forward_gru_cis(img, max_seq) # (64,15,353)
        
        multi_hot, adj = self.compute_adj(img.shape[0], word_recon, self.num_words, k)
        
        embed_pre_words = self.embedding(multi_hot) # (64, 353, dim_latent)
        # gcn
        support = self.linear_weight(embed_pre_words) # (64, 353, dim_embed)
        feature_gcn = torch.bmm(adj, support) # (64, 353, 353) x (64, 353, dim_embed)
        feature_gcn = F.dropout(feature_gcn, self.rate_drop, training=self.training)
        feature_gcn = self.relu(self.linear_gcn(feature_gcn.reshape(feature_gcn.shape[0],-1))) #

        output = self.classifier_gcn(feature_gcn)
        
        
        return feature_gcn, output
    
    def forward_gru_mm(self, img, words, indexVectors, k, max_seq):  #y:ingredients
        feature_v, word_recon = self.img2word.forward_gru_cis(img, max_seq) # (64,15,353)
        
        multi_hot, adj = self.compute_adj(img.shape[0], word_recon, self.num_words, k)
        
        embed_pre_words = self.embedding(multi_hot) # (64, 353, dim_latent)
        feature_v_rep = feature_v.unsqueeze(1).repeat(1,self.num_words,1)
        embed_mm = torch.cat((embed_pre_words,feature_v_rep),2)
        # gcn
        support = self.linear_weight_mm(embed_mm) # (64, 353, dim_embed)
        feature_gcn = torch.bmm(adj, support) # (64, 353, 353) x (64, 353, dim_embed)
        feature_gcn = F.dropout(feature_gcn, self.rate_drop, training=self.training)
        feature_gcn = self.relu(self.linear_gcn(feature_gcn.reshape(feature_gcn.shape[0],-1))) #

        output = self.classifier_gcn(feature_gcn)
        
        
        return feature_gcn, output
    
    def forward_gru_sl(self, img, words, k, max_seq, label, cls_word_prob, beta_pri,metrix_sl):  #y:ingredients
        batch_size = img.shape[0]
        word_recon = self.img2word.forward_gru_cis(img, max_seq) # (64,15,353)
        word_recon_topk = word_recon.argmax(2)
        multi_hot, adj = self.compute_adj(batch_size,word_recon_topk, word_recon, self.num_words, k)
        embed_pre_words = self.embedding(multi_hot)
        # class-aware self-learning
        weight_cls = self.softmax(self.weight_sl(multi_hot.float()))
        adj_pri = torch.zeros((self.num_words, self.num_words),requires_grad=True).repeat(batch_size, 1, 1).cuda() #(64,353,353)
        weight_pri = torch.zeros((1, self.num_words),requires_grad=True).repeat(batch_size, 1, 1).cuda() #(64,353,353)
        word_prob = torch.zeros((1, self.num_words),requires_grad=True).repeat(batch_size, 1, 1).cuda() #(64,353,353)
        
        for cur in range(batch_size):
            cur_pri = weight_cls[cur].unsqueeze(0).mm(metrix_sl)
            weight_pri[cur] = cur_pri
            cur_pri = cur_pri.t().mm(cur_pri)
            adj_pri[cur] = cur_pri.unsqueeze(0)
            word_prob[cur] = cls_word_prob[label[cur]]
        adj = (1-beta_pri)*adj + beta_pri * adj_pri
        
        # gcn
        support = self.linear_weight(embed_pre_words)
        feature_gcn = self.relu(torch.bmm(adj, support))
        feature_gcn = F.dropout(feature_gcn, self.rate_drop, training=self.training)
        feature_gcn = self.relu(self.linear_gcn(feature_gcn.reshape(feature_gcn.shape[0],-1))) #

        output = self.classifier_gcn(feature_gcn)
        return output, feature_gcn, weight_pri, adj_pri, word_prob
    
    def forward_nn_sl(self, img, words, k, label, cls_word_prob, beta_pri, metrix_sl):  #y:ingredients

        batch_size = img.shape[0]
        _,_,word_recon = self.img2word.forward_wide(img, words)
        word_recon_topk = word_recon.topk(k)[1]
        multi_hot, adj = self.compute_adj(img.shape[0],word_recon_topk, word_recon, self.num_words, k)
        embed_pre_words = self.embedding(multi_hot)
        # class-aware self-learning
        weight_cls = self.softmax(self.weight_sl(multi_hot.float()))
        adj_pri = torch.zeros((self.num_words, self.num_words),requires_grad=True).repeat(batch_size, 1, 1).cuda() #(64,353,353)
        weight_pri = torch.zeros((1, self.num_words),requires_grad=True).repeat(batch_size, 1, 1).cuda() #(64,353,353)
        word_prob = torch.zeros((1, self.num_words),requires_grad=True).repeat(batch_size, 1, 1).cuda() #(64,353,353)
        
        for cur in range(batch_size):
            cur_pri = weight_cls[cur].unsqueeze(0).mm(metrix_sl)
            weight_pri[cur] = cur_pri
            cur_pri = cur_pri.t().mm(cur_pri)
            adj_pri[cur] = cur_pri.unsqueeze(0)
            label_pos = np.where(label[cur].cpu().numpy()==1)[0]
            if label_pos.shape[0] > 1:
                word_prob[cur] = torch.sum(cls_word_prob[label_pos],dim=0)/label_pos.shape[0]
            else:
                word_prob[cur] = cls_word_prob[label_pos]
        adj = (1-beta_pri)*adj + beta_pri * adj_pri
        
        
        # gcn
        support = self.linear_weight(embed_pre_words)
        feature_gcn = self.relu(torch.bmm(adj, support))
        feature_gcn = F.dropout(feature_gcn, self.rate_drop, training=self.training)
        feature_gcn = self.relu(self.linear_gcn(feature_gcn.reshape(feature_gcn.shape[0],-1))) #

        output = self.classifier_gcn(feature_gcn)
        
        
        return output, feature_gcn, weight_pri, adj_pri, word_prob
    
    def forward_nn(self, img, words, k):  #y:ingredients
        _,_,word_recon = self.img2word.forward_wide(img, words)
        word_recon_topk = word_recon.topk(k)[1]
        multi_hot, adj = self.compute_adj(img.shape[0],word_recon_topk, word_recon, self.num_words, k)
        embed_pre_words = self.embedding(multi_hot)
        # gcn
        support = self.linear_weight(embed_pre_words)
        feature_gcn = self.relu(torch.bmm(adj, support))
        feature_gcn = F.dropout(feature_gcn, self.rate_drop, training=self.training)
        feature_gcn = self.relu(self.linear_gcn(feature_gcn.reshape(feature_gcn.shape[0],-1))) #

        output = feature_gcn, self.classifier_gcn(feature_gcn)
        
        
        return output
    
    
    
    def compute_adj(self, batch_size, recon_words, num_words, topk):
        pre_words_top = torch.topk(recon_words,1)[1].reshape(batch_size,-1)[:,:topk]
        # adj
        multi_hot = torch.zeros([num_words]).repeat(batch_size, 1).long()
        adj = torch.eye(num_words).repeat(batch_size, 1, 1) #(64,353,353)
        
        # compute where to fill
        for cur in range(batch_size):
            # adj
            pre_cur = torch.unique(pre_words_top[cur]).cpu()
            pos_cur = np.array(np.meshgrid(pre_cur, pre_cur)).T.reshape(-1, 2)
            adj[cur][pos_cur[:,0],pos_cur[:,1]] = 1

            # multi_hot
            multi_hot[cur][pre_cur]=1
            
        multi_hot = multi_hot.cuda()
        adj = adj.cuda()
        return multi_hot, adj
    
    def minmax_oprtator(self, data):
        min = torch.min(data, dim=2)[0].unsqueeze(2).expand(data.shape[0],data.shape[1],data.shape[2]) 
        max = torch.max(data, dim=2)[0].unsqueeze(2).expand(data.shape[0],data.shape[1],data.shape[2])  
        return (data - min)/(max-min)
    
    def get_topk(self, predicts_t, k):
        # min-max
        predicts_t_minmax = self.minmax_oprtator(predicts_t)
        predicts_max = torch.max(predicts_t_minmax, dim=1)[0] #(64,446)
        predicts_topk_id = torch.sort(predicts_max, dim=1, descending=True)[1][:,:k] #(64,25)
        return predicts_topk_id  
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)                

class V2S_CMI(nn.Module):
    def __init__(self, model_img2word, dim_latent, num_cls, num_words, method_cmi, topk):
        super(V2S_CMI, self).__init__()
        self.method_cmi = method_cmi
        self.num_words = num_words
        self.topk = topk
        self.embedding = nn.Embedding(num_words+1, dim_latent)
        self.linear_semantic = nn.Linear(self.topk, 1)
        
        self.classifier_cmi = nn.Linear(dim_latent, num_cls)
        self.relu = nn.LeakyReLU()
        self._initialize_weights()

        self.img2word = model_img2word
        
    def forward_nn(self, img, words, k):  #y:ingredients
        _,_,word_recon = self.img2word.forward_wide(img, words)
        word_recon_topk = word_recon.topk(k)[1]
        embed_pre_words = self.embedding(word_recon_topk)
        # cmi
        if self.method_cmi=='linear':
            embed_semantic = self.linear_semantic(embed_pre_words.transpose(1,2)).squeeze(2)
        elif self.method_cmi=='mean':
            embed_semantic = torch.mean(embed_pre_words, dim=1).squeeze(1)

        output = self.classifier_cmi(embed_semantic)
        return embed_semantic, output
    
    def forward_gru(self, img, words, k, max_seq):  #y:ingredients
        word_recon = self.img2word.forward_gru_cis(img, k) # (64,15,353)
        word_recon_topk = self.get_topk(word_recon,k)
        embed_pre_words = self.embedding(word_recon_topk)
        # cmi
        if self.method_cmi=='linear':
            embed_semantic = self.linear_semantic(embed_pre_words.transpose(1,2)).squeeze(2)
        elif self.method_cmi=='mean':
            embed_semantic = torch.mean(embed_pre_words, dim=1).squeeze(1)


        output = self.classifier_cmi(embed_semantic)
        
        
        return embed_semantic, output
    
    # get top-k index    
    def get_topk(self, predicts_t, k):
        # min-max
        predicts_t_minmax = self.minmax_oprtator(predicts_t)
        predicts_max = torch.max(predicts_t_minmax, dim=1)[0] #(64,446)
        predicts_topk_id = torch.sort(predicts_max, dim=1, descending=True)[1][:,:k] #(64,25)
        return predicts_topk_id
    
    def minmax_oprtator(self, data):
        min = torch.min(data, dim=2)[0].unsqueeze(2).expand(data.shape[0],data.shape[1],data.shape[2]) 
        max = torch.max(data, dim=2)[0].unsqueeze(2).expand(data.shape[0],data.shape[1],data.shape[2])  
        return (data - min)/(max-min)
    
    def compute_multihot(self, batch_size, index_vector, recon_words, num_words, topk):
        pre_words_top = torch.topk(recon_words,topk)[1].reshape(batch_size,-1)
        multi_hot = torch.tensor(torch.zeros([num_words]).repeat(batch_size, 1),dtype=torch.long) #(64,353)
        
        # compute where to fill
        for cur in range(batch_size):
            pre_cur = torch.unique(pre_words_top[cur])
            pos_cur = np.array(np.meshgrid(pre_cur, pre_cur)).T.reshape(-1, 2)
            # multi_hot
            multi_hot[cur][index_vector[cur]]=1
            
        multi_hot = multi_hot.cuda()
        return multi_hot
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)

class V2S2C(nn.Module):
    def __init__(self, model_img2word, dim_latent, num_cls, num_words):
        super(V2S2C, self).__init__()
        self.num_words = num_words
        self.embedding = nn.Embedding(self.num_words+1, dim_latent)

        self.classifier_multi_hot = nn.Linear(num_words, num_cls)
        self.classifier_embedding = nn.Linear(dim_latent, num_cls)
        self.relu = nn.LeakyReLU()
        self._initialize_weights()
        
        self.img2word = model_img2word
        

    def forward_multi_hot(self, img, k, max_seq):  #y:ingredients
        batch_size = img.shape[0]
        word_recon = self.img2word.forward_gru_cis(img, max_seq) # (64,15,353)
        word_recon_topk = self.get_topk(word_recon,k)
        multi_hot = self.compute_multi_hot(batch_size, word_recon_topk, self.num_words)
        multi_hot = multi_hot.to_torch().cuda()
        output = self.classifier_multi_hot(multi_hot)
        return output
    
    def forward_embedding(self, img, k, max_seq):  #y:ingredients
        word_recon = self.img2word.forward_gru_cis(img, max_seq) # (64,15,353)
        word_recon_topk = word_recon.argmax(2)
        embed_pre_words = self.embedding(word_recon_topk)
        # cmi
        embed_semantic = torch.mean(embed_pre_words, dim=1).squeeze(1)
        output = self.classifier_embedding(embed_semantic)
        return output
    
    def minmax_oprtator(self, data):
        min = torch.min(data, dim=2)[0].unsqueeze(2).expand(data.shape[0],data.shape[1],data.shape[2]) 
        max = torch.max(data, dim=2)[0].unsqueeze(2).expand(data.shape[0],data.shape[1],data.shape[2])  
        return (data - min)/(max-min)
    
    def get_topk(self, predicts_t, k):
        # min-max
        predicts_t_minmax = self.minmax_oprtator(predicts_t)
        predicts_max = torch.max(predicts_t_minmax, dim=1)[0] #(64,446)
        predicts_topk_id = torch.sort(predicts_max, dim=1, descending=True)[1][:,:k] #(64,25)
        return predicts_topk_id  
    

#     def forward_multi_hot(self, img, k, max_seq):  #y:ingredients
#         batch_size = img.shape[0]
#         import ipdb;ipdb.set_trace()
#         word_recon = self.img2word.forward_gru_cis(img, max_seq) # (64,15,353)
#         word_recon_topk = word_recon.argmax(2)
#         multi_hot = self.compute_multi_hot(batch_size, word_recon_topk, word_recon, self.num_words, k)
        
#         output = self.classifier_multi_hot(multi_hot.float())
        
#         return output
    
#     def compute_multi_hot(self, batch_size, index_vector, recon_words, num_words, topk):
#         pre_words_top = torch.topk(recon_words,topk)[1].reshape(batch_size,-1)
#         multi_hot = torch.zeros([num_words]).repeat(batch_size, 1).long()
        
#         # compute where to fill
#         for cur in range(batch_size):
#             # multi_hot
#             multi_hot[cur][index_vector[cur]]=1
            
#         multi_hot = multi_hot.cuda()
#         return multi_hot
        
    # get top-k index    
 
                
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
    
class FeatureFusion(nn.Module):
    def __init__(self, latent_len, dim_align, num_class, method_fusion='add', beta=0.5):
        super(FeatureFusion, self).__init__()
        self.method_fusion = method_fusion
        self.beta = beta
        self.classifier_align = nn.Linear(dim_align, num_class)
        self.classifier_cat = nn.Linear(dim_align * 2, num_class)
        self.classifier_latent = nn.Linear(latent_len, num_class)

        self.linearl2a = nn.Linear(latent_len, dim_align)
        self.lineara2a = nn.Linear(dim_align, dim_align)

        self.linearcat = nn.Linear(dim_align * 2, dim_align)

        self.relu = nn.LeakyReLU()
        self._initialize_weights()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, feature_visual, feature_semantic):
        if self.method_fusion == 'add':
            feature_semantic = self.linearl2a(feature_semantic)
            feature_fusion = ((1.0 - self.beta) * feature_visual) + (self.beta * feature_semantic)
            output = self.classifier_align(feature_fusion)
        elif self.method_fusion == 'cat':
            feature_semantic = self.linearl2a(feature_semantic)
            feature_fusion = torch.cat((feature_visual, feature_semantic), 1)
            output = self.classifier_cat(feature_fusion)
        elif self.method_fusion == 'min':
            feature_semantic = self.linearl2a(feature_semantic)
            feature_fusion = torch.min(feature_semantic, feature_visual)
            output = self.classifier_align(feature_fusion)
        elif self.method_fusion == 'max':
            feature_semantic = self.linearl2a(feature_semantic)
            feature_fusion = torch.max(feature_semantic, feature_visual)
            output = self.classifier_align(feature_fusion)
        elif self.method_fusion == 's_only':
            feature_semantic = self.linearl2a(feature_semantic)
            output = self.classifier_align(feature_semantic)
        elif self.method_fusion == 'v_only':
            output = self.classifier_align(feature_visual)
        return output

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                
def select_visual_network(img_net_type, image_size,opt):
    if img_net_type == 'resnet50':
        from resnet import resnet50
        img_encoder = resnet50(image_size, pretrained=False)
    elif img_net_type == 'resnet152':
        from resnet import resnet152
        img_encoder = resnet152(image_size, pretrained=False)
    elif img_net_type == 'CLIP':
        from clip import clip
        model_name = 'ViT-B/16'
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        img_encoder, _ = clip.load(model_name, device=device, jit=False)
    elif img_net_type == 'resnet101':
        from resnet import resnet101
        img_encoder = resnet101(image_size, pretrained=False)
    elif img_net_type == 'resnet18':
        from resnet import resnet18
        img_encoder = resnet18(image_size, pretrained=True)
    elif img_net_type == 'vgg':
        from vgg import vgg19_bn
        img_encoder = vgg19_bn(image_size, pretrained=True)
    elif img_net_type == 'wrn':
        from wrn import WideResNet
        img_encoder = WideResNet(image_size)
    elif img_net_type == 'wiser':
        from wiser import wiser
        img_encoder = wiser(image_size)
    elif img_net_type == 'resnet50cbam':
        from resnet_cbam import resnet50_cbam
        img_encoder = resnet50_cbam(image_size, pretrained=True)
    elif img_net_type == 'resnet18cbam':
        from resnet_cbam import resnet18_cbam
        img_encoder = resnet18_cbam(image_size, pretrained=True)
    elif img_net_type == 'vit':
        import Transformer.model_ViT as ViT
        img_encoder = ViT.VisionTransformer(image_size=(32, 32),patch_size=(32, 32),dropout_rate=opt.vit_drop)
    elif img_net_type == 'vit_224':
        import Transformer.model_ViT as ViT
        img_encoder = ViT.VisionTransformer(image_size=(224, 224),patch_size=(16, 16),dropout_rate=opt.vit_drop)
    elif img_net_type == 'repvgg':
        import Transformer.model_RepVGG as RepVGG
        img_encoder = RepVGG.create_RepVGG_A2()
    elif img_net_type == 'repmlp':
        import Transformer.model_RepMLP as RepMLP
        img_encoder = RepMLP.create_RepMLPNet_T224()
    else:
        assert 1 < 0, 'Please indicate backbone network of image channel with any of resnet50/vgg19bn/wrn/wiser'

    return img_encoder

def select_word_network(path_data, CUDA, word_net_type, latent_len, dataset, num_word):

    if dataset == 'vireo':
        if word_net_type=='gru':
            gloveVector = matio.loadmat(path_data + 'wordVector.mat')['wordVector']
            from net_semantic import gru_encoder_t, gru_decoder_t
            word_encoder = gru_encoder_t(CUDA, gloveVector, latent_len)
            word_decoder = gru_decoder_t(latent_len)
        elif word_net_type=='nn':
            from net_semantic import nn_encoder_t_vireo, nn_decoder_t_vireo
            word_encoder = gru_encoder_t(latent_len, num_word)
            word_decoder = nn_decoder_t_vireo(latent_len)
    elif dataset == 'wide':
        from net_semantic import nn_encoder_t_vireo, nn_decoder_t_vireo
        word_encoder = nn_encoder_t_vireo(latent_len, num_word)
        word_decoder = nn_decoder_t_vireo(latent_len, num_word)
    
    return word_encoder, word_decoder


# load weights
def get_updateModel_v(model, path):
    pretrained_dict = torch.load(path, map_location='cpu')
    model_dict = model.state_dict()
    shared_dict = {k.replace('encoder_v.',''): v for k, v in pretrained_dict.items() if k.replace('encoder_v.','') in model_dict}
    model_dict.update(shared_dict)
    model.load_state_dict(model_dict)

    return model

def get_updateModel_s(model, path):
    pretrained_dict = torch.load(path, map_location='cpu')
    model_dict = model.state_dict()
    shared_dict = {k.replace('encoder_t.',''): v for k, v in pretrained_dict.items() if k.replace('encoder_t.','') in model_dict}
    model_dict.update(shared_dict)
    model.load_state_dict(model_dict)

    return model

def get_updateModel_decoder_s(model, path):
    pretrained_dict = torch.load(path, map_location='cpu')
    model_dict = model.state_dict()
    shared_dict = {k.replace('decoder_s.',''): v for k, v in pretrained_dict.items() if k.replace('decoder_s.','') in model_dict}
    model_dict.update(shared_dict)
    model.load_state_dict(model_dict)

    return model

def get_updateModel(model, path):
    pretrained_dict = torch.load(path, map_location='cpu')
    model_dict = model.state_dict()
    import ipdb

    shared_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    k_count = 0
    for k, v in pretrained_dict.items():
        print(k)
        k_count += 1 

    k_count = 0
    for k, v in shared_dict.items():
        print(k)
        k_count += 1 
    model_dict.update(shared_dict)
    model.load_state_dict(model_dict)
    return model

def get_updateModel_v_vit(model, path):
    pretrained_dict = torch.load(path, map_location='cpu')
    model_dict = model.state_dict()

    shared_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    print(shared_dict)
#     shared_dict.pop('classifier.weight')
#     shared_dict.pop('classifier.bias')
    model_dict.update(shared_dict)
    model.load_state_dict(model_dict)

    return model

def get_updateModel_v_vit2(model, path):
    pretrained_dict = torch.load(path, map_location='cpu')

    model_dict = model.state_dict()
    shared_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    for k, v in shared_dict.items():
        print(k)
    model_dict.update(shared_dict)
    model.load_state_dict(model_dict)

    return model

def get_updateModel_v_repvgg(model, path):
    pretrained_dict = torch.load(path, map_location='cpu')
    model_dict = model.state_dict()
    shared_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(shared_dict)
    model.load_state_dict(model_dict)

    return model

def get_updateModel_v_repmlp(model, path):
    pretrained_dict = torch.load(path, map_location='cpu')['model']
    model_dict = model.state_dict()
    shared_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(shared_dict)
    model.load_state_dict(model_dict)

    return model



class feature_fusion(nn.Module):
    def __init__(self, latent_len, dim_align, num_class):
        super( feature_fusion, self).__init__()
        self.classifier = nn.Linear(dim_align*2,num_class)
        self.linear1 = nn.Linear(latent_len,dim_align)
        self.relu = nn.LeakyReLU()
        self._initialize_weights()
    def forward(self, feature_pht, feature_gcn, beta):
        feature_gcn = self.relu(self.linear1(feature_gcn))
#         # Add
#         beta = 0.5
        feature_fusion = torch.cat((feature_gcn, feature_pht),dim=1)

        output = self.classifier(feature_fusion)
        return output, beta
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)

def set_optimizer(model, opt):
    optimizer = optim.SGD(model.parameters(),lr =opt.lr)
    scheduler = MultiStepLR(optimizer, milestones=[120, 160], gamma=0.1)

    warmup_scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5, after_scheduler=scheduler)
    return optimizer,warmup_scheduler

def set_optimizer_CLIP(model, opt):

    name_list = {'classifier.bias','classifier.weight'}
    for k, v in model.named_parameters():
        print(k)
        if k not in name_list:
            v.requires_grad = False
      
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),lr =opt.lr)
    scheduler = MultiStepLR(optimizer, milestones=[120, 160], gamma=0.1)

    warmup_scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5, after_scheduler=scheduler)
    return optimizer,warmup_scheduler


def set_optimizer_LPT(model, opt):

    name_list = {'prompt_learner.Prompt_Tokens_pool','prompt_learner.Prompt_Tokens','prompt_learner.Prompt_Keys','prompt_learner.head.weight','prompt_learner.head.bias'}
    for k, v in model.named_parameters():
        print(k)
        if k not in name_list:
            v.requires_grad = False
      
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),lr =opt.lr)
    scheduler = MultiStepLR(optimizer, milestones=[120, 160], gamma=0.1)

    warmup_scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5, after_scheduler=scheduler)
    return optimizer,warmup_scheduler

def set_optimizer_LPT_ccim(model, opt):

    name_list = {'prompt_learner.Prompt_Tokens_pool','prompt_learner.Prompt_Tokens','prompt_learner.Prompt_Keys','prompt_learner.head.weight','prompt_learner.head.bias','ccim.causal_intervention.query.weight','ccim.causal_intervention.query.bias','key.weight','ccim.causal_intervention.key.weight','ccim.causal_intervention.w_t.weight','key.bias','Prompt_Tokens','key1.bias','key2.bias','key3.bias','key4.bias','key5.bias','key6.bias','key7.bias','key8.bias','key9.bias','key10.bias','key11.bias','key1.weight','key2.weight','key3.weight','key4.weight','key5.weight','key6.weight','key7.weight','key8.weight','key9.weight','key10.weight','key11.weight'}
    for k, v in model.named_parameters():
        if k not in name_list:
            v.requires_grad = False

    for k,v in model.blocks[-1].named_parameters():
        print(k)
        v.requires_grad = True
    for k,v in model.blocks[-2].named_parameters():
        print(k)
        v.requires_grad = True
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),lr =opt.lr)
    scheduler = MultiStepLR(optimizer, milestones=[120, 160], gamma=0.1)

    warmup_scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5, after_scheduler=scheduler)
    return optimizer,warmup_scheduler


def set_optimizer_LPT_ccim_vireo(model, opt):
    name_list = {'prompt_learner.Prompt_Tokens_pool','prompt_learner.Prompt_Tokens','prompt_learner.Prompt_Keys','prompt_learner.head.weight','prompt_learner.head.bias','ccim.causal_intervention.query.weight','ccim.causal_intervention.query.bias','key.weight','ccim.causal_intervention.key.weight','ccim.causal_intervention.w_t.weight','key.bias','Prompt_Tokens','key1.bias','key2.bias','key3.bias','key4.bias','key5.bias','key6.bias','key7.bias','key8.bias','key9.bias','key10.bias','key11.bias','key1.weight','key2.weight','key3.weight','key4.weight','key5.weight','key6.weight','key7.weight','key8.weight','key9.weight','key10.weight','key11.weight','ccim.causal_intervention.key.weight','ccim.causal_intervention.query.weight','fc.weight'}
    for k, v in model.named_parameters():
        print(k)
        if k not in name_list:
            v.requires_grad = False
    for k,v in model.blocks[-1].named_parameters():
        print(k)
        v.requires_grad = True
    for k,v in model.blocks[-2].named_parameters():
        print(k)
        v.requires_grad = True
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), weight_decay=opt.weight_decay, lr=opt.lr)
    return optimizer

def set_optimizer_LPT_Gapco(model, opt):

    name_list = {'encoder_q.prompt_learner.Prompt_Tokens_pool','encoder_q.prompt_learner.Prompt_Tokens','encoder_q.prompt_learner.Prompt_Keys','encoder_q.prompt_learner.head.weight','encoder_q.prompt_learner.head.bias','encoder_q.ccim.causal_intervention.query.weight','encoder_q.ccim.causal_intervention.query.bias','encoder_q.key.weight','encoder_q.ccim.causal_intervention.key.weight','encoder_q.ccim.causal_intervention.w_t.weight','encoder_q.key.bias','encoder_q.Prompt_Tokens','encoder_q.key1.bias','encoder_q.key2.bias','encoder_q.key3.bias','encoder_q.key4.bias','encoder_q.key5.bias','encoder_q.key6.bias','encoder_q.key7.bias','encoder_q.key8.bias','encoder_q.key9.bias','encoder_q.key10.bias','encoder_q.key11.bias','encoder_q.key1.weight','encoder_q.key2.weight','encoder_q.key3.weight','encoder_q.key4.weight','encoder_q.key5.weight','encoder_q.key6.weight','encoder_q.key7.weight','encoder_q.key8.weight','encoder_q.key9.weight','encoder_q.key10.weight','encoder_q.key11.weight','linear.weight'}
    for k, v in model.named_parameters():
        if k not in name_list:
            print(k)
            v.requires_grad = False

#     for k,v in model.blocks[-1].named_parameters():
#         print(k)
#         v.requires_grad = True
#     for k,v in model.blocks[-2].named_parameters():
#         print(k)
#         v.requires_grad = True
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),lr =opt.lr)
    scheduler = MultiStepLR(optimizer, milestones=[120, 160], gamma=0.1)

    warmup_scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5, after_scheduler=scheduler)
    return optimizer,warmup_scheduler


def set_optimizer_LPT_cuda(model, opt):

    name_list = {'prompt_learner.Prompt_Tokens_pool','prompt_learner.Prompt_Tokens','prompt_learner.Prompt_Keys','prompt_learner.head.weight','prompt_learner.head.bias','ccim.causal_intervention.query.weight','ccim.causal_intervention.query.bias','key.weight','ccim.causal_intervention.key.weight','ccim.causal_intervention.w_t.weight','key.bias','Prompt_Tokens','key1.bias','key2.bias','key3.bias','key4.bias','key5.bias','key6.bias','key7.bias','key8.bias','key9.bias','key10.bias','key11.bias','key1.weight','key2.weight','key3.weight','key4.weight','key5.weight','key6.weight','key7.weight','key8.weight','key9.weight','key10.weight','key11.weight'}
    for k, v in model.named_parameters():
        if k not in name_list:
            v.requires_grad = False

#     for k,v in model.blocks[-1].named_parameters():
#         print(k)
#         v.requires_grad = True
#     for k,v in model.blocks[-2].named_parameters():
#         print(k)
#         v.requires_grad = True
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),lr =opt.lr)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    return optimizer, scheduler

def set_optimizer_VPT_TDE_vireo(model, opt):
    name_list = {'prompt_learner.Prompt_Tokens','norm.weight','norm.bias','prompt_learner.Prompt_Tokens','prompt_learner.head.weight','prompt_learner.head.bias','Prompt_Tokens'}
    for k, v in model.named_parameters():
        print(k)
        if k not in name_list:
            v.requires_grad = False
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), weight_decay=opt.weight_decay, lr=opt.lr)
    return optimizer

def set_optimizer_VPT(model, opt):

    name_list = {'prompt_learner.Prompt_Tokens','norm.weight','norm.bias','prompt_learner.Prompt_Tokens','prompt_learner.head.weight','prompt_learner.head.bias'}
    for k, v in model.named_parameters():
        print(k)
        if k not in name_list:
            v.requires_grad = False

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),lr =opt.lr)
    scheduler = MultiStepLR(optimizer, milestones=[120, 160], gamma=0.1)

    warmup_scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5, after_scheduler=scheduler)
    return optimizer,warmup_scheduler

def set_optimizer_VPT_ccim(model, opt):

    name_list = {'prompt_learner.Prompt_Tokens','norm.weight','norm.bias','prompt_learner.Prompt_Tokens','prompt_learner.head.weight','prompt_learner.head.bias','ccim.causal_intervention.query.weight','ccim.causal_intervention.query.bias','key.weight','ccim.causal_intervention.key.weight','ccim.causal_intervention.w_t.weight','key.bias','Prompt_Tokens'}

    for k, v in model.named_parameters():
        
        if k not in name_list:
            print(k)
            v.requires_grad = False
    
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),lr =opt.lr)
    scheduler = MultiStepLR(optimizer, milestones=[120, 160], gamma=0.1)

    warmup_scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5, after_scheduler=scheduler)
    return optimizer,warmup_scheduler

def set_optimizer_VPT_lgcam(model, opt):
   
    name_list = {'backbone.prompt_learner.Prompt_Tokens','backbone.norm.weight','backbone.norm.bias','backbone.prompt_learner.Prompt_Tokens','backbone.prompt_learner.head.weight','backbone.prompt_learner.head.bias','feat_lg.w_q.weight','feat_ll.linear2.bias','feat_ll.linear2.weight','feat_ll.linear1.bias','feat_ll.linear1.weight','feat_ll.w_kv.weight','feat_ll.w_q.weight','classifier.bias','classifier.weight','feat_lg.w_kv.weight','feat_lg.linear1.weight','feat_lg.linear1.bias','feat_lg.linear2.weight','feat_lg.linear2.bias'}
    for k, v in model.named_parameters():
        
        if k not in name_list:
            print(k)
            v.requires_grad = False

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),lr =opt.lr)
    scheduler = MultiStepLR(optimizer, milestones=[120, 160], gamma=0.1)

    warmup_scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5, after_scheduler=scheduler)
    return optimizer,warmup_scheduler


def set_optimizer_Hivision(model, opt):

    name_list = {'prototype_0.weight','prototype_1.weight','prototype_2.weight','prototype_3.weight','prototype_4.weight','prototype_5.weight','prototype_6.weight','prototype_7.weight','prototype_8.weight','prototype_9.weight','prototype_10.weight','matrix_0.weight','matrix_1.weight','matrix_2.weight','matrix_3.weight','matrix_4.weight','matrix_5.weight','matrix_6.weight','matrix_8.weight','matrix_7.weight','matrix_9.weight','matrix_10.weight','fc.weight','fc.bias','norm.weight','norm.bias','head.weight','head.bias'}
    for k, v in model.named_parameters():
        
        if k not in name_list:
            print(k)
            v.requires_grad = False

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),lr =opt.lr)
    scheduler = MultiStepLR(optimizer, milestones=[120, 160], gamma=0.1)

    warmup_scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5, after_scheduler=scheduler)
    return optimizer,warmup_scheduler

def set_optimizer_Hivision2(model, opt):

    name_list = {'prototype_0.weight','prototype_1.weight','prototype_2.weight','prototype_3.weight','prototype_4.weight','prototype_5.weight','prototype_6.weight','prototype_7.weight','prototype_8.weight','prototype_9.weight','prototype_10.weight','matrix_0.weight','matrix_1.weight','matrix_2.weight','matrix_3.weight','matrix_4.weight','matrix_5.weight','matrix_6.weight','matrix_8.weight','matrix_7.weight','matrix_9.weight','matrix_10.weight','fc.weight','fc.bias','norm.weight','norm.bias','head.weight','head.bias','prompt_learner.Prompt_Tokens','norm.weight','norm.bias','prompt_learner.Prompt_Tokens','prompt_learner.head.weight','prompt_learner.head.bias'}
    for k, v in model.named_parameters():
        
        if k not in name_list:
            print(k)
            v.requires_grad = False

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),lr =opt.lr)
    scheduler = MultiStepLR(optimizer, milestones=[120, 160], gamma=0.1)

    warmup_scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5, after_scheduler=scheduler)
    return optimizer,warmup_scheduler

def set_optimizer_causal(model, opt):

    name_list = {'fc.weight','fc.bias'}
    for k, v in model.named_parameters():
        
        if k not in name_list:
            print(k)
            v.requires_grad = False

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),lr =opt.lr)
    scheduler = MultiStepLR(optimizer, milestones=[120, 160], gamma=0.1)

    warmup_scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5, after_scheduler=scheduler)
    return optimizer,warmup_scheduler

def set_optimizer_fusion(model, opt):

    name_list = {'classifier.weight','classifier.bias','lineara2a.weight','lineara2a.bias','lineara2a2.weight','lineara2a2.bias'}
    for k, v in model.named_parameters():
        
        if k not in name_list:
            print(k)
            v.requires_grad = False

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),lr =opt.lr)
    scheduler = MultiStepLR(optimizer, milestones=[120, 160], gamma=0.1)

    warmup_scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5, after_scheduler=scheduler)
    return optimizer,warmup_scheduler

def set_optimizer2(model, opt):
    optimizer = optim.Adam(model.parameters(), weight_decay=opt.weight_decay, lr=opt.lr)
    return optimizer

def set_classifier_optimizer(model, opt):
    name_list = {'classifier.weight', 'classifier.bias'}
    for k, v in model.named_parameters():
        if k not in name_list:
            v.requires_grad = False
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),lr =opt.lr)
    return optimizer

import torch.nn.init as init
def set_optimizer_flouer(model, opt):
    name_list = {'fc.weight', 'fc.bias'}

    for k, v in model.named_parameters():
        if k in name_list:
            if 'weight' in k:
                print(1)
                init.kaiming_uniform_(v)
            elif 'bias' in k:
                init.constant_(v, 0.0)
        if k not in name_list:
            
            v.requires_grad = False
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), weight_decay=opt.weight_decay, lr=opt.lr)
    return optimizer



def set_classifier_optimizer2(model, opt):
    name_list = {'classifier.weight', 'classifier.bias'}
    for k, v in model.named_parameters():
        if k not in name_list:
            v.requires_grad = False
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), weight_decay=opt.weight_decay, lr=opt.lr)
    return optimizer


def set_classifier_optimizer3(model, opt):
    name_list = {'classifier2.weight', 'classifier2.bias'}
    for k, v in model.named_parameters():
        if k not in name_list:
            v.requires_grad = False
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),lr =opt.lr)
    return optimizer


def set_classifier_optimizer(model, opt):
    # params in ingre prediction net
    pretrained_dict = torch.load(opt.path_pretrain_v, map_location='cpu')
    model_dict = model.state_dict()
    
    for k, v in model.named_parameters():
        if k in pretrained_dict:
            v.requires_grad =False
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), weight_decay=opt.weight_decay, lr=opt.lr)
    
    return optimizer


def build(CUDA, opt):

    # network for image channel
    if opt.net_v in ['resnet18','resnet18cbam']:
        dim_feat_v = 512
    elif opt.net_v in ['resnet50','resnet50cbam', 'wrn', 'wiser']:
        dim_feat_v = 2048
    elif opt.net_v in ['resnet152','resnet101']:
        dim_feat_v = 2048
    elif opt.net_v == 'vgg':
        dim_feat_v = 4096
    elif opt.net_v.startswith('vit'):
        dim_feat_v = 768
    elif opt.net_v.startswith('swin'):
        dim_feat_v = 768
    elif opt.net_v == 'repvgg':
        dim_feat_v = 1408
    elif opt.net_v == 'CLIP':
        dim_feat_v = 512
    elif opt.net_v == 'repmlp':
        dim_feat_v = 512
    # build model

    if opt.modality == 'v':
#         import ipdb
#         ipdb.set_trace()
#         print(opt.net_v)
#         encoder_v = models.resnet50(pretrained=True)
#         encoder_v = torch.nn.Sequential(*list(encoder_v.children())[:-1])
#         encoder_v = select_visual_network(opt.net_v, opt.size_img, opt)
        if opt.rebalance=='decouple_crt':
            model = Decouple_crt(encoder_v, opt.net_v, opt.method, opt.num_cls, dim_feat_v, opt.dim_align)
        elif opt.method=='lgcam':
            import build_model_prompt
            import build_model
            encoder_v = build_model_prompt.build_linear_model(num_classes=100, img_size=[224,224], base_model='vit_base_patch16_224_in21k', model_idx='ViT', patch_size=16)
            #'global_dict/cifar_global_dict_512.npy'
            global_dict = torch.from_numpy(
                np.load('global_dict/cifar_global_dict_512.npy')).cuda()
            model = LGCAM(encoder_v, global_dict=global_dict, num_classes=100,opt=opt)
            
            model.cuda()
            return model
        elif opt.method=='lgcam_resnet':
            import build_model_prompt
            import build_model
            #'global_dict/cifar_global_dict_512.npy'
            global_dict = torch.from_numpy(
                np.load('global_dict/resnet/cifar_ratio0.02_global_dict_128.npy')).cuda()
            model = LGCAM(encoder_v, global_dict=global_dict, num_classes=100,opt=opt)

            model.cuda()
            return model
        elif opt.method=='ccim_resnet':
            import build_model_prompt
            import build_model
            confounder = np.load(f'backdoor/resnet/cifar_ratio0.02_global_dict_64.npy', allow_pickle=True)
            confounder = torch.Tensor(confounder)
            confounder = torch.stack([x for x in confounder])
            prob = np.load(f'backdoor/resnet/backdoor_labels_64.npy')
            prob = torch.from_numpy(prob).reshape(-1, 1)
            confounder = confounder.cuda()
            prob = prob.cuda()
            ccim = CCIM(num_joint_feature=2048, num_gz=2048, strategy='dp_cause',encoder_v = encoder_v).cuda()

            #'global_dict/cifar_global_dict_512.npy'

            model = VisualNet(ccim, opt.net_v, opt.method, opt.num_cls, dim_feat_v,opt)

            model.cuda()
            return model
        elif opt.method=='lgcam_VPT':
            import build_model_prompt
            import build_model
            encoder_v = build_model_prompt.build_promptmodel(num_classes=100, img_size=[224,224], base_model='vit_base_patch16_224_in21k', model_idx='ViT', patch_size=16,Prompt_Token_num=10,VPT_type="Deep",opt=opt)

            #'global_dict/cifar_global_dict_512.npy'
            global_dict = torch.from_numpy(
                np.load('global_dict/ratio0.02_VPT/cifar_global_dict_512.npy')).cuda()
            model = LGCAM(encoder_v, global_dict=global_dict, num_classes=100,opt=opt)
            model.cuda()
            return model
        elif opt.method=='GPaco_lgcam':

#             pre_path =  'result/ratio0.02/vit/datset=CIFAR100~net_v=vit_224~method=...~bs=64~decay=4~lr=0.2~lrd_rate=0.1~drop=0.0/model_best.pt'
#             encoder_v  = get_updateModel_v_vit2(encoder_v,pre_path)
            import build_model_prompt
            encoder_v1 = build_model_prompt.build_Gpaco_model(num_classes=100, img_size=[224,224], base_model='vit_base_patch16_224_in21k', model_idx='ViT', patch_size=16)
            
            encoder_v = MoCo(encoder_v1,dim=opt.moco_dim, K=opt.moco_k,m=opt.moco_m, T=opt.moco_t, mlp=False,num_classes=100)
            encoder_v = get_updateModel_v_vit2(encoder_v,'result/ratio0.02/vit/GPaco/datset=CIFAR100~net_v=vit_224~method=GPaco~bs=32~decay=4~lr=0.01~lrd_rate=0.1~drop=0.0/model_best.pt')
            global_dict = torch.from_numpy(
                np.load('global_dict/cifar_global_dict_512.npy')).cuda()
            model = LGCAM_GPaco(encoder_v, global_dict=None, num_classes=100,opt=opt)
            model.cuda()
            return model
        elif opt.method=='GPaco':
            print(opt.net_v)
            import build_model_prompt
            encoder_v1 = build_model_prompt.build_Gpaco_model(num_classes=100, img_size=[224,224], base_model='vit_base_patch16_224_in21k', model_idx='ViT', patch_size=16)

            model = MoCo(encoder_v1,dim=opt.moco_dim, K=opt.moco_k,m=opt.moco_m, T=opt.moco_t, mlp=False,num_classes=100)
        elif opt.net_v in ['vit','vit_224','repvgg','repmlp','swin']:
            if opt.net_v=='vit':
                import build_model_prompt
                encoder_v = build_model_prompt.build_Gpaco_model(num_classes=100, img_size=[224,224], base_model='vit_base_patch16_224_in21k', model_idx='ViT', patch_size=16)
            elif opt.net_v=='vit_224':
                path_pretrain = 'Transformer/imagenet21k+imagenet2012_ViT-B_16-224.pth'
                encoder_v = get_updateModel_v_vit(encoder_v, path_pretrain)
            elif opt.net_v=='repvgg':
                path_pretrain = 'Transformer/RepVGG-A2-train.pth'
                encoder_v = get_updateModel_v_repvgg(encoder_v, path_pretrain)
            elif opt.net_v=='repmlp':
                path_pretrain = 'Transformer/RepMLPNet-T224-train-acc76.62.pth'
                encoder_v = get_updateModel_v_repmlp(encoder_v, path_pretrain)
            elif opt.net_v=='swin':
                state_dict = torch.load("swin-transformer/pytorch_model_swin.bin", map_location='cpu')
                encoder_v = timm.create_model("swin_base_patch4_window7_224" , pretrained=False)
                encoder_v.load_state_dict(state_dict,strict=False)
                encoder_v.global_pool = nn.AdaptiveAvgPool2d(1) 
                num_classes = 100
                encoder_v.head = nn.Linear(encoder_v.head.in_features, num_classes) 
            model = VisualNet(encoder_v, opt.net_v, opt.method, opt.num_cls, dim_feat_v,opt)
        elif opt.net_v in ['CLIP']:
            import clip
            model_name = 'ViT-B/16'
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            encoder_v, _ = clip.load(model_name, device=device, jit=False)
            model = VisualNet(encoder_v, opt.net_v, opt.method, opt.num_cls, dim_feat_v,opt)
        else:
            model = VisualNet(encoder_v, opt.net_v, opt.method, opt.num_cls, dim_feat_v,opt)
            if opt.method  == 'H2T':
                model = VisualNetH2T(encoder_v, opt.net_v, opt.method, opt.num_cls, dim_feat_v,opt)
    elif opt.modality == 's':
        print(opt.net_s)
        if opt.method=='recon':
            encoder_s, decoder_s = select_word_network(opt.path_data, CUDA, opt.net_s, opt.dim_latent, opt.dataset, opt.num_word)
            encoder_s = get_updateModel_s(encoder_s, opt.path_pretrain_s)
            model = S2S_recon(opt.net_s, encoder_s, decoder_s, opt.dim_latent, opt.num_word)
        else:
            encoder_s, decoder_s = select_word_network(opt.path_data, CUDA, opt.net_s, opt.dim_latent, opt.dataset, opt.num_word)
            model = SemanticNet(encoder_s, opt.dim_latent, opt.num_cls)
    if opt.modality == 'v+s':
        if opt.method=='cma':
            print(opt.net_v)
            print(opt.net_s)
            encoder_v = select_visual_network(opt.net_v, opt.size_img, opt)
            encoder_s, decoder_s = select_word_network(opt.path_data, CUDA, opt.net_s, opt.dim_latent, opt.dataset, opt.num_word)
            encoder_v = get_updateModel_v(encoder_v, opt.path_pretrain_v)
            encoder_s = get_updateModel_s(encoder_s, opt.path_pretrain_s)
            model = V2S_Align(encoder_s, encoder_v, opt.dim_latent, dim_feat_v, opt.dim_align, opt.num_cls)
        elif opt.method=='pht':
            print(opt.net_v)
            print(opt.net_s)
            encoder_v = select_visual_network(opt.net_v, opt.size_img, opt)
            encoder_s, decoder_s = select_word_network(opt.path_data, CUDA, opt.net_s, opt.dim_latent, opt.dataset, opt.num_word)
            encoder_v = get_updateModel_v(encoder_v, opt.path_pretrain_v)
            encoder_s = get_updateModel_s(encoder_s, opt.path_pretrain_s)
            model = V2S_ATNet(encoder_s, encoder_v, opt.dim_latent, dim_feat_v, opt.pht_partial, opt.num_cls)
        elif opt.method=='img2word':
            print(opt.net_v)
            print(opt.net_s)
            encoder_v = select_visual_network(opt.net_v, opt.size_img, opt)
            encoder_s, decoder_s = select_word_network(opt.path_data, CUDA, opt.net_s, opt.dim_latent, opt.dataset, opt.num_word)
            encoder_v = get_updateModel_v(encoder_v, opt.path_pretrain_v)
            encoder_s = get_updateModel_s(encoder_s, opt.path_pretrain_s)
            decoder_s = get_updateModel_decoder_s(decoder_s, opt.path_pretrain_decoder_s)
            model = Img2word(encoder_v, [encoder_s, decoder_s], dim_feat_v, opt.dim_latent, opt.dim_align, opt.num_cls, opt.net_s)

        elif opt.method in ['slf', 'cmi','slf_multi_hot', 'slf_embedding','slf_mm']:
            print(opt.net_v)
            print(opt.net_s)
            encoder_v = select_visual_network(opt.net_v, opt.size_img, opt)
            encoder_s, decoder_s = select_word_network(opt.path_data, CUDA, opt.net_s, opt.dim_latent, opt.dataset, opt.num_word)
            model_img2word = Img2word(encoder_v, [encoder_s, decoder_s], dim_feat_v, opt.dim_latent, opt.dim_align, opt.num_word, opt.net_s)
            model_img2word = get_updateModel(model_img2word, opt.path_pretrain_img2word)
            if opt.method.startswith('slf'):
                model = V2S_CIS(model_img2word, dim_feat_v, opt.dim_latent, opt.dim_embed, opt.num_cls, opt.num_word, opt.adj, opt.net_s)
            elif opt.method=='cmi':
                model = V2S_CMI(model_img2word, opt.dim_latent, opt.num_cls, opt.num_word, opt.cmi, opt.topk)
            elif opt.method in ['slf_multi_hot','slf_embedding']:
                model = V2S2C(model_img2word, opt.dim_latent, opt.num_cls, opt.num_word)
        elif opt.method=='ff':
            model = feature_fusion(opt.dim_latent, opt.dim_align, opt.num_cls)
        elif opt.method=='cmlt':
            model = FeatureFusion(opt.dim_latent, opt.dim_align, opt.num_cls, method_fusion='max', beta=0.5)
    if CUDA:        
        model = model.cuda()
    return model