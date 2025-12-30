import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import normal
# -------------------------------- long-tailed loss --------------------------------
class PaCoLoss(nn.Module):
    def __init__(self, alpha, beta=1.0, gamma=1.0, supt=1.0, temperature=1.0, base_temperature=None, K=128, num_classes=1000):
        super(PaCoLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = temperature if base_temperature is None else base_temperature
        self.K = 0.05
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.supt = supt
        self.num_classes = num_classes

    def cal_weight_for_classes(self, cls_num_list):
        cls_num_list = torch.Tensor(cls_num_list).view(1, self.num_classes)
        self.weight = cls_num_list / cls_num_list.sum()
        self.weight = self.weight.to(torch.device('cuda'))

    def forward(self, features, labels=None, sup_logits=None, mask=None, epoch=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        ss = features.shape[0]
#         batch_size =int( ( features.shape[0] - self.K ) // 2) +  1
        batch_size = 64
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels[:batch_size], labels.T).float().to(device)

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features[:batch_size], features.T),
            self.temperature)

        # add supervised logits
        anchor_dot_contrast = torch.cat(( (sup_logits + torch.log(self.weight + 1e-9) ) / self.supt, anchor_dot_contrast), dim=1)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # add ground truth 
        one_hot_label = torch.nn.functional.one_hot(labels[:batch_size,].view(-1,), num_classes=self.num_classes).to(torch.float32)
        mask = torch.cat((one_hot_label * self.beta, mask * self.alpha), dim=1)

        # compute log_prob
        logits_mask = torch.cat((torch.ones(batch_size, self.num_classes).to(device), self.gamma * logits_mask), dim=1)
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
       
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss
#AGCL Loss
class ASLSingleLabel(nn.Module):
    def __init__(self, gamma_pos=0, gamma_neg=4, eps: float = 0.1, reduction='mean'):
        super(ASLSingleLabel, self).__init__()

        self.eps = eps
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.targets_classes = []  # prevent gpu repeated memory allocation
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.reduction = reduction

    def forward(self, inputs, target, reduction=None):
        num_classes = inputs.size()[-1]
        log_preds = self.logsoftmax(inputs)
        

        # ASL weights
        with torch.no_grad():
            self.targets_classes = torch.zeros_like(inputs).scatter_(1, target.long().unsqueeze(1), 1)
            targets = self.targets_classes
            anti_targets = 1 - targets
        xs_pos = torch.exp(log_preds)
        xs_neg = torch.ones_like(xs_pos) - xs_pos
        xs_pos_new = torch.mul(xs_pos.clone(), targets.clone().detach()) #* targets.detach()
        xs_neg_new = torch.mul(xs_neg.clone(), anti_targets.clone().detach()) #* anti_targets.detach()
        asymmetric_w = torch.pow(1 - xs_pos_new - xs_neg_new,
                                 self.gamma_pos * targets + self.gamma_neg * anti_targets)
        log_preds = log_preds * asymmetric_w

        if self.eps > 0:  # label smoothing
            self.targets_classes.mul_(1 - self.eps).add_(self.eps / num_classes)

        # loss calculation
        loss = - self.targets_classes.mul(log_preds)

        loss = loss.sum(dim=-1)
        if self.reduction == 'mean':
            loss = loss.mean()

        return loss

def focal_loss_new(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)   #目标类概率
    loss = (1 - p.detach()) ** gamma * input_values
    return loss.mean()

class GCLLoss(nn.Module):
    
    def __init__(self, cls_num_list, m=0.5, weight=None, s=30, train_cls=False, noise_mul = 1., gamma=0.):
        super(GCLLoss, self).__init__()
        cls_list = torch.cuda.FloatTensor(cls_num_list)
        m_list = torch.log(cls_list)
        m_list = m_list.max()-m_list
        self.m_list = m_list
        assert s > 0
        self.m = m
        self.s = s
        self.weight = weight
        self.simpler = normal.Normal(0, 1/3)
        self.train_cls = train_cls
        self.noise_mul = noise_mul
        self.gamma = gamma
           
                                         
    def forward(self, cosine, target):
        index = torch.zeros_like(cosine, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
             
        noise = self.simpler.sample(cosine.shape).clamp(-1, 1).to(cosine.device) #self.scale(torch.randn(cosine.shape).to(cosine.device))  
        
        #cosine = cosine - self.noise_mul * noise/self.m_list.max() *self.m_list   
        cosine = cosine - self.noise_mul * noise.abs()/self.m_list.max() *self.m_list         
        output = torch.where(index, cosine-self.m, cosine)                    
        if self.train_cls:
            return focal_loss_new(F.cross_entropy(self.s*output, target, reduction='none', weight=self.weight), self.gamma)
        else:    
            return F.cross_entropy(self.s*output, target, weight=self.weight)     

class AGCL(nn.Module):

    def __init__(self, cls_num_list, m=0.5, weight=None, s=30, train_cls=False, noise_mul = 1., gamma=0., gamma_pos=0., gamma_neg=4.):
        super(AGCL, self).__init__()
        cls_list = torch.cuda.FloatTensor(cls_num_list)
        m_list = torch.log(cls_list)
        m_list = m_list.max()-m_list
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.m = m
        self.simpler = normal.Normal(0, 1/3)
        self.train_cls = train_cls
        self.noise_mul = noise_mul
        self.gamma = gamma
        self.loss_func = ASLSingleLabel(gamma_pos=gamma_pos, gamma_neg=gamma_neg)


    def forward(self, cosine, target):
        index = torch.zeros_like(cosine, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        noise = self.simpler.sample(cosine.shape).clamp(-1, 1).to(cosine.device) #self.scale(torch.randn(cosine.shape).to(cosine.device))

        #cosine = cosine - self.noise_mul * noise/self.m_list.max() *self.m_list
        cosine = cosine - self.noise_mul * noise.abs()/self.m_list.max() *self.m_list
        output = torch.where(index, cosine-self.m, cosine)
        if self.train_cls:
            return focal_loss_new(F.cross_entropy(self.s*output, target, reduction='none', weight=self.weight), self.gamma)
        else:
            return self.loss_func(self.s*output, target)
        
# focal loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma

    def forward(self, logits, target, criterion):
        return self.loss_compute(criterion(logits, target), self.gamma)
    def loss_compute(self, loss_input, gamma):
        """Computes the focal loss"""
        p = torch.exp(-loss_input)
        loss = (1 - p) ** gamma * loss_input
        return loss.mean()

#vit-balbce-loss
class BCE_loss(nn.Module):

    def __init__(self, args,
                target_threshold=None, 
                type=None,
                reduction='mean', 
                pos_weight=None):
        super(BCE_loss, self).__init__()

        self.lam = 1.
        self.K = 1.
        self.smoothing = 0.0
        self.target_threshold = target_threshold
        self.weight = None
        self.pi = None
        self.reduction = reduction
        self.register_buffer('pos_weight', pos_weight)

        if type == 'Bal':
            self._cal_bal_pi(args)
        if type == 'CB':
            self._cal_cb_weight(args)

    def _cal_bal_pi(self, args):
        cls_num = torch.Tensor(args.cls_num)
        self.pi = cls_num / torch.sum(cls_num)

    def _cal_cb_weight(self, args):
        eff_beta = 0.9999
        effective_num = 1.0 - np.power(eff_beta, args.cls_num)
        per_cls_weights = (1.0 - eff_beta) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(args.cls_num)
        self.weight = torch.FloatTensor(per_cls_weights).cuda()

    def _bal_sigmod_bias(self, x):
        pi = self.pi.cuda()
        bias = torch.log(pi) - torch.log(1-pi)
        x = x + self.K * bias
        return x

    def _neg_reg(self, labels, logits, weight=None):
        if weight == None:
            weight = torch.ones_like(labels).cuda()
        pi = self.pi.cuda()
        bias = torch.log(pi) - torch.log(1-pi)
        logits = logits * (1 - labels) * self.lam + logits * labels # neg + pos
        logits = logits + self.K * bias
        weight = weight / self.lam * (1 - labels) + weight * labels # neg + pos
        return logits, weight

    def _one_hot(self, x, target):
        num_classes = x.shape[-1]
        off_value = self.smoothing / num_classes
        on_value = 1. - self.smoothing + off_value
        target = target.long().view(-1, 1)
        target = torch.full((target.size()[0], num_classes),
            off_value, device=x.device, 
            dtype=x.dtype).scatter_(1, target, on_value)
        return target

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert x.shape[0] == target.shape[0]
        if target.shape != x.shape:
            target = self._one_hot(x, target)
        if self.target_threshold is not None:
            target = target.gt(self.target_threshold).to(dtype=target.dtype)
        weight = self.weight
        if self.pi != None: x = self._bal_sigmod_bias(x)
        # if self.lam != None:
        #     x, weight = self._neg_reg(target, x)
        C = x.shape[-1] # + log C
        return C * F.binary_cross_entropy_with_logits(
                    x, target, weight, self.pos_weight,
                    reduction=self.reduction)

# cross-modal focal loss
class CM_FocalLoss(nn.Module):
    def __init__(self, gamma=0.):
        super(CM_FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
    def forward(self, logits_m1, logits_m2, target, criterion):
        return self.loss_compute(criterion(logits_m1, target), criterion(logits_m2, target), self.gamma)
    def loss_compute(self, loss_input_m1, loss_input_m2, gamma):
        """Computes the focal loss"""
        p = torch.exp(-loss_input_m1)
        q = torch.exp(-loss_input_m2)
        w = q*2*p*q/(p+q)
        loss = (1 - w) ** gamma * loss_input_m1
        return loss.mean()
    
# LDAM loss
class LDAMLoss(nn.Module):
    
    def __init__(self, cls_num_list, max_m=0.5, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list)) # set max to be 0.5
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s

    def forward(self, x, target, criterion):
        # for single-label
        if target[0].shape==torch.Size([]):
            index = torch.zeros_like(x, dtype=torch.uint8)
            index.scatter_(1, target.data.view(-1, 1), 1) # (size_batch, num_class)
            index_float = index.type(torch.cuda.FloatTensor)
        else:
            index = torch.tensor(target,dtype=torch.uint8).cuda()
            index_float = target
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1)) # (1, num_class) * (num_class, size_batch) -> (1, size_batch)
        batch_m = batch_m.view((-1, 1)) # (1, size_batch)->(size_batch, 1)
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x) # replace index==1 only
        return criterion(self.s*output, target).mean()
    
    
# -------------------------------- cross-modal --------------------------------
def get_distance_matrix(shift_vector):
    #construct the shift vector
    shift_vector_repeated = shift_vector.unsqueeze(1).repeat(1,len(shift_vector))
    distance_matrix = torch.abs(shift_vector_repeated - shift_vector)
    return distance_matrix

# theta_0_2, theta_1_2        
def get_diverse_loss(shift, seq_len_list, weight, use_diverse_loss_version):
    shift_x = shift[0] # valid_seq_batch
    shift_y = shift[1] # valid_seq_batch
    
    diverse_loss_all = 0

    # len(seq_len_list)=batch_size, seq_len_list is the lengths of valid sequence in batch
    for i in range(len(seq_len_list)):
        if i == 0:
            start_seq = 0
        else:
            start_seq = seq_len_list[:i].sum()
        
        cur_img_shift_x_list = shift_x[start_seq:(start_seq+seq_len_list[i])] # current sample's shift_x
        cur_img_shift_y_list = shift_y[start_seq:(start_seq+seq_len_list[i])] # current sample's shift_y
        
        if use_diverse_loss_version < 3:
            cur_img_shift_x_mean = cur_img_shift_x_list.mean().detach()
            cur_img_shift_y_mean = cur_img_shift_y_list.mean().detach()

            cur_img_diverse_x = cur_img_shift_x_list - cur_img_shift_x_mean
            cur_img_diverse_y = cur_img_shift_y_list - cur_img_shift_y_mean

            diverse_value = torch.abs(cur_img_diverse_x).mean() + torch.abs(cur_img_diverse_y).mean()
            if use_diverse_loss_version == 1:
                cur_img_shift_loss = 1/(diverse_value+1e-2)
            elif use_diverse_loss_version == 2:
                cur_img_shift_loss = torch.exp(-diverse_value)
        
        elif use_diverse_loss_version == 4:
            distance_matrix_x = get_distance_matrix(cur_img_shift_x_list)
            distance_matrix_y = get_distance_matrix(cur_img_shift_y_list)
            distance_matrix_sum = distance_matrix_x.sum()*0.5 + distance_matrix_y.sum()*0.5
            cur_img_shift_loss = torch.exp(-distance_matrix_sum)
            
        diverse_loss_all += cur_img_shift_loss
        
    #import ipdb; ipdb.set_trace()
    return (diverse_loss_all/len(seq_len_list))*weight

class xERMLoss(nn.Module):
    def __init__(self, gamma):
        super(xERMLoss, self).__init__()
        self.CE_loss = nn.CrossEntropyLoss(reduction='none')
        self.gamma = gamma
#         if critertion != None:
#             self.CE_loss = critertion

    def forward(self, logits_TE, logits_TDE, logits_student, labels):
        # calculate weight
        TDE_acc = self.CE_loss(logits_TDE, labels)
        TE_acc = self.CE_loss(logits_TE, labels)
        TDE_acc = torch.pow(TDE_acc, self.gamma)
        TE_acc = torch.pow(TE_acc, self.gamma)
        weight = TDE_acc/(TDE_acc + TE_acc)
        # student td loss
        te_loss = self.CE_loss(logits_student, labels)

        # student tde loss
        prob_tde = F.softmax(logits_TDE, -1).clone().detach()
        prob_student = F.softmax(logits_student, -1)
        tde_loss = - prob_tde * prob_student.log()
        tde_loss = tde_loss.sum(1)

        loss = (weight*tde_loss).mean() + ((1 - weight)*te_loss).mean()

        return loss



def get_distance_vector(point_vector):
    distance_vector = torch.abs(point_vector - point_vector.mean())
    return distance_vector.mean()

def get_anti_outlier_loss(point_vectors, weight):
    loss_all = 0
    for point_vector in point_vectors:
        loss_value = get_distance_vector(point_vector)
        loss_all+= loss_value
    return loss_all * weight    
    

def get_shift_loss(shift, loc_range, weight):
    shift_x = shift[0] #(batch_size, x^t)
    shift_y = shift[1]
    shift_loss = (torch.abs(shift_x) - loc_range)**2 + (torch.abs(shift_y) - loc_range)**2 
    return torch.mean(shift_loss * 0.5)*weight
#     shift_loss_x = torch.max(torch.abs(shift_x)-loc_range,torch.zeros(len(shift_x)).cuda())
#     shift_loss_y = torch.max(torch.abs(shift_y)-loc_range,torch.zeros(len(shift_x)).cuda())
#     return torch.mean((shift_loss_x + shift_loss_y) * 0.5)*weight
# [theata_0_0, theta_1_1]
def get_scale_upperbound_loss(scale, upperbound, weight):
    scale_x = scale[0] # valid_seq_batch
    scale_y = scale[1] # valid_seq_batch
    upperbound_loss_x = torch.max(torch.abs(scale_x)-upperbound,torch.zeros(len(scale_x)).cuda()) # max(|x|-upperbound,0)
    upperbound_loss_y = torch.max(torch.abs(scale_y)-upperbound,torch.zeros(len(scale_x)).cuda()) # max(|y|-upperbound,0)
    return torch.mean(torch.abs(upperbound_loss_x) + torch.abs(upperbound_loss_y))*weight # mean(|upperbound_loss_x|+|upperbound_loss_y|)*weight

def get_scale_lowerbound_loss(scale, lowerbound, weight):
    scale_x = scale[0] # valid_seq_batch
    scale_y = scale[1] # valid_seq_batch
    lowerbound_loss_x = torch.max(lowerbound-scale_x,torch.zeros(len(scale_x)).cuda()) # max(upperbound-|x|,0)
    lowerbound_loss_y = torch.max(lowerbound-scale_y,torch.zeros(len(scale_x)).cuda()) # max(upperbound-|y|,0)
    return torch.mean(torch.abs(lowerbound_loss_x) + torch.abs(lowerbound_loss_y))*weight # mean(|lowerbound_loss_x|+|lowerbound_loss_y|)*weight    

    
    