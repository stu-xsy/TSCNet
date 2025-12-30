# -*- coding: utf-8 -*-
import io
import os
import dataloader
import os.path
import time
import argparse
from ImbalanceCIFAR import *
import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms
from build_dataset import build_dataset
from utils import *
import itertools
#load environmental settings
import opts
from loss_methods import FocalLoss, LDAMLoss, xERMLoss
from sample_methods import ClassBalanceSampler
opt = opts.opt_algorithm()
from tau_norm import forward, pnorm
# import torch
# torch.backends.cudnn.enabled = False
from ccim import *
#-----------------------------------------------------------------dataset information--------------------------------------------------------------------

opt.dataset = 'CIFAR100'
opt.word_net = 'nn'
data_root = {'ImageNet': '/data4/imagenet/ILSVRC/Data/CLS-LOC',
             'Places': '/data4/Places/places365_standard',
             'CIFAR10': '/CIFAR10',
             'CIFAR100': '/CIFAR100',}

opt.num_cls = 100 # number of classes in the dataset

opt.size_img = [224, 224]
dataset = 'CIFAR100_LT'
splits = ['train', 'test']

cls_num_train,many_index,medium_index,dataloader_train =  dataloader.load_data(data_root=data_root[dataset.rstrip('_LT')],
                                    dataset=dataset, phase='train', 
                                    batch_size=opt.batch_size,
                                    sampler_dic=None,
                                    num_workers=2,
                                    top_k_class=None,
                                    cifar_imb_ratio=opt.cifar_imb_ratio,opt=opt)

cls_num_test,dataloader_test = dataloader.load_data(data_root=data_root[dataset.rstrip('_LT')],
                                    dataset=dataset, phase='test', 
                                    batch_size=opt.batch_size,
                                    sampler_dic=None,
                                    num_workers=2,
                                    top_k_class=None,
                                    cifar_imb_ratio=opt.cifar_imb_ratio,opt=opt)

freq_train = cls_num_train
freq_test = [100 for i in range(100)]

ind_train = [i for i in range(100)]
ind_test = [i for i in range(100)]
dataloadder = {}
dataloadder['train'] = dataloader_train
dataloadder['test'] = dataloader_test
# ind_train, freq_train, ind_test, freq_test = get_cls_freq(opt.dataset)
# cls_num_train, cls_num_test = get_cls_num_list(opt.dataset)
# #--------------------------------------------------------------------settings----------------------------------------------------------------------------

# basic
CUDA = 1  # 1 for True; 0 for False
SEED = 1
measure_best = 0 # best measurement
epoch_best = 0
torch.manual_seed(SEED)
kwargs = {'num_workers': 2, 'pin_memory': True} if CUDA else {}
if CUDA:
    torch.cuda.set_device(0)
    torch.cuda.manual_seed(SEED)
    
# log and model paths
result_path = os.path.join(opt.result_path, para_name(opt))
if not os.path.exists(result_path):
    os.makedirs(result_path)

EPOCHS = 40
#-----------------------------------------------------------------Rebalance------------------------------------------------------------------------------
def reweight_method(opt, epoch):
    if opt.rebalance=='cb_reweight':
        beta = opt.beta_rw
        effective_num = 1.0 - np.power(beta, cls_num_train)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_train)
        per_cls_weights = torch.tensor(per_cls_weights,dtype=torch.float).cuda()
        return per_cls_weights
    elif opt.rebalance=='ldam_drw':
        idx = 1 if epoch // int(opt.epoch_drw) > 0 else 0
        betas = [0, opt.beta_rw]
        # the 2nd stages
        effective_num = 1.0 - np.power(betas[idx], cls_num_train)
        per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_train)
        per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
        per_cls_weights = torch.tensor(per_cls_weights,dtype=torch.float).cuda()
        return per_cls_weights
    else:
        return None

def resample_method(opt, train_set, cls_num_train):
    if opt.rebalance=='cb_resample':
        train_sampler = ClassBalanceSampler(train_set, cls_num_train, opt.beta_rs, result_path, ind_train)
        return train_sampler
    else:
        return None

#-------------------------------------------------------------dataset & dataloader-----------------------------------------------------------------------
if opt.net_v == 'vit':
    transform_img_train = transforms.Compose([
        transforms.Resize([384, 384]),
        transforms.ToTensor(),])
    transform_img_test = transforms.Compose([
        transforms.Resize([384, 384]),
        transforms.ToTensor(),])
elif opt.net_v == 'vit_224':
    transform_img_train = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),])
    transform_img_test = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),])
else:
    transform_img_train = transforms.Compose([
        transforms.Resize([32, 32]),
        transforms.ToTensor(),])
    transform_img_test = transforms.Compose([
        transforms.Resize([32, 32]),
        transforms.ToTensor(),])


# # create dataset
dataset_train = build_dataset(opt, 'train', transform_img_train)
dataset_test = build_dataset(opt, 'test', transform_img_test)

#第一阶段train_sampler为none
train_sampler = resample_method(opt, dataset_train, cls_num_train)

# dataloader
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=32, shuffle=(train_sampler is None),sampler=train_sampler, **kwargs)
# wrn/vgg -> batch_size may not 100
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=64, **kwargs)

#-----------------------------------------------------------------Model----------------------------------------------------------------------------------
# model define
import build_model_prompt
import build_model

#定义confounder
confounder = np.load(f'backdoor/vpt/cifar_ratio0.02_global_dict_256.npy', allow_pickle=True)
confounder = torch.Tensor(confounder)
confounder = torch.stack([x for x in confounder])

#定义先验prob和ccim注意力机制
prob = np.load(f'backdoor/label_128_prior.npy')
prob = torch.from_numpy(prob).reshape(-1, 1)
ccim = CCIM(num_joint_feature=768, num_gz=768, strategy='ad_cause').cuda()
confounder = confounder.cuda()
prob = prob.cuda()

import build_model_prompt
#定义VPT
model = build_model_prompt.build_ccim_model_lprompt(num_classes=100, img_size=[224,224], base_model='vit_base_patch16_224_in21k', model_idx='ViT', patch_size=16,Prompt_Token_num=10, VPT_type="Deep",opt=opt) 
model =model.cuda()

if opt.net_v.endswith('cbam'):
    optimizer, optimizer_finetune = build_model.set_optimizer(model,opt)
if opt.rebalance=='decouple_crt':
    model = build_model.get_updateModel(model, opt.path_pretrain_v)
    optimizer = build_model.set_classifier_optimizer(model,opt)
else:
    optimizer,warm_lr = build_model.set_optimizer_LPT_ccim(model,opt)

# ----------------------------------------------------------------Train----------------------------------------------------------------------------------

def train_epoch(epoch, decay, optimizer, modality):
    
    # vireo measurement
    acc1_v = AverageMeter('Acc@1', ':6.2f')
    acc5_v = AverageMeter('Acc@5', ':6.2f')
    losses = AverageMeter('Loss', ':.4e')
    
    hit_cls = np.zeros(opt.num_cls)

    model.train()
    total_time = time.time()

    for batch_idx, (data, label, indexes) in enumerate(dataloadder['train']):
        start_time = time.time()           
        # prediction and loss
        batch_size_cur = data.size(0)

        if CUDA:
            data = data.cuda()
            label = label.cuda()
        #perform prediction
        output = model(data)

        #compute loss
        weight_cls = reweight_method(opt, epoch)
        if opt.rebalance=='focal':
            focal_loss = FocalLoss(gamma=opt.gamma_focal)
            criterion = nn.CrossEntropyLoss(reduction='none', weight=weight_cls)
            loss_cls_v = focal_loss(output, label, criterion)
        elif opt.rebalance=='ldam_drw':
            ldam_loss = LDAMLoss(cls_num_train, max_m=opt.m_ldam, s=opt.s_ldam)
            criterion = nn.CrossEntropyLoss(reduction='none', weight=weight_cls)
            loss_cls_v = ldam_loss(output, label, criterion)
        else:
            criterion = nn.CrossEntropyLoss(weight=weight_cls)
            loss_cls_v = criterion(output, label)

        final_loss = loss_cls_v

        # optimization
        losses.update(final_loss.item(), batch_size_cur)
        
        optimizer.zero_grad()
        final_loss.backward()
        optimizer.step()
        warm_lr.step(epoch=epoch)
        # upddate log
        [acc_top1, acc_top5], hit_top1 = accuracy_hit(output, label, opt.num_cls, topk=(1, 5))
        hit_cls += hit_top1
        acc1_v.update(acc_top1[0], batch_size_cur)
        acc5_v.update(acc_top5[0], batch_size_cur)
            
        optimizer_cur = optimizer
        log_out = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                  'Time {data_time:.3f}\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {acc1_v.val:.4f} ({acc1_v.avg:.4f})\t'
                  'Acc@5 {acc5_v.val:.4f} ({acc5_v.avg:.4f})'.format(
            epoch, batch_idx, len(dataloadder['train']), data_time=round((time.time() - total_time), 4), loss=losses, acc1_v=acc1_v, acc5_v=acc5_v, lr=optimizer_cur.param_groups[-1]['lr']))
        print(log_out)
        log_train.write(log_out + '\n')
        log_train.flush()
    epoch_acc_cls = hit_cls[ind_train]/freq_train
    ind_head = many_index
    ind_mid = medium_index
    epoch_acc_head = np.sum(hit_cls[ind_train][:ind_head])/np.sum(freq_train[:ind_head])
    epoch_acc_mid = np.sum(hit_cls[ind_train][ind_head:ind_mid])/np.sum(freq_train[ind_head:ind_mid])
    epoch_acc_tail = np.sum(hit_cls[ind_train][ind_mid:])/np.sum(freq_train[ind_mid:])
    log_cls_train.write(
        'Epoch{}:\nacc_top1:{:.4f} acc_top5:{:.4f}\nacc_head(>500):{:.4f} acc_mid(>300):{:.4f} acc_tail:{:.4f}\nacc_cls:{}\nhit_cls:{}\n\n'.format(
        epoch, acc1_v.avg, acc5_v.avg, epoch_acc_head, epoch_acc_mid, epoch_acc_tail, np.around(epoch_acc_cls,4).tolist(),hit_cls[ind_train].tolist()))
    log_cls_train.flush()


# ----------------------------------------------------------------Test----------------------------------------------------------------------------------
def test_epoch(epoch):
    # vireo measurement
    acc1_v = AverageMeter('Acc@1', ':6.2f')
    acc5_v = AverageMeter('Acc@5', ':6.2f')
    losses = AverageMeter('Loss', ':.4e')
  
    hit_cls = np.zeros(opt.num_cls)

    model.eval()
    total_time = time.time()
    with torch.no_grad():
        for batch_idx, (data, label, indexes)  in enumerate(dataloadder['test']):
            start_time = time.time()           

            # prediction and loss
            batch_size_cur = data.size(0)
            if CUDA:
                data = data.cuda()
                label = label.cuda()
            #perform prediction
            #output,_ = model_TDE(data)
            output= model(data)
            #compute loss
            criterion = nn.CrossEntropyLoss()
            loss_cls_v = criterion(output, label)

            final_loss = loss_cls_v

            # optimization
            losses.update(final_loss.item(), batch_size_cur)
            # upddate log
            [acc_top1, acc_top5], hit_top1 = accuracy_hit(output, label, opt.num_cls, topk=(1, 5))
            hit_cls += hit_top1
            acc1_v.update(acc_top1[0], batch_size_cur)
            acc5_v.update(acc_top5[0], batch_size_cur)
    epoch_acc_cls = hit_cls[ind_test]/freq_test
    ind_head = many_index
    ind_mid = medium_index
    epoch_acc_head = np.sum(hit_cls[ind_test][:ind_head])/np.sum(freq_test[:ind_head])
    epoch_acc_mid = np.sum(hit_cls[ind_test][ind_head:ind_mid])/np.sum(freq_test[ind_head:ind_mid])
    epoch_acc_tail = np.sum(hit_cls[ind_test][ind_mid:])/np.sum(freq_test[ind_mid:])
    log_out = 'Epoch{}:\nacc_top1:{:.4f} acc_top5:{:.4f}\nacc_head(>500):{:.4f} acc_mid(>300):{:.4f} acc_tail:{:.4f}\nacc_cls:{}\nhit_cls:{}\n\n'.format(epoch, acc1_v.avg, acc5_v.avg, epoch_acc_head, epoch_acc_mid, epoch_acc_tail, np.around(epoch_acc_cls,4).tolist(), hit_cls[ind_test].tolist())
    log_test.write(log_out)
    print(log_out)
    log_test.flush()
    return acc1_v.avg


def lr_scheduler(epoch, optimizer, lr_decay_iter, decay_rate):
    if not (epoch % lr_decay_iter):
        for i in range(len(optimizer.param_groups)):
            optimizer.param_groups[i]['lr'] = optimizer.param_groups[i]['lr'] * decay_rate


    
if __name__ == '__main__':
    log_train = open(os.path.join(result_path, 'log_train.csv'), 'w')
    log_test = open(os.path.join(result_path, 'log_test.csv'), 'w')
    log_cls_train = open(os.path.join(result_path, 'log_train_cls.csv'),'w')
#     measure_cur = test_epoch(1) 
    for epoch in range(1, EPOCHS):        
        train_epoch(epoch, opt.lr_decay, optimizer, opt.modality)
        measure_cur = test_epoch(epoch)        
        if measure_cur > measure_best:
            torch.save(model.state_dict(), result_path + '/model_best.pt')
            measure_best = measure_cur
            epoch_best = epoch
    
        if epoch == EPOCHS:
            torch.save(model.state_dict(), result_path + '/model_final.pt')

    state = 'Max is achieved on epoch {} with Top1:{}'.format(epoch_best, measure_best)

    log_test.write(state)
    log_test.flush()


