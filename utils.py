import torch
import torch.nn as nn
import shutil
import os
import numpy as np
import ipdb
from sklearn.metrics import average_precision_score
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from sklearn import manifold
from matplotlib import rcParams
from sklearn.metrics import average_precision_score
#-----------------------------------------process data with feature-----------------------
def data_norm(data, data_max, data_min):

    data_mean = sum(data)/len(data)
    data_std = np.std(data)
    for x in range(len(data)):
        data[x][0] = round((data[x][0] - data_min[0]) / (data_max[0] - data_min[0]), 4)
        data[x][1] = round((data[x][1] - data_min[1]) / (data_max[1] - data_min[1]), 4)
    return data

def plt_feature_MSCN(feature,result_path):
    #, 'head3','tail1','tail2','tail3'
    class_names = ['head1']
    colors = plt.cm.get_cmap('tab20c', len(class_names))

    tsne = manifold.TSNE(n_components=2, init='random', random_state=0)
    feature_head = tsne.fit_transform(feature)
    mmax = np.amax(feature_head, axis=0)
    mmin = np.amin(feature_head, axis=0)
    feature_head = data_norm(feature_head, mmax, mmin)
    for i in range(len(class_names)):
        plt.scatter(feature_head[:, 0], feature_head[:, 1],s=3,color=colors(i))

    plt.title('t-SNE Visualization')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    result_png = result_path + '/tsne_MSCN.png'
#     print(result_png)

    plt.savefig(result_png)
    plt.clf()

def plt_feature_another(feature, labels,result_path):
    #, 'head3','tail1','tail2','tail3'
    class_names = ['tail1', 'tail2', 'tail3', 'tail4', 'tail5','tail6', 'tail7', 'tail8', 'tail9', 'tail10']
#     class_names = ['head1', 'head2','tail']
    colors = plt.cm.get_cmap('tab20c', len(class_names))
    selected_classes = [96,80,84,83,98,97,85,88,89,93] 
#     selected_classes = [12,39,40,85,94,86] 
#     selected_classes = [95,96,98,47,35,30,92,62] 
    feature = feature[np.isin(labels, selected_classes)]
    selected_labels = labels[np.isin(labels, selected_classes)]
    #combined_features = np.vstack((features1, features2))
    tsne = manifold.TSNE(n_components=2, init='random', random_state=0,perplexity=30)
    feature_head = tsne.fit_transform(feature)
    mmax = np.amax(feature_head, axis=0)
    mmin = np.amin(feature_head, axis=0)
    feature_head = data_norm(feature_head, mmax, mmin)
#     for i in range(len(class_names)):
#         class_mask = (selected_labels == selected_classes[i])
#         index = -1
#         for class_m in class_mask:
#             index  = index + 1
#             if class_m == True:
#                 class_m = index 
# #                 if (feature_head[class_m, 1]>2.5) | (feature_head[class_m, 1]<-2.5) | (feature_head[class_m, 1]<-1.5)& (feature_head[class_m, 0]>1.5) |  (feature_head[class_m, 0]<-1.5):
#                 plt.scatter(feature_head[class_m, 0], feature_head[class_m, 1],s=7,color=colors(i))
    for i in range(len(class_names)):
        class_mask = (selected_labels == selected_classes[i])
        plt.scatter(feature_head[class_mask, 0], feature_head[class_mask, 1],s=4, label=class_names[i])

    plt.title('t-SNE Visualization')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    result_png = result_path + '/tsne_tail[96,99,84,83,98,97,85,88,89,91] .png'
#     print(result_png)

    plt.savefig(result_png)
    plt.clf()

def plt_feature_train_test(feature_test, labels_test,feature_train, labels_train,result_path):
    #, 'head3','tail1','tail2','tail3'
    class_names_test = ['test1','test2','test3','test4']
    class_names_train = ['train1','train2','train3','train4']
#     class_names = ['head1', 'head2','tail']
    color = np.stack(labels_test, axis=0)
#     selected_classes = [35,36,38] 42434445
    selected_classes = [2,3,4,5]
#--------------------------feature_test---------------------------
    feature_test = feature_test[np.isin(labels_test, selected_classes)]
    selected_labels_test = labels_test[np.isin(labels_test, selected_classes)]
    #combined_features = np.vstack((features1, features2))
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    feature_head_test = tsne.fit_transform(feature_test)

       
#--------------------------feature_train---------------------------
    feature_train = feature_train[np.isin(labels_train, selected_classes)]
    selected_labels_train = labels_train[np.isin(labels_train, selected_classes)]
    #combined_features = np.vstack((features1, features2))
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    feature_head_train = tsne.fit_transform(feature_train )
    
#--------------------------------plot---------------------------------   

#     mmax = np.amax(feature_head_test, axis=0)
#     mmax2 = np.amax(feature_head_train, axis=0)
#     mmin = np.amin(feature_head_test, axis=0)
#     mmin2 = np.amin(feature_head_train, axis=0)
#     mmax = np.maximum(mmax, mmax2)
#     mmin = np.minimum(mmin, mmin2)
#     feature_head_test = data_norm(feature_head_test, mmax, mmin)
    colors = plt.cm.get_cmap('tab20', len(class_names_test)+12)
    for i in range(len(class_names_test)):
        class_mask = (selected_labels_test == selected_classes[i])
        plt.scatter(feature_head_test[class_mask, 0], feature_head_test[class_mask, 1],s=3,color = colors(i),label = class_names_test[i])

#     feature_head_train  = data_norm(feature_head_train , mmax, mmin)
    for i in range(len(class_names_train)):
        class_mask = (selected_labels_train == selected_classes[i])
        plt.scatter(feature_head_train[class_mask, 0], feature_head_train[class_mask, 1],s=3,color = colors(i+6),label = class_names_train[i])

        
    plt.title('t-SNE Visualization')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    result_png = result_path + '/tsne_tail2,3,4,5.png'
#     print(result_png)

    plt.savefig(result_png)
    plt.clf()

    
    
def plt_feature_another2(feature, labels,result_path):
    class_names =['tail1', 'tail2', 'tail3', 'tail4', 'tail5','tail6', 'tail7', 'tail8', 'tail9', 'tail10']

    color = np.stack(labels, axis=0)

    selected_classes = [96,80,84,83,98,97,85,88,89,93]
    index_count = 0
    index1=[]
    index2 = []
    index3 =[]
    index_all = []
    for label in labels:
        if label in selected_classes:
            index_all.append(index_count)
        if label == 13:            
            index1.append(index_count)
            index_count = index_count+ 1
        elif label ==81:
            index3.append(index_count)
            index_count = index_count+ 1
        elif label == 90:
            index2.append(index_count)
            index_count = index_count+ 1
        else:
            index_count = index_count+ 1
            continue
#     import ipdb
#     ipdb.set_trace()
    feature = feature[np.isin(labels, selected_classes)]
    selected_labels = labels[np.isin(labels, selected_classes)]
    #combined_features = np.vstack((features1, features2))
    tsne = manifold.TSNE(n_components=2, init='random', random_state=0)
    feature_head = tsne.fit_transform(feature)
    mmax = np.amax(feature_head, axis=0)
    mmin = np.amin(feature_head, axis=0)
    feature_head = data_norm(feature_head, mmax, mmin)
    index_count2 =0 
    index_feature = []
    for features in feature_head:

        if features[0] > 0.7 and features[0]< 0.9  and features[1] > 0.1 and features[1] < 0.2:
            index_feature.append(index_count2)
            index_count2 = index_count2 + 1
        else:
            index_count2 = index_count2 + 1
    
    import ipdb
    ipdb.set_trace()
    index_select = [index_all[index] for index in index_feature]
    for i in range(len(class_names)):
        class_mask = (selected_labels == selected_classes[i])
        plt.scatter(feature_head[class_mask, 0], feature_head[class_mask, 1],s=4, label=class_names[i])

#     plt.scatter(feature_head[:, 0], feature_head[:, 1], c=selected_labels, cmap=plt.cm.get_cmap('tab10'))
    plt.title('t-SNE Visualization')                                        
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend(loc='lower left')
    result_png = result_path + '/vit_compare_cifar_ratio_4.png'
#     print(result_png)

    plt.savefig(result_png)
    plt.clf()


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0) # batch_size

        _, pred = output.topk(maxk, 1, True, True) # pred: (batch_size, maxk)
        pred = pred.t() # pred: (maxk, batch_size)
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        results = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            results.append(correct_k.mul_(100.0 / batch_size))
        return results
    


def accuracy_hit(output, target, num_cls, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0) # batch_size

        _, pred = output.topk(maxk, 1, True, True) # pred: (batch_size, maxk)
        pred = pred.t() # pred: (maxk, batch_size)
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        results = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            if k==1:
                hit_k = correct[:k].reshape(-1).float()
                hit_cls = np.zeros(num_cls)
                ind = 0
                for i in range(len(hit_k)):
                    if hit_k[i] == 1:
                        hit_cls[target[i]]+=1
            results.append(correct_k.mul_(1.0 / batch_size))
        return results, hit_cls
    
def AP_matrix(results, gt_labels):
    """Evaluate mAP of a dataset.
    Args:
        results (ndarray): shape (num_samples, num_classes)
        gt_labels (ndarray): ground truth labels of each image, shape (num_samples, num_classes)
        dataset (None or str or list): dataset name or dataset classes, there
            are minor differences in metrics for different datsets, e.g.
            "voc07", "imagenet_det", etc.
        print_summary (bool): whether to print the mAP summary
    Returns:
        tuple: (mAP, [AP_1, AP_2, ..., AP_C]
    """
    # print(results)
    AP = average_precision_score(gt_labels, results, None)

    return AP
    
    
def precision_recall_matrix(predicts, labels, n=10):
    # labels: (64, 81)
    # predicts: (64, 81)
    final_pre = torch.zeros([0]).cuda()
    final_recall = torch.zeros([0]).cuda()
    for cur in range(n):
        pre_topk = predicts.topk(cur+1)[1]
        pre_topk_onehot = torch.zeros(labels.shape).cuda()
        pre_topk_onehot = pre_topk_onehot.scatter_(1,pre_topk,1)

        hit = torch.sum(pre_topk_onehot * labels,dim=1)
        # precision
        precision = hit/(cur+1)
        recall = hit*(1./torch.sum(labels,dim=1))
        final_pre = torch.cat((final_pre, precision.unsqueeze(1)),dim=1)
        final_recall = torch.cat((final_recall, recall.unsqueeze(1)),dim=1)
        
    
    return final_pre, final_recall

    

def precision_recall(predicts, labels, n=10):
    import ipdb
    ipdb.set_trace()
    predicts = predicts.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    # from top to bottom
    sorted_predicts = (-predicts).argsort()
    top_n_inds = sorted_predicts[:, :n]

    # compute top-n hits for each sample
    hit = np.zeros([len(labels), n])
    for i in range(len(labels)):  # for each sample in [0,batch_size-1]
        # calculate the performance from top-1 to top-n
        for j in range(1, n + 1):  # for each value of n in [1,n]
            for k in range(j):  # for each rank of j
                if labels[i, top_n_inds[i, k]] - 1 == 0:
                    hit[i, j - 1] += 1  # j-1 since hit is 0-indexed
    # compute precision
    denominator = np.arange(n) + 1  # 10
    denominator = np.tile(denominator, [len(labels), 1])
    # get precision
    precision = hit / denominator  # (128,10)
    precision[np.isnan(precision)] = 0.
    # compute recall
    # get denominator, the sum of the number of ingre in this recipe

    denominator = np.sum(labels, axis=1)  # (128)

    denominator = np.tile(np.expand_dims(denominator, axis=1), [1, n])  # (128,10)
    
    # get recall
    recall = hit / denominator  # (128,10)
    recall[np.isnan(recall)] = 0.
    return precision, recall


def frozen_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False
        

def para_name(opt):
    if opt.modality == 'v':
        name_para = 'datset={}~net_v={}~method={}~bs={}~decay={}~lr={}~lrd_rate={}'.format(
        opt.dataset,
        opt.net_v,
        opt.method,
        opt.batch_size,
        opt.lr_decay,
        opt.lr,
        opt.lrd_rate
        ) 
        if opt.net_v.startswith('vit'):
            name_para += '~drop={}'.format(opt.vit_drop)
        if opt.rebalance!='none':
            name_para += '~{}'.format(opt.rebalance)
        if opt.rebalance=='focal':
            name_para += '~gamma={}'.format(opt.gamma_focal)
        if opt.rebalance in ['cb_reweight','ldam_drw']:
            name_para += '~beta={}'.format(opt.beta_rw)
        if opt.rebalance in ['cb_resample']:
            name_para += '~beta={}'.format(opt.beta_rs)
        if opt.rebalance.startswith('ldam'):
            name_para += '~m={}~s={}'.format(opt.m_ldam, opt.s_ldam)
        if opt.rebalance.endswith('drw'):
            name_para += '~e_drw={}'.format(opt.epoch_drw)
            
    if opt.modality == 's':
        name_para = 'datset={}~net_s={}~method={}~lr={}~dim_latent={}'.format(
        opt.dataset,
        opt.net_s,
        opt.method,
        opt.lr,
        opt.dim_latent
        )
    if opt.modality == 'v+s':
        name_para = 'datset={}~net_v={}~net_s={}~method={}~lr={}~ws={}~wa={}~type_a={}~dim_latent={}'.format(
        opt.dataset,
        opt.net_v,
        opt.net_s,
        opt.method,
        opt.lr,
        opt.w_semantic,
        opt.w_align,
        opt.type_align,
        opt.dim_latent
        )
        if opt.method in ['slf', 'slf_multi_hot','slf_embedding','slf_mm']:
            name_para+= '~adj={}~topk={}~dim_embed={}'.format(opt.adj,opt.topk, opt.dim_embed)
            if opt.adj.startswith('sl'):
                name_para+= '~pri={}~cls={}'.format(opt.beta_pri, opt.beta_loss_cls)
        if opt.method=='cmi':
            name_para+= '~cmi={}~topk={}'.format(opt.cmi,opt.topk)
        if opt.encoder_finetune:
            name_para+= '~lr_finetune={}'.format(opt.lr_finetune)
        if opt.method.endswith('cmfl'):
            name_para+= '~gamma_fl={}~beta_cmfl={}'.format(opt.gamma_focal, opt.beta_cmfl)
        if opt.method.startswith('vae'):
            name_para += '~beta_vae={}'.format(opt.beta_vae)
        if opt.method == 'vae_cm_disentangle':
            name_para += '~beta_cm_vae={}'.format(opt.beta_cm_vae)
        if opt.method=='pht':
            name_para += '~pht_partial={}'.format(opt.pht_partial)
        if opt.type_align in ['kl_l2', 'kl_ln_l2']:
            name_para += '~wa2={}'.format(opt.w_align2)
    if opt.dataset == 'wide':
        name_para += '~pos_weight={}'.format(opt.pos_weight)
    if opt.ext_str != '':
        name_para +='~{}'.format(opt.ext_str)
    if opt.method == 'TDE':
        name_para += '~alpha={}'.format(opt.alpha)
        name_para += '~gamma={}'.format(opt.gamma)
        name_para += '~num_head={}'.format(opt.num_head)
        name_para += '~tau={}'.format(opt.tau)
    return name_para

class AverageMeter(object):
    
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def get_cls_num_list(dataset):
    if dataset == 'vireo':
#         cls_num_train = [613.0, 592.0, 589.0, 481.0, 598.0, 424.0, 606.0, 498.0, 427.0, 572.0, 430.0, 307.0, 595.0, 508.0, 265.0, 541.0, 211.0, 553.0, 150.0, 353.0, 425.0, 163.0, 563.0, 478.0, 636.0, 602.0, 513.0, 418.0, 289.0, 549.0, 533.0, 526.0, 411.0, 597.0, 553.0, 409.0, 583.0, 598.0, 615.0, 522.0, 493.0, 342.0, 457.0, 611.0, 393.0, 414.0, 238.0, 468.0, 269.0, 448.0, 263.0, 462.0, 576.0, 409.0, 443.0, 480.0, 357.0, 323.0, 574.0, 560.0, 566.0, 512.0, 592.0, 570.0, 353.0, 330.0, 525.0, 328.0, 513.0, 547.0, 564.0, 468.0, 436.0, 381.0, 472.0, 168.0, 535.0, 319.0, 371.0, 468.0, 209.0, 294.0, 374.0, 480.0, 511.0, 434.0, 298.0, 340.0, 288.0, 447.0, 220.0, 197.0, 285.0, 408.0, 148.0, 214.0, 304.0, 183.0, 337.0, 195.0, 334.0, 311.0, 302.0, 355.0, 204.0, 328.0, 327.0, 238.0, 292.0, 186.0, 182.0, 473.0, 366.0, 283.0, 251.0, 337.0, 343.0, 392.0, 325.0, 349.0, 382.0, 347.0, 184.0, 412.0, 370.0, 236.0, 286.0, 460.0, 409.0, 192.0, 244.0, 243.0, 573.0, 456.0, 216.0, 367.0, 114.0, 203.0, 164.0, 175.0, 514.0, 391.0, 302.0, 295.0, 273.0, 511.0, 387.0, 439.0, 372.0, 325.0, 374.0, 439.0, 316.0, 185.0, 118.0, 459.0, 506.0, 120.0, 186.0, 369.0, 367.0, 304.0, 307.0, 469.0, 226.0, 481.0, 315.0, 211.0, 277.0, 227.0, 208.0, 527.0]
        cls_num_train = [601, 489, 478, 230, 549, 127, 574, 241, 132, 427, 136, 43, 512, 252, 28, 339, 20, 372, 13, 70, 129, 14, 389, 210, 630, 561, 282, 124, 34, 355, 324, 309, 115, 524, 363, 108, 467, 536, 615, 296, 235, 62, 167, 588, 103, 121, 25, 183, 29, 159, 28, 179, 457, 110, 152, 219, 73, 48, 447, 380, 407, 270, 501, 417, 68, 55, 302, 53, 276, 347, 398, 191, 142, 92, 200, 14, 332, 47, 84, 187, 20, 36, 90, 214, 258, 139, 37, 61, 33, 156, 22, 18, 32, 105, 13, 21, 40, 15, 59, 18, 57, 44, 39, 71, 19, 54, 52, 25, 35, 16, 15, 205, 75, 31, 27, 58, 63, 101, 49, 66, 94, 65, 16, 118, 82, 24, 32, 174, 113, 17, 26, 26, 436, 163, 22, 78, 12, 19, 14, 15, 289, 98, 38, 36, 30, 264, 96, 149, 86, 50, 88, 145, 46, 16, 12, 171, 246, 13, 17, 80, 76, 41, 42, 196, 23, 225, 45, 21, 30, 23, 19, 317]
        cls_num_test = [307.0, 297.0, 295.0, 241.0, 300.0, 213.0, 303.0, 249.0, 214.0, 287.0, 216.0, 154.0, 298.0, 255.0, 133.0, 271.0, 106.0, 277.0, 76.0, 177.0, 213.0, 82.0, 282.0, 240.0, 319.0, 302.0, 257.0, 210.0, 145.0, 275.0, 267.0, 264.0, 206.0, 299.0, 277.0, 205.0, 292.0, 300.0, 308.0, 262.0, 247.0, 172.0, 229.0, 306.0, 197.0, 208.0, 120.0, 235.0, 135.0, 225.0, 132.0, 232.0, 288.0, 205.0, 222.0, 241.0, 179.0, 162.0, 288.0, 281.0, 284.0, 257.0, 297.0, 285.0, 177.0, 165.0, 263.0, 165.0, 257.0, 274.0, 282.0, 235.0, 219.0, 191.0, 237.0, 84.0, 268.0, 160.0, 186.0, 235.0, 105.0, 148.0, 188.0, 241.0, 256.0, 218.0, 150.0, 171.0, 145.0, 224.0, 111.0, 99.0, 143.0, 205.0, 75.0, 108.0, 153.0, 92.0, 169.0, 98.0, 168.0, 156.0, 152.0, 178.0, 103.0, 165.0, 164.0, 120.0, 147.0, 93.0, 92.0, 237.0, 183.0, 142.0, 126.0, 169.0, 172.0, 197.0, 163.0, 175.0, 192.0, 174.0, 93.0, 207.0, 186.0, 119.0, 144.0, 231.0, 205.0, 96.0, 123.0, 122.0, 287.0, 229.0, 109.0, 184.0, 58.0, 102.0, 83.0, 88.0, 258.0, 196.0, 152.0, 148.0, 137.0, 256.0, 194.0, 220.0, 187.0, 163.0, 188.0, 220.0, 159.0, 93.0, 60.0, 230.0, 254.0, 61.0, 94.0, 185.0, 184.0, 153.0, 154.0, 235.0, 114.0, 241.0, 158.0, 106.0, 139.0, 114.0, 105.0, 264.0]
    elif dataset=='wide':
        cls_num_train = [519.0, 19915.0, 3115.0, 753.0, 2177.0, 2438.0, 167.0, 1363.0, 10283.0, 956.0, 314.0, 1422.0, 1292.0, 31823.0, 316.0, 973.0, 472.0, 510.0, 1485.0, 41.0, 511.0, 633.0, 996.0, 209.0, 5126.0, 1527.0, 383.0, 694.0, 1739.0, 299.0, 13201.0, 317.0, 1028.0, 2262.0, 7945.0, 1068.0, 40.0, 1896.0, 568.0, 3019.0, 2331.0, 6725.0, 29491.0, 1529.0, 8487.0, 839.0, 478.0, 602.0, 253.0, 4724.0, 5512.0, 3721.0, 264.0, 1200.0, 872.0, 43345.0, 3245.0, 93.0, 1081.0, 609.0, 1499.0, 2153.0, 5022.0, 124.0, 275.0, 233.0, 1006.0, 420.0, 1570.0, 1392.0, 1602.0, 543.0, 3242.0, 2268.0, 3497.0, 20694.0, 355.0, 815.0, 195.0, 8681.0, 195.0]
        cls_num_test = [324.0, 13156.0, 2104.0, 507.0, 1521.0, 1546.0, 129.0, 964.0, 7065.0, 637.0, 220.0, 946.0, 911.0, 21243.0, 246.0, 598.0, 353.0, 303.0, 1014.0, 20.0, 316.0, 460.0, 647.0, 170.0, 3401.0, 1108.0, 267.0, 445.0, 1156.0, 203.0, 8884.0, 222.0, 703.0, 1560.0, 5319.0, 721.0, 20.0, 1301.0, 372.0, 2059.0, 1524.0, 4364.0, 19739.0, 1049.0, 5680.0, 545.0, 349.0, 417.0, 167.0, 3108.0, 3691.0, 2546.0, 215.0, 872.0, 569.0, 29314.0, 2131.0, 59.0, 695.0, 382.0, 1017.0, 1474.0, 3382.0, 69.0, 193.0, 181.0, 656.0, 288.0, 1070.0, 893.0, 1071.0, 406.0, 2076.0, 1546.0, 2351.0, 14066.0, 272.0, 533.0, 123.0, 5824.0, 114.0]
    return cls_num_train, cls_num_test

def get_cls_freq(dataset):
    if dataset == 'vireo':
        ind_train = [24, 38, 0, 43, 6, 25, 37, 4, 33, 12, 62, 1, 2, 36, 52, 58, 132, 9, 63, 60, 70, 22, 59, 17, 34, 29, 69, 15, 76, 30, 171, 31, 66, 39, 140, 68, 26, 61, 145, 84, 13, 156, 7, 40, 165, 3, 83, 55, 23, 111, 74, 163, 47, 71, 79, 51, 127, 155, 42, 133, 49, 89, 54, 147, 151, 72, 85, 10, 8, 20, 5, 27, 45, 123, 32, 128, 53, 35, 93, 44, 117, 141, 146, 120, 73, 150, 82, 148, 78, 124, 159, 160, 135, 112, 56, 103, 64, 19, 119, 121, 116, 41, 87, 98, 115, 100, 65, 67, 105, 106, 149, 118, 57, 77, 152, 166, 101, 11, 162, 96, 161, 142, 102, 86, 143, 81, 108, 28, 88, 126, 92, 113, 168, 144, 48, 14, 50, 114, 130, 131, 46, 107, 125, 169, 164, 90, 134, 95, 167, 16, 80, 170, 104, 137, 91, 99, 129, 109, 158, 153, 122, 97, 110, 139, 75, 138, 21, 18, 94, 157, 154, 136]
        freq_train = [636, 615, 613, 611, 606, 602, 598, 598, 597, 595, 592, 592, 589, 583, 576, 574, 573, 572, 570, 566, 564, 563, 560, 553, 553, 549, 547, 541, 535, 533, 527, 526, 525, 522, 514, 513, 513, 512, 511, 511, 508, 506, 498, 493, 481, 481, 480, 480, 478, 473, 472, 469, 468, 468, 468, 462, 460, 459, 457, 456, 448, 447, 443, 439, 439, 436, 434, 430, 427, 425, 424, 418, 414, 412, 411, 409, 409, 409, 408, 393, 392, 391, 387, 382, 381, 374, 374, 372, 371, 370, 369, 367, 367, 366, 357, 355, 353, 353, 349, 347, 343, 342, 340, 337, 337, 334, 330, 328, 328, 327, 325, 325, 323, 319, 316, 315, 311, 307, 307, 304, 304, 302, 302, 298, 295, 294, 292, 289, 288, 286, 285, 283, 277, 273, 269, 265, 263, 251, 244, 243, 238, 238, 236, 227, 226, 220, 216, 214, 211, 211, 209, 208, 204, 203, 197, 195, 192, 186, 186, 185, 184, 183, 182, 175, 168, 164, 163, 150, 148, 120, 118, 114]
        ind_test = [24, 38, 0, 43, 6, 25, 37, 4, 33, 12, 62, 1, 2, 36, 52, 58, 9, 132, 63, 60, 22, 70, 59, 17, 34, 29, 69, 15, 76, 30, 171, 31, 66, 39, 140, 68, 26, 61, 145, 84, 13, 156, 7, 40, 55, 165, 83, 3, 23, 111, 74, 71, 163, 79, 47, 51, 127, 155, 133, 42, 49, 89, 54, 147, 151, 72, 85, 10, 8, 20, 5, 27, 45, 123, 32, 93, 35, 128, 53, 117, 44, 141, 146, 120, 73, 150, 82, 148, 78, 124, 159, 135, 160, 112, 56, 103, 19, 64, 119, 121, 116, 41, 87, 115, 98, 100, 105, 65, 67, 106, 149, 118, 57, 77, 152, 166, 101, 162, 11, 96, 161, 142, 102, 86, 81, 143, 108, 88, 28, 126, 92, 113, 168, 144, 48, 14, 50, 114, 130, 131, 46, 107, 125, 164, 169, 90, 134, 95, 167, 16, 80, 170, 104, 137, 91, 99, 129, 158, 109, 153, 122, 110, 97, 139, 75, 138, 21, 18, 94, 157, 154, 136]
        freq_test = [319, 308, 307, 306, 303, 302, 300, 300, 299, 298, 297, 297, 295, 292, 288, 288, 287, 287, 285, 284, 282, 282, 281, 277, 277, 275, 274, 271, 268, 267, 264, 264, 263, 262, 258, 257, 257, 257, 256, 256, 255, 254, 249, 247, 241, 241, 241, 241, 240, 237, 237, 235, 235, 235, 235, 232, 231, 230, 229, 229, 225, 224, 222, 220, 220, 219, 218, 216, 214, 213, 213, 210, 208, 207, 206, 205, 205, 205, 205, 197, 197, 196, 194, 192, 191, 188, 188, 187, 186, 186, 185, 184, 184, 183, 179, 178, 177, 177, 175, 174, 172, 172, 171, 169, 169, 168, 165, 165, 165, 164, 163, 163, 162, 160, 159, 158, 156, 154, 154, 153, 153, 152, 152, 150, 148, 148, 147, 145, 145, 144, 143, 142, 139, 137, 135, 133, 132, 126, 123, 122, 120, 120, 119, 114, 114, 111, 109, 108, 106, 106, 105, 105, 103, 102, 99, 98, 96, 94, 93, 93, 93, 92, 92, 88, 84, 83, 82, 76, 75, 61, 60, 58]
    elif dataset == 'wide':
        ind_train = [55, 13, 42, 75, 1, 30, 8, 79, 44, 34, 41, 50, 24, 62, 49, 51, 74, 56, 72, 2, 39, 5, 40, 73, 33, 4, 61, 37, 28, 70, 68, 43, 25, 60, 18, 11, 69, 7, 12, 53, 58, 35, 32, 66, 22, 15, 9, 54, 45, 77, 3, 27, 21, 59, 47, 38, 71, 0, 20, 17, 46, 16, 67, 26, 76, 31, 14, 10, 29, 64, 52, 48, 65, 23, 78, 80, 6, 63, 57, 19, 36]
        freq_train = [43345, 31823, 29491, 20694, 19915, 13201, 10283, 8681, 8487, 7945, 6725, 5512, 5126, 5022, 4724, 3721, 3497, 3245, 3242, 3115, 3019, 2438, 2331, 2268, 2262, 2177, 2153, 1896, 1739, 1602, 1570, 1529, 1527, 1499, 1485, 1422, 1392, 1363, 1292, 1200, 1081, 1068, 1028, 1006, 996, 973, 956, 872, 839, 815, 753, 694, 633, 609, 602, 568, 543, 519, 511, 510, 478, 472, 420, 383, 355, 317, 316, 314, 299, 275, 264, 253, 233, 209, 195, 195, 167, 124, 93, 41, 40]
        ind_test = [55, 13, 42, 75, 1, 30, 8, 79, 44, 34, 41, 50, 24, 62, 49, 51, 74, 56, 2, 72, 39, 33, 5, 73, 40, 4, 61, 37, 28, 25, 70, 68, 43, 60, 18, 7, 11, 12, 69, 53, 35, 32, 58, 66, 22, 9, 15, 54, 45, 77, 3, 21, 27, 47, 71, 59, 38, 16, 46, 0, 20, 17, 67, 76, 26, 14, 31, 10, 52, 29, 64, 65, 23, 48, 6, 78, 80, 63, 57, 19, 36]
        freq_test = [29314, 21243, 19739, 14066, 13156, 8884, 7065, 5824, 5680, 5319, 4364, 3691, 3401, 3382, 3108, 2546, 2351, 2131, 2104, 2076, 2059, 1560, 1546, 1546, 1524, 1521, 1474, 1301, 1156, 1108, 1071, 1070, 1049, 1017, 1014, 964, 946, 911, 893, 872, 721, 703, 695, 656, 647, 637, 598, 569, 545, 533, 507, 460, 445, 417, 406, 382, 372, 353, 349, 324, 316, 303, 288, 272, 267, 246, 222, 220, 215, 203, 193, 181, 170, 167, 129, 123, 114, 69, 59, 20, 20]
    return np.array(ind_train), np.array(freq_train), np.array(ind_test), np.array(freq_test)