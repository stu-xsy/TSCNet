# -*- coding: utf-8 -*-
import io
import os
import os.path
import time
import argparse
import torch
import torch.utils.data
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms
from build_dataset import build_dataset
from torch import optim
from utils import *
import itertools
#load environmental settings
import opts
from sklearn.metrics import confusion_matrix
from loss_methods import FocalLoss, LDAMLoss
opt = opts.opt_algorithm()

feature_path = 'result/ratio0.02/resnet152-save-all/feature.npy'
label_path = 'result/ratio0.02/resnet152-save-all/label.npy'
result_path = 'result/ratio0.02/resnet152-save-all/'
features = np.load(feature_path)
labels = np.load(label_path)
plt_feature_another2(features, labels,result_path)


feature_path = 'result/ratio0.01/vit-save-all/feature.npy'
label_path = 'result/ratio0.01/vit-save-all/label.npy'
result_path = 'result/ratio0.01/vit-save-all/'
features = np.load(feature_path)
labels = np.load(label_path)

plt_feature_another(features, labels,result_path)


# feature_path = 'result/ratio0.02/vit-TDE-save/feature.npy'
# label_path = 'result/ratio0.02/vit-TDE-save/label.npy'
# result_path = 'result/ratio0.02/vit-TDE-save/'
# features = np.load(feature_path)
# labels = np.load(label_path)
# import ipdb
# ipdb.set_trace()
# plt_feature_another(features, labels,result_path)

# feature_path = 'result/ratio0.02/ccim-save-all/feature.npy'
# label_path = 'result/ratio0.02/ccim-save-all/label.npy'
# result_path = 'result/ratio0.02/ccim-save-all/'
# features = np.load(feature_path)
# labels = np.load(label_path)
# plt_feature_another(features, labels,result_path)


# feature_path = 'result/ratio0.02/test/lgcam/feature.npy'
# label_path = 'result/ratio0.02/test/lgcam/label.npy'
# result_path = 'result/ratio0.02/test/lgcam/'
# features = np.load(feature_path)
# labels = np.load(label_path)
# plt_feature_another(features, labels,result_path)


# feature_path = 'result/ratio0.02/test/ccim/feature.npy'
# label_path = 'result/ratio0.02/test/ccim/label.npy'
# result_path = 'result/ratio0.02/test/ccim/'
# features = np.load(feature_path)
# labels = np.load(label_path)
# plt_feature_another(features, labels,result_path)
# # feature_path = 'result/ratio0.02/feature/vit/feature.npy'
# # label_path = 'result/ratio0.02/feature/vit/label.npy'
# # result_path = 'result/ratio0.02/vit-save/vit/'
# # features = np.load(feature_path)
# # labels = np.load(label_path)
# # plt_feature_another(features, labels,result_path)


# ratio 0.01
# feature_path = 'result/ratio0.1/ccim_save_all/feature.npy'
# label_path = 'result/ratio0.1/ccim_save_all/label.npy'
# result_path = 'result/ratio0.1/ccim_save_all/'
# features = np.load(feature_path)
# labels = np.load(label_path)

# plt_feature_another(features, labels,result_path)


# feature_path = 'result/ratio0.1/lgcam_save_all/feature.npy'
# label_path = 'result/ratio0.1/lgcam_save_all/label.npy'
# result_path = 'result/ratio0.1/lgcam_save_all/'
# features = np.load(feature_path)
# labels = np.load(label_path)

# plt_feature_another(features, labels,result_path)

# feature_path = 'result/ratio0.1/vit_new/feature.npy'
# label_path = 'result/ratio0.1/vit_new/label.npy'
# result_path = 'result/ratio0.1/vit_new/'
# features = np.load(feature_path)
# labels = np.load(label_path)
# plt_feature_another(features, labels,result_path)


# feature_path = 'result/ratio0.1/vit_new/TDE-save/feature.npy'
# label_path = 'result/ratio0.1/vit_new/TDE-save/label.npy'
# result_path = 'result/ratio0.1/vit_new/TDE-save/'
# features = np.load(feature_path)
# labels = np.load(label_path)
# plt_feature_another(features, labels,result_path)

# feature_path = 'MSCN/cake_feature.npy'

# result_path = 'MSCN/'
# features = np.load(feature_path)
# plt_feature_MSCN(features,result_path)

# feature_path = 'result/ratio0.1/resnet_50_32/TDE-save-all/feature.npy'
# label_path = 'result/ratio0.1/resnet_50_32/TDE-save-all/label.npy'
# result_path = 'result/ratio0.1/resnet_50_32/TDE-save-all/'
# features = np.load(feature_path)
# labels = np.load(label_path)
# plt_feature_another(features, labels,result_path)

# feature_path = 'result/ratio0.1/resnet_50_32/lgcam/save-all/feature.npy'
# label_path = 'result/ratio0.1/resnet_50_32/lgcam/save-all/label.npy'
# result_path = 'result/ratio0.1/resnet_50_32/lgcam/save-all/'
# features = np.load(feature_path)
# labels = np.load(label_path)
# plt_feature_another(features, labels,result_path)
# feature_path = 'result/ratio0.02/feature/vit/feature.npy'
# label_path = 'result/ratio0.02/feature/vit/label.npy'
# result_path = 'result/ratio0.02/vit-save/vit/'
# features = np.load(feature_path)
# labels = np.load(label_path)
# plt_feature_another(features,labels,result_path)

# feature_path_train = 'result/ratio0.02/feature/our_train/feature.npy'
# label_path_train = 'result/ratio0.02/feature/our_train/label.npy'

# feature_path_test = 'result/ratio0.02/feature/our_test/feature.npy'
# label_path_test = 'result/ratio0.02/feature/our_test/label.npy'

# result_path = 'result/ratio0.02/feature/our_train/'

# features_train = np.load(feature_path_train)
# labels_train = np.load(label_path_train)

# features_test = np.load(feature_path_test)
# labels_test = np.load(label_path_test)
# # plt_feature_train_test(features_test, labels_test,features_train,labels_train,result_path)
# plt_feature_another(features_train,labels_train,result_path)
# #------------------------------------base------------------------------------------------------------

# feature_path_train = 'result/ratio0.02/feature/vit_new_train/feature.npy'
# label_path_train = 'result/ratio0.02/feature/vit_new_train/label.npy'

# feature_path_test = 'result/ratio0.02/feature/vit_new_test/feature.npy'
# label_path_test = 'result/ratio0.02/feature/vit_new_test/label.npy'

# result_path = 'result/ratio0.02/feature/vit_new_train/'

# features_train = np.load(feature_path_train)
# labels_train = np.load(label_path_train)

# features_test = np.load(feature_path_test)
# labels_test = np.load(label_path_test)
# # plt_feature_train_test(features_test, labels_test,features_train,labels_train,result_path)
# plt_feature_another(features_train,labels_train,result_path)
# #---------------------------------VPT-----------------------------------

# feature_path_train = 'result/ratio0.02/feature/VPT_train/feature2.npy'
# label_path_train = 'result/ratio0.02/feature/VPT_train/label2.npy'

# feature_path_test = 'result/ratio0.02/feature/VPT_test/feature.npy'
# label_path_test = 'result/ratio0.02/feature/VPT_test/label.npy'

# result_path = 'result/ratio0.02/feature/VPT_train/'

# features_train = np.load(feature_path_train)
# labels_train = np.load(label_path_train)

# features_test = np.load(feature_path_test)
# labels_test = np.load(label_path_test)
# # plt_feature_train_test(features_test, labels_test,features_train,labels_train,result_path)
# plt_feature_another(features_train,labels_train,result_path)



# # #---------------------------------VPT-----------------------------------

# feature_path_train = 'result/ratio0.01/feature/VPT/train/feature.npy'
# label_path_train = 'result/ratio0.01/feature/VPT/train/label.npy'

# feature_path_test = 'result/ratio0.01/feature/VPT/test/feature.npy'
# label_path_test = 'result/ratio0.01/feature/VPT/test/label.npy'

# result_path = 'result/ratio0.01/feature/VPT/train/'

# features_train = np.load(feature_path_train)
# labels_train = np.load(label_path_train)

# features_test = np.load(feature_path_test)
# labels_test = np.load(label_path_test)
# # plt_feature_train_test(features_test, labels_test,features_train,labels_train,result_path)
# plt_feature_another(features_test,labels_test,result_path)


# #--------------------------------------------------ratio0.01-------------------------------------------
# feature_path_train = 'result/ratio0.01/feature/our/train/feature.npy'
# label_path_train = 'result/ratio0.01/feature/our/train/label.npy'

# feature_path_test = 'result/ratio0.01/feature/our/test/feature.npy'
# label_path_test = 'result/ratio0.01/feature/our/test/label.npy'

# result_path = 'result/ratio0.01/feature/our/train/'

# features_train = np.load(feature_path_train)
# labels_train = np.load(label_path_train)

# features_test = np.load(feature_path_test)
# labels_test = np.load(label_path_test)
# # plt_feature_train_test(features_test, labels_test,features_train,labels_train,result_path)
# plt_feature_another(features_test,labels_test,result_path)
# # #------------------------------------base------------------------------------------------------------

# # feature_path_train = 'result/ratio0.01/feature/vit/train/feature.npy'
# # label_path_train = 'result/ratio0.01/feature/vit/train/label.npy'

# feature_path_test = 'result/ratio0.01/feature/vit/feature.npy'
# label_path_test = 'result/ratio0.01/feature/vit/label.npy'

# result_path = 'result/ratio0.01/feature/vit/'

# features_train = np.load(feature_path_train)
# labels_train = np.load(label_path_train)

# features_test = np.load(feature_path_test)
# labels_test = np.load(label_path_test)
# # plt_feature_train_test(features_test, labels_test,features_train,labels_train,result_path)
# plt_feature_another(features_test,labels_test,result_path)


# #---------------------------------LPT-----------------------------------

# feature_path_train = 'result/ratio0.01/feature/LPT_new/feature.npy'
# label_path_train = 'result/ratio0.01/feature/LPT_new/label.npy'

# feature_path_test = 'result/ratio0.01/feature/LPT_new/feature.npy'
# label_path_test = 'result/ratio0.01/feature/LPT_new/label.npy'

# result_path = 'result/ratio0.01/feature/LPT_new/'

# features_train = np.load(feature_path_train)
# labels_train = np.load(label_path_train)

# features_test = np.load(feature_path_test)
# labels_test = np.load(label_path_test)
# # plt_feature_train_test(features_test, labels_test,features_train,labels_train,result_path)
# plt_feature_another(features_test,labels_test,result_path)