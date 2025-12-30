"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os

import torch

import numpy as np

import torch.nn.functional as F
import ipdb


import argparse


# ----------------------------------------------------------------------------------
    
# load weight



def cos_similarity(A, B):
    feat_dim = A.size(1)

    normB = torch.norm(B, 2, 1, keepdim=True)
    B = B / normB
    AB = torch.mm(A, B.t())

    return AB

def linear_classifier(inputs, weights, bias):
    return torch.addmm(bias, inputs, weights.t())

def logits2preds(logits, labels):
    _, nns = logits.max(dim=1)
    preds = np.array([labels[i] for i in nns])
    return preds



def dotproduct_similarity(A, B):
    feat_dim = A.size(1)
    AB = torch.mm(A, B.t())

    return AB

def forward(weights, feat):

    logits = dotproduct_similarity(feat, weights)
    return logits

def pnorm(weights, p):
    normB = torch.norm(weights, 2, 1)
    ws = weights.clone()
   
    for i in range(weights.size(0)):
        ws[i] = ws[i] / torch.pow(normB[i], p)
    return ws

#
# for p in np.linspace(0,2,21):
#     ws = pnorm(weights, p)
#     logits = forward(ws)
#     preds = logits2preds(logits, c_labels)
#     preds2accs(preds, testset, trainset)
