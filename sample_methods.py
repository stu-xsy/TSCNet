import torch
import numpy as np
from collections import Counter
import os
import random
import random
import numpy as np
from torch.utils.data.sampler import Sampler

class ClassBalanceSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, label_to_count, beta, result_path, ind_train, indices=None, num_samples=None):

        self.dataset = dataset
        self.indices = list(range(len(dataset))) \
            if indices is None else indices
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        self.label_to_count = label_to_count
        effective_num = 1.0 - np.power(beta, label_to_count)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)

        # weight for each sample
        # single-label
        if dataset.labels[0].shape==torch.Size([]):
            weights = [per_cls_weights[self._get_label(dataset, idx)]
                       for idx in self.indices]
        else:
            # multi-label
            weights = self._multi_label_weight(dataset,per_cls_weights)
            
        
        
        self.weights = torch.DoubleTensor(weights)
        
        self.ind_train = ind_train
        
    def _get_label(self, dataset, idx):
        return dataset.labels[idx]
    
    def _multi_label_weight(self, dataset, per_cls_weights):
        label_dataset = dataset.img_label
        weight_sum = torch.mm(label_dataset.double(),torch.tensor(per_cls_weights).unsqueeze(1))
        label_weight = 1.0/torch.sum(label_dataset, dim=1)
        weights = weight_sum.squeeze(1)*label_weight.double()
        return weights
        
    def __iter__(self):
        data_sampled = torch.multinomial(self.weights, self.num_samples, replacement=True).tolist()
        if self.dataset.labels[0].shape==torch.Size([]):
            samplecount = np.array(sample_count(self.dataset, self.label_to_count, data_sampled))

        else:
            class_list = torch.sum(self.dataset.labels[torch.tensor(data_sampled)],dim=0)[self.ind_train]
            print(class_list)

        return iter(data_sampled)

    def __len__(self):
        return self.num_samples
    
def sample_count(dataset, label_to_count, data_sampled):
    num_class = len(label_to_count)
    labels = np.array(dataset.labels)
    class_list = [0] * num_class
    class_count = Counter(labels[data_sampled])
    for c in class_count:
        class_list[c] += class_count[c]
    print(class_list)
    return class_list

class RandomCycleIter:
    
    def __init__ (self, data, test_mode=False):
        self.data_list = list(data)
        self.length = len(self.data_list)
        self.i = self.length - 1
        self.test_mode = test_mode
        
    def __iter__ (self):
        return self
    
    def __next__ (self):
        self.i += 1
        
        if self.i == self.length:
            self.i = 0
            if not self.test_mode:
                random.shuffle(self.data_list)
            
        return self.data_list[self.i]
    
def class_aware_sample_generator (cls_iter, data_iter_list, n, num_samples_cls=1):

    i = 0
    j = 0
    while i < n:
        if j >= num_samples_cls:
            j = 0
    
        if j == 0:
            temp_tuple = next(zip(*[data_iter_list[next(cls_iter)]]*num_samples_cls))
            yield temp_tuple[j]
        else:
            yield temp_tuple[j]
        
        i += 1
        j += 1

class ClassAwareSampler (Sampler):
    
    def __init__(self, data_source, num_samples_cls=1,):
        num_classes = len(np.unique(data_source.labels))
        self.class_iter = RandomCycleIter(range(num_classes))
        cls_data_list = [list() for _ in range(num_classes)]
        for i, label in enumerate(data_source.labels):
            cls_data_list[label].append(i)
        self.data_iter_list = [RandomCycleIter(x) for x in cls_data_list]
        self.num_samples = max([len(x) for x in cls_data_list]) * len(cls_data_list) 
        self.num_samples_cls = num_samples_cls
      
    def __iter__ (self):
        return class_aware_sample_generator(self.class_iter, self.data_iter_list,
                                            self.num_samples, self.num_samples_cls)
    
    def __len__ (self):
        return self.num_samples
