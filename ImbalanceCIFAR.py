"""
Adopted from https://github.com/Megvii-Nanjing/BBN
Customized by Kaihua Tang
"""
import torch
import scipy.io as matio
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import random
import io
import cv2
import time
import numpy as np
from math import sqrt
import torch
import numpy as np

from einops import rearrange
from opt_einsum import contract
from torch.utils.data import Dataset
def gen_afa_from_config(image_size):

    _gen_fourier_aug = GeneralFourierOnline(
        img_size=image_size, groups=range(1, image_size + 1), phases=(0., 1.)
    )
    return _gen_fourier_aug



# 傅里叶变换
def colorful_spectrum_mix(img1, img2, alpha, ratio=1.0):
    """Input image size: ndarray of [H, W, C]"""
    lam = np.random.uniform(0, alpha)

    assert img1.shape == img2.shape
    h, w, c = img1.shape
    h_crop = int(h * sqrt(ratio))
    w_crop = int(w * sqrt(ratio))
    h_start = h // 2 - h_crop // 2
    w_start = w // 2 - w_crop // 2

    img1_fft = np.fft.fft2(img1, axes=(0, 1))
    img2_fft = np.fft.fft2(img2, axes=(0, 1))
    img1_abs, img1_pha = np.abs(img1_fft), np.angle(img1_fft)
    img2_abs, img2_pha = np.abs(img2_fft), np.angle(img2_fft)

    img1_abs = np.fft.fftshift(img1_abs, axes=(0, 1))
    img2_abs = np.fft.fftshift(img2_abs, axes=(0, 1))

    img1_abs_ = np.copy(img1_abs)
    img2_abs_ = np.copy(img2_abs)
    img1_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
        lam * img2_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img1_abs_[
                                                                                          h_start:h_start + h_crop,
                                                                                          w_start:w_start + w_crop]
    img2_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
        lam * img1_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img2_abs_[
                                                                                          h_start:h_start + h_crop,
                                                                                          w_start:w_start + w_crop]

    img1_abs = np.fft.ifftshift(img1_abs, axes=(0, 1))
    img2_abs = np.fft.ifftshift(img2_abs, axes=(0, 1))

    img21 = img1_abs * (np.e ** (1j * img1_pha))
    img12 = img2_abs * (np.e ** (1j * img2_pha))
    img21 = np.real(np.fft.ifft2(img21, axes=(0, 1)))
    img12 = np.real(np.fft.ifft2(img12, axes=(0, 1)))
    img21 = np.uint8(np.clip(img21, 0, 255))
    img12 = np.uint8(np.clip(img12, 0, 255))

    return img21, img12


class test_CIFAR100(Dataset):
    def __init__(self, indices, state, cifar_dataset):
        self.indices = indices
        self.state = state
        self.dataset = cifar_dataset

    def __getitem__(self,idx):
        data, label, _ = self.dataset.get_item(self.indices[idx], self.state[idx], train=False)
        return data, label, self.indices[idx], self.state[idx]
    
    def __len__(self):
        return len(self.indices)

    
class IMBALANCECIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10
    def __init__(self, phase, imbalance_ratio,opt, root = '/gruntdata5/kaihua/datasets',imb_type='exp'):
        train = True if phase == "train" else False
        super(IMBALANCECIFAR10, self).__init__(root, train, transform=None, target_transform=None, download=True)
        self.train = train
        self.weighted_alpha = 1
        self.img_num_lists = []
        self.method = opt.method
        self.net = opt.net_v
        self.txt_file_path = 'all_image_paths.txt'
        self.time = time.time()
        self.image_index = 0
        self.flouer_transform = gen_afa_from_config(224)
        #记得改正

        if self.train:
            img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imbalance_ratio)
            self.img_num_lists = img_num_list
            self.gen_imbalanced_data(img_num_list)
            if opt.net_v == 'vit_224' or opt.net_v == 'swin':
                self.transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
                self.transform_aug = transforms.Compose([
                    transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomApply([
                       transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                     ], p=0.8),
                    transforms.RandomGrayscale(p=0.2),   
                    transforms.Resize(224),
                    transforms.ToTensor(),

                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])        
            elif opt.net_v == 'CLIP' or opt.net_v == 'swin':
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)), 
                    ])
            else:
                self.transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
        else:
            if opt.net_v == 'vit_224' or opt.net_v == 'swin' :
                self.transform = transforms.Compose([
                                 transforms.Resize(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                ])
            elif opt.net_v == 'CLIP' :
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)), 
                    ])
            else:
                self.transform = transforms.Compose([
                                 transforms.Resize(32),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                ])

        self.labels = self.targets
        if self.method == 'insert_bg': 
            img_path_file = 'all_image_paths.txt'

            with io.open(img_path_file, encoding='utf-8') as file:
                path_to_images = file.read().split('\n')[:-1]
            # label 
            for path in range(len(path_to_images)):
                path_to_images[path] = path_to_images[path][:-4]+'back.jpg' 
            self.path_to_images = path_to_images

        
#--------------------------------------new_label-----------------------
        max_mag = 5
        max_ops = 5
        self.min_state = 0
        self.max_state = max(max_mag, max_ops) + 1
        states = torch.arange(self.min_state, self.max_state)
        if self.max_state == 1:
            self.ops = torch.tensor([0])
            self.mag = torch.tensor([0])
            
        elif max_mag > max_ops:
            self.ops = (states * max_ops / max_mag).ceil().int()
            self.mag = states.int()
        else:
            self.mag = (states * max_mag / max_ops).ceil().int()
            self.ops = states.int()
        
        print(f"Magnitude set = {self.mag}")
        print(f"Operation set = {self.ops}")

        self.curr_state = torch.zeros(len(self.data))
        self.score_tmp = torch.zeros((len(self.targets), self.max_state))
        self.num_test = torch.zeros((len(self.targets), self.max_state))
#         self.aug_prob = aug_prob

        print("{} Mode: Contain {} images".format(phase, len(self.data)))
    def get_img_num_lists(self):
        return self.img_num_lists
    def _get_class_dict(self):
        class_dict = dict()
        for i, anno in enumerate(self.get_annotations()):
            cat_id = anno["category_id"]
            if not cat_id in class_dict:
                class_dict[cat_id] = []
            class_dict[cat_id].append(i)
        return class_dict
    def update_scores(self, correct, index, state):

        for s in np.unique(state):
            pos = np.where(state == s)
            score_result = np.bincount(index[pos], correct[pos], len(self.score_tmp))
            num_test_result = np.bincount(index[pos], np.ones(len(index))[pos], len(self.score_tmp))
            self.score_tmp[:,s] += score_result
            self.num_test[:,s] += num_test_result

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)

        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
#             np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets
        
    def get_weighted_sampler(self):
        cls_num_list = self.img_num_lists
        cls_weight = 1.0 / (np.array(cls_num_list) ** self.weighted_alpha)
        cls_weight = cls_weight / np.sum(cls_weight) * len(cls_num_list)
        samples_weight = np.array([cls_weight[t] for t in self.labels])
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()

        print("samples_weight", samples_weight)
        sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(self.labels), replacement=True)
        return sampler
    def get_item(self, index, state, train=True):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        img = self.transform(img)
        return img, target, index
    
    def __getitem__(self, index):
        img1, label = self.data[index], self.labels[index]
        if self.method == 'Floer':
            #在method中手动切换到第二阶段
            random_number = random.randint(1, 9999)
            img2, label2 = self.data[random_number], self.labels[random_number]
#           alpha 可以使用课程学习方案自主调节 0.05*self.curr_state[index].float() 

            imgF,imgF2 = colorful_spectrum_mix(img1, img2, alpha = 1.2, ratio= 0.9)          
            img = Image.fromarray(img1)
            imgF = Image.fromarray(imgF)

            img = self.transform(img)
            imgF = self.transform(imgF)
            return [img,imgF],label,index
        if self.method == 'insert_bg':
            random_number = random.randint(1, 12000)
            img = Image.fromarray(img1)
            if self.train:                
                img_back = Image.open(self.path_to_images[random_number]).convert('RGB')
                img_mix = insert_resized_image(img_back,img)
                img_mix.save("output.png")
                img_mix = self.transform(img_mix)
                img = self.transform(img)
                return [img,img_mix],label,index
            img = self.transform(img)
            return img,label,index
    
        img = Image.fromarray(img1)
        if self.method == 'gradcam':
            if self.net == 'vit_224':
                return cv2.resize(img1, (224, 224)),self.transform(img),label, index
            else :
                return cv2.resize(img1, (224, 224)),self.transform(img),label, index
            
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label, index
    def get_super_class(self):
        return self.coarse_labels
    def __len__(self):
        return len(self.labels)

    def get_num_classes(self):
        return self.cls_num

    def get_annotations(self):
        annos = []
        for label in self.labels:
            annos.append({'category_id': int(label)})
        return annos
        
    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

class IMBALANCECIFAR100(IMBALANCECIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """
    cls_num = 100
    
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
