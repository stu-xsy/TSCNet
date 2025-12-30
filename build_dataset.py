import io
import scipy.io as matio
import numpy as np
from PIL import Image
import torch
import random
import torch.utils.data
import ipdb
from math import sqrt
import torchvision.transforms as transforms
def default_loader(image_path):
    return Image.open(image_path).convert('RGB')



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


class dataset_stage1(torch.utils.data.Dataset):
    def __init__(self, opt, mode, transform=None):
        self.mode = mode
        # load image paths
        if  self.mode == 'train':
            list_image = 'TR'
        elif self.mode == 'test':
            list_image = 'TE'
        if opt.dataset == 'vireo':
            cls_num_train = [601, 489, 478, 230, 549, 127, 574, 241, 132, 427, 136, 43, 512, 252, 28, 339, 20, 372, 13, 70, 129, 14, 389, 210, 630, 561, 282, 124, 34, 355, 324, 309, 115, 524, 363, 108, 467, 536, 615, 296, 235, 62, 167, 588, 103, 121, 25, 183, 29, 159, 28, 179, 457, 110, 152, 219, 73, 48, 447, 380, 407, 270, 501, 417, 68, 55, 302, 53, 276, 347, 398, 191, 142, 92, 200, 14, 332, 47, 84, 187, 20, 36, 90, 214, 258, 139, 37, 61, 33, 156, 22, 18, 32, 105, 13, 21, 40, 15, 59, 18, 57, 44, 39, 71, 19, 54, 52, 25, 35, 16, 15, 205, 75, 31, 27, 58, 63, 101, 49, 66, 94, 65, 16, 118, 82, 24, 32, 174, 113, 17, 26, 26, 436, 163, 22, 78, 12, 19, 14, 15, 289, 98, 38, 36, 30, 264, 96, 149, 86, 50, 88, 145, 46, 16, 12, 171, 246, 13, 17, 80, 76, 41, 42, 196, 23, 225, 45, 21, 30, 23, 19, 317]         
            img_path_file = opt.path_data + list_image + '.txt'
            with io.open(img_path_file, encoding='utf-8') as file:
                path_to_images = file.read().split('\n')[:-1]
            # label 
            new_path_to_images = [] 
            label_number_new = np.zeros(172)
            img_label = []
            new_img_label = []
            for path in path_to_images:
                label_str = path.split('/')[1]
                label_tmp = int(label_str)-1
                img_label.append(label_tmp)
                if label_number_new[label_tmp] == cls_num_train[label_tmp]:
                    continue
                new_path_to_images.append(path)
                label_number_new[label_tmp] += 1
                new_img_label.append(label_tmp)
            new_img_label = torch.tensor(new_img_label)
            img_label = torch.tensor(img_label)

        elif opt.dataset == 'backdoor':
        
            img_path_file = 'all_image_paths.txt'
            
            with io.open(img_path_file, encoding='utf-8') as file:
                path_to_images = file.read().split('\n')[:-1]
            # label 
            for path in range(len(path_to_images)):
                path_to_images[path] = path_to_images[path][:-4]+'back.jpg'
                
            self.img_label = []
            for path in path_to_images:
                label_str = path.split('/')[2]
                label_tmp = int(label_str)
                self.img_label.append(label_tmp)
            
            self.img_label = torch.tensor(self.img_label)            

        elif opt.dataset == 'wide':
            img_path_file = opt.path_data + list_image
            if self.mode == 'train':
                img_path_label = opt.path_data + 'NUS_train_labels'#+'.npy' # nus-wide-128
            elif self.mode == 'test':
                img_path_label = opt.path_data + 'NUS_test_labels'#+'.npy' # nus-wide-128
            # load as npy
            #path_to_label = np.load(img_path_label) 
            with io.open(img_path_label, encoding='utf-8') as file:
                path_to_label = file.read().split('\n')[:-1]
            self.img_label = []
            for path in path_to_label:
                label_str = path.split(' ')
                label_str.pop()
                self.img_label.append([float(i) for i in label_str])                
            self.img_label = torch.tensor(self.img_label)
                
            with io.open(img_path_file, encoding='utf-8') as file:
                path_to_images = file.read().split('\n')[:-1]

        self.dataset = opt.dataset
        self.image_path =opt.path_img
#         self.img_label = img_label
        self.path_to_images = path_to_images
        if  self.mode == 'train':

            self.img_label = new_img_label
            self.labels = new_img_label
            self.path_to_images = new_path_to_images
        else:
            self.img_label = img_label
            
#             self.path_to_images = path_to_images
        self.transform = transform
        if opt.dataset == 'backdoor':
             self.transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
#             self.transform = transforms.Compose([
#             transforms.RandomCrop(32, padding=4),
#             transforms.RandomHorizontalFlip(),
#             transforms.Resize(224),
#             transforms.ToTensor(),
#             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#             ])            
        self.loader = default_loader

        # import ipdb; ipdb.set_trace()
    def __getitem__(self, index):
        # get image matrix and transform to tensor
        path = self.path_to_images[index]

        if self.dataset == 'vireo':
#             index2 = random.randint(1, 20000)
#             path2 = self.path_to_images[index2]

#             img1 = self.loader(self.image_path + path)
#             label = self.img_label[index]
            
#             img2 = self.loader(self.image_path + path2)
#             label2 = self.img_label[index2]
            
#             img1 = np.array(img1)
#             img2 = np.array(img2)
            
#             imgF,imgF2 = colorful_spectrum_mix(img1, img2, 0.8, ratio= 0.81)
            
#             img1 = Image.fromarray(img1)
#             imgF = Image.fromarray(imgF)

#             if self.transform is not None:
#                 img = self.transform(img1)
#                 imgF = self.transform(imgF)
#             return img,imgF, label   

            img = self.loader(self.image_path + path)
            label = self.img_label[index]
            if self.transform is not None:
                img = self.transform(img)
            return img,label 

        elif self.dataset == 'backdoor':
            img = self.loader(path)
            label = self.img_label[index]
#             label_str = path.split('/')[1]
#             label = int(label_str)-1
        elif self.dataset == 'wide':
            img = self.loader(self.image_path + path)
            # nus-wide-128
            #label = torch.tensor(self.img_label[index],dtype=torch.float)
#             label_str = self.img_label[index].split(' ')
#             label_str.pop()
#             label = torch.tensor([float(i) for i in label_str])
            label = self.img_label[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.path_to_images)


class dataset_stage2(torch.utils.data.Dataset):
    def __init__(self, opt, mode):        
        # get labels
        if mode == 'train':
            list_image = 'TR'
        elif mode == 'test':
            list_image = 'TE'

        if opt.dataset == 'vireo':
            img_path_file = opt.path_data + list_image + '.txt'
        elif opt.dataset == 'wide':
            img_path_file = opt.path_data + list_image
            if mode == 'train':
                img_path_label = opt.path_data + 'NUS_train_labels'#+'.npy' # nus-wide-128
            elif mode == 'test':
                img_path_label = opt.path_data + 'NUS_test_labels'#+'.npy' # nus-wide-128
            # load as npy
            # path_to_label = np.load(img_path_label) 
            with io.open(img_path_label, encoding='utf-8') as file:
                path_to_label = file.read().split('\n')[:-1]
            self.img_label = path_to_label
        with io.open(img_path_file, encoding='utf-8') as file:
            path_to_images = file.read().split('\n')[:-1]

        if opt.dataset == 'vireo':
            if mode == 'train':
                # words -> 由 0-1 组成的
                words = matio.loadmat(opt.path_data + 'ingredient_train_feature.mat')['ingredient_train_feature']
                # indexVecotrs -> word vector 文件
                indexVectors = matio.loadmat(opt.path_data + 'indexVector_train.mat')['indexVector_train']
            elif mode == 'test':
                words = matio.loadmat(opt.path_data + 'ingredient_test_feature.mat')['ingredient_test_feature']
                indexVectors = matio.loadmat(opt.path_data + 'indexVector_test.mat')['indexVector_test']
        elif opt.dataset == 'wide':
#             import ipdb; ipdb.set_trace()
            if mode == 'train':
                words = np.load(opt.path_data + 'new_train_tags.npy')
                indexVectors = np.load(opt.path_data + 'np_train_word_vector.npy')                    
            elif mode == 'test':
                words = np.load(opt.path_data + 'new_test_tags.npy')
                indexVectors = np.load(opt.path_data + 'np_test_word_vector.npy')
              
        words = words.astype(np.float32)
        indexVectors = indexVectors.astype(np.long)
        self.words = words
        self.indexVectors = indexVectors
        self.path_to_images = path_to_images
        self.dataset = opt.dataset
        
    def __getitem__(self, index):

        # get ingredient vector
        words = self.words[index, :]
        # get index vector for gru input
        indexVector = self.indexVectors[index, :]
        
        # get label
        if self.dataset == 'vireo':     
            path = self.path_to_images[index]
            label_str = path.split('/')[1]
            label = int(label_str)-1      
        elif self.dataset == 'wide':
            #img = self.loader(self.image_path + path)
            # nus-wide-128
            #label = torch.tensor(self.img_label[index],dtype=torch.float)
            label_str = self.img_label[index].split(' ')
            label_str.pop()
            label = torch.tensor([float(i) for i in label_str])

        return ([indexVector, words],label)

    def __len__(self):
        return len(self.indexVectors)



class dataset_stage3(torch.utils.data.Dataset):
    def __init__(self, opt, mode, transform):
        # image channel data
        if mode == 'train':
            list_image = 'TR'
        elif mode == 'test':
            list_image = 'TE'
                
        if opt.dataset == 'vireo':
            img_path_file = opt.path_data + list_image +'.txt'
        elif opt.dataset == 'wide':
            # ipdb.set_trace()
            img_path_file = opt.path_data + list_image
            if mode == 'train':
                img_path_label = opt.path_data + 'NUS_train_labels'
            elif mode == 'test':
                img_path_label = opt.path_data + 'NUS_test_labels'
            with io.open(img_path_label, encoding='utf-8') as file:
                path_to_label = file.read().split('\n')[:-1]
            self.img_label = []
            for path in path_to_label:
                label_str = path.split(' ')
                label_str.pop()
                self.img_label.append([float(i) for i in label_str])                
            self.img_label = torch.tensor(self.img_label)

        # word channel data
        if opt.dataset == 'vireo':
            if mode == 'train':
                words = matio.loadmat(opt.path_data + 'ingredient_train_feature.mat')['ingredient_train_feature']
                indexVectors = matio.loadmat(opt.path_data + 'indexVector_train.mat')['indexVector_train']
            elif mode == 'test':
                words = matio.loadmat(opt.path_data + 'ingredient_test_feature.mat')['ingredient_test_feature']
                indexVectors = matio.loadmat(opt.path_data + 'indexVector_test.mat')['indexVector_test']
        elif opt.dataset == 'wide':
            if mode == 'train':
                words = np.load(opt.path_data + 'new_train_tags.npy')
                indexVectors = np.load(opt.path_data + 'np_train_word_vector.npy') 
            elif mode == 'test':
                words = np.load(opt.path_data + 'new_test_tags.npy')
                indexVectors = np.load(opt.path_data + 'np_test_word_vector.npy')
        elif opt.dataset == 'food101':
            if mode == 'train':
                img_path_file = opt.path_data + 'train_images.txt'
                with io.open(opt.path_data + 'train_labels.txt', encoding='utf-8') as file:
                    labels = file.read().split('\n')[:-1]
                words = matio.loadmat(opt.path_data + 'ingredient_all_feature.mat')['ingredient_all_feature']
                indexVectors = matio.loadmat(opt.path_data + 'ingredient_all_feature.mat')['ingredient_all_feature']
            elif mode == 'test':
                img_path_file = opt.path_data + 'test_images.txt'
                with io.open(opt.path_data + 'test_labels.txt', encoding='utf-8') as file:
                    labels = file.read().split('\n')[:-1]
                words = matio.loadmat(opt.path_data + 'ingredient_all_feature.mat')['ingredient_all_feature']   
            self.img_label = np.array(labels, dtype=int)
        with io.open(img_path_file, encoding='utf-8') as file:
            path_to_images = file.read().split('\n')[:-1]
        
        # ipdb.set_trace()
        self.dataset = opt.dataset
        self.path_img = opt.path_img
        
        self.path_to_images = path_to_images
        self.transform = transform
        self.loader = default_loader
        words = words.astype(np.float32)
        
        self.words = words
        if opt.dataset!='food101':
            indexVectors = indexVectors.astype(np.long)
            self.indexVectors = indexVectors

    def __getitem__(self, index):
        # get image matrix and transform to tensor
        path = self.path_to_images[index]
        if self.dataset == 'vireo':
            img = self.loader(self.path_img + path)
            label_str = path.split('/')[1]
            label = int(label_str)-1
        elif self.dataset == 'wide':
            img = self.loader(self.path_img + path)
            label = self.img_label[index]
        elif self.dataset == 'food101':
            img = self.loader(self.path_img + path + '.jpg')
            label = self.img_label[index]
        if self.transform is not None:
            img = self.transform(img)
        
        # get ingredient vector
        
        # get index vector for gru input
        if self.dataset != 'food101':
            words = self.words[index, :]
            indexVector = self.indexVectors[index, :]
            return [img, indexVector, words], label
        else:
            words = self.words[label]
            return [img, words], label
    
    def __len__(self):
        return len(self.path_to_images)



def build_dataset(opt, mode, transform):

    if opt.modality=='v': #to pretrain visual channel
        dataset = dataset_stage1(opt, mode, transform)
    elif opt.modality=='s': #to pretrain semantic channel
        dataset = dataset_stage2(opt, mode)
    elif opt.modality=='v+s': #to pretrain image channel
        dataset = dataset_stage3(opt, mode, transform)
    else:
        assert 1 < 0, 'Please fill the correct train stage!'

    return dataset