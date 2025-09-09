from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import os
import torch
# from torchnet.meter import AUCMeter
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import pickle
import clip
            
def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict

class cifar_dataset(Dataset): 
    def __init__(self, dataset, r, noise_mode, root_dir, transform, mode, noise_type, 
            noise_file='', pred=[], probability=[], unlearning_mask = [], transform_c=[]): 
        
        self.r = r # noise ratio
        self.transform = transform
        self.transform_c = transform_c
        self.mode = mode  
        self.noise_mode = noise_mode
        self.noise_type = noise_type
        if dataset == 'cifar10':
            self.noise_path = '%s/CIFAR-10_human.pt' % root_dir
        else:
            self.noise_path = '%s/CIFAR-100_human.pt' % root_dir
        # class transition for asymmetric noise
        self.transition = {0:0,2:0,4:7,7:7,1:1,9:1,3:5,5:3,6:6,8:8} 
     
        if self.mode=='test':
            if dataset=='cifar10':                
                test_dic = unpickle('%s/test_batch'%root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))  
                self.test_label = test_dic['labels']
            elif dataset=='cifar100':
                test_dic = unpickle('%s/test'%root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))  
                self.test_label = test_dic['fine_labels']                            
        else:    
            train_data=[]
            train_label=[]
            if dataset=='cifar10': 
                for n in range(1,6):
                    dpath = '%s/data_batch_%d'%(root_dir,n)
                    data_dic = unpickle(dpath)
                    train_data.append(data_dic['data'])
                    train_label = train_label+data_dic['labels']
                train_data = np.concatenate(train_data)
            elif dataset=='cifar100':    
                train_dic = unpickle('%s/train'%root_dir)
                train_data = train_dic['data']
                train_label = train_dic['fine_labels']
            train_data = train_data.reshape((50000, 3, 32, 32))
            train_data = train_data.transpose((0, 2, 3, 1))
            self.train_label = train_label
            
            if os.path.exists(noise_file):
                noise_label = json.load(open(noise_file,"r"))
            else:    #inject noise   
                if noise_mode=='sym' or noise_mode =='asym':
                    noise_label = []
                    idx = list(range(50000))
                    random.shuffle(idx)
                    num_noise = int(self.r*50000)            
                    noise_idx = idx[:num_noise]
                    for i in range(50000):
                        if i in noise_idx:
                            if noise_mode=='sym':
                                if dataset=='cifar10': 
                                    noiselabel = random.randint(0,9)
                                elif dataset=='cifar100':    
                                    noiselabel = random.randint(0,99)
                                noise_label.append(noiselabel)
                            elif noise_mode=='asym':   
                                noiselabel = self.transition[train_label[i]]
                                noise_label.append(noiselabel)                    
                        else:    
                            noise_label.append(train_label[i])   
                    print("save noisy labels to %s ..."%noise_file)        
                    json.dump(noise_label,open(noise_file,"w"))  
                    # Add CIFAR-N support
                elif  noise_mode=='cifarn':
                    if noise_type != 'clean':
                        train_noisy_labels = self.load_label()
                        train_noisy_labels = train_noisy_labels.tolist()
         
                    noise_label = train_noisy_labels
            
            if self.mode == 'all' or self.mode == 'labels_zero' or self.mode=='warmup_c':
                self.train_data = train_data
                self.noise_label = noise_label
            else:                   
                if self.mode == "labeled":
                    if len(unlearning_mask) != 0:
                        unlearning_mask = unlearning_mask.numpy()
                        pred_idx = (pred \
                                & (~unlearning_mask)).nonzero()[0]
                    else:
                        pred_idx = pred.nonzero()[0] 
                    
                    self.probability = [probability[i] for i in pred_idx]   
                    # Prepare for index acquisition
                    self.pred_idx = pred_idx
                    clean = (np.array(noise_label)==np.array(train_label))                                                       
                    p = precision_score(clean, pred)
                    recall = recall_score(clean, pred)
                    roc_auc = roc_auc_score(clean[pred_idx], self.probability) 
                    roc_auc_p = roc_auc_score(clean, probability) 

                elif self.mode == "unlabeled":
                    if len(unlearning_mask) != 0:
                        unlearning_mask = unlearning_mask.numpy()
                        pred_idx = ((1 - pred) \
                                & (~unlearning_mask)).nonzero()[0]
                    else:
                        pred_idx = (1-pred).nonzero()[0]  
                    # Prepare for index acquisition
                    self.pred_idx = pred_idx
                
                elif self.mode == "unlearning":                                             
                    pred_idx = unlearning_mask.nonzero(as_tuple=True)[0]

                self.train_data = train_data[pred_idx]
                self.noise_label = [noise_label[i] for i in pred_idx]                          
                print("%s data has a size of %d"%(self.mode,len(self.noise_label)))   
        
    def load_label(self):
        noise_label = torch.load(self.noise_path)
        if isinstance(noise_label, dict):
            if "clean_label" in noise_label.keys():
                clean_label = torch.tensor(noise_label['clean_label'])
            return noise_label[self.noise_type].reshape(-1)
        else:
            raise Exception('Input Error')         
                
    def __getitem__(self, index):
        if self.mode=='labeled':
            img, target, prob, idx =\
                 self.train_data[index], self.noise_label[index],\
                    self.probability[index], self.pred_idx[index]
            img = Image.fromarray(img)
            img1 = self.transform(img) 
            img2 = self.transform(img) 
            img_c1 = self.transform_c(img) 
            img_c2 = self.transform_c(img) 
            return img1, img2, img_c1, img_c2, target, prob, idx            
        elif self.mode=='unlabeled':
            img, idx = self.train_data[index], self.pred_idx[index]
            img = Image.fromarray(img)
            img1 = self.transform(img) 
            img2 = self.transform(img) 
            img_c1 = self.transform_c(img) 
            img_c2 = self.transform_c(img) 
            return img1, img2, img_c1, img_c2, idx
        elif self.mode=='all':
            img, target, clean_labels =\
                  self.train_data[index], self.noise_label[index], self.train_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)   
            return img, target, index, clean_labels     
        elif self.mode=='test':
            img, target = self.test_data[index], self.test_label[index]
            img = Image.fromarray(img)
            img_r = self.transform(img)  
            img_c = self.transform_c(img)            
            return img_r, img_c, target
        elif self.mode=='labels_zero':
            img, target, clean_labels =\
                  self.train_data[index], self.noise_label[index], self.train_label[index]
            img = Image.fromarray(img)  
            img = self.transform(img)
            return img, target, index, clean_labels
        elif self.mode=='unlearning':
            img, target =\
                  self.train_data[index], self.noise_label[index]
            img = Image.fromarray(img)  
            img = self.transform(img)
            return img, target  
        elif self.mode=='warmup_c':
            img, target = \
                self.train_data[index], self.noise_label[index]
            img = Image.fromarray(img)
            img1 = self.transform(img) 
            img2 = self.transform_c(img)          
            return img1, img2, target      
           
    def __len__(self):
        if self.mode!='test':
            return len(self.train_data)
        else:
            return len(self.test_data)         
        
        
class cifar_dataloader():  
    def __init__(self, dataset, r, noise_mode, batch_size, num_workers,
        root_dir, noise_type, noise_file=''):
        self.dataset = dataset
        self.r = r
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.noise_file = noise_file
        self.noise_type = noise_type
        self.noise_mode = noise_mode
        # Class names for CLIP
        if dataset=='cifar10':
            with open(os.path.join(root_dir, 'batches.meta'), 'rb') as f:
                data = pickle.load(f, encoding='latin1')  
            self.label_names = data['label_names']
        if dataset=='cifar100':
            with open(os.path.join(root_dir, 'meta'), 'rb') as f:
                data = pickle.load(f, encoding='latin1') 
            self.label_names = data['fine_label_names']
        # For CLIP preprocessing
        _, self.clip_transform = clip.load("ViT-B/32", device='cuda')
        SA_data_augmentation = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomSolarize(threshold=128, p=0.5),
        ])
        WA_data_augmentation = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomCrop(224, padding=28),
            transforms.RandomHorizontalFlip(),
        ])       
        self.transform_ctest = transforms.Compose([
                self.clip_transform,         
                ])
        self.transform_cSA = transforms.Compose([
                SA_data_augmentation, 
                self.clip_transform,         
                ])
        self.transform_cWA = transforms.Compose([
                WA_data_augmentation, 
                self.clip_transform,         
                ])
        if self.dataset=='cifar10':
            self.transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                ]) 
            self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                ])    
        elif self.dataset=='cifar100':    
            self.transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ]) 
            self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ])              
    def run(self,mode,pred=[],prob=[],unlearning_mask=[],unlearning_batch_size=[]):
        if mode=='warmup':
            all_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, 
                                        noise_type = self.noise_type, r=self.r, root_dir=self.root_dir, 
                                        transform=self.transform_train, 
                                        mode="all",noise_file=self.noise_file)                
            trainloader = DataLoader(
                dataset=all_dataset, 
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers)             
            return trainloader
        
        elif mode=='warmup_vit':
            all_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, 
                                        noise_type = self.noise_type, r=self.r, root_dir=self.root_dir, 
                                        transform=self.transform_cWA,transform_c=self.transform_cSA, 
                                        mode="warmup_c",noise_file=self.noise_file)                
            trainloader = DataLoader(
                dataset=all_dataset, 
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers)             
            return trainloader
                                     
        elif mode=='train':
            labeled_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, 
                                            noise_type = self.noise_type, r=self.r, root_dir=self.root_dir, 
                                            transform=self.transform_train, 
                                            transform_c=self.transform_cWA, mode="labeled",
                                            noise_file=self.noise_file, pred=pred, 
                                            probability=prob, unlearning_mask=unlearning_mask)              
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                drop_last=True)   
            
            unlabeled_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode,
                                             noise_type = self.noise_type, r=self.r, root_dir=self.root_dir, 
                                             transform=self.transform_train, 
                                             transform_c=self.transform_cWA, mode="unlabeled", 
                                             noise_file=self.noise_file, pred=pred, 
                                             unlearning_mask=unlearning_mask)                    
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                drop_last=True)     
            return labeled_trainloader, unlabeled_trainloader

        elif mode=='only_clean':
            labeled_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, 
                                            noise_type = self.noise_type, r=self.r, root_dir=self.root_dir, 
                                            transform=self.transform_cWA, 
                                            transform_c=self.transform_cSA, mode="labeled",
                                            noise_file=self.noise_file, pred=pred, 
                                            probability=prob, unlearning_mask=unlearning_mask)              
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset, 
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers,
                drop_last=True)   
            return labeled_trainloader
        
        elif mode=='test':
            test_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, 
                                         noise_type = self.noise_type, r=self.r, root_dir=self.root_dir, 
                                         transform=self.transform_test, 
                                         transform_c=self.transform_ctest, mode='test')      
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return test_loader
        
        elif mode=='eval_train':
            eval_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, 
                                         noise_type = self.noise_type, r=self.r, root_dir=self.root_dir, 
                                         transform=self.transform_test, 
                                         mode='all', noise_file=self.noise_file)      
            eval_loader = DataLoader(
                dataset=eval_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return eval_loader   
        
        elif mode=='eval_vit':
            eval_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, 
                                         noise_type = self.noise_type, r=self.r, root_dir=self.root_dir, 
                                         transform=self.transform_ctest, 
                                         mode='all', noise_file=self.noise_file)      
            eval_loader = DataLoader(
                dataset=eval_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return eval_loader   
             
        elif mode=='labels_zero_shot':
            labels_dataset = cifar_dataset(dataset=self.dataset, 
                    noise_mode=self.noise_mode,noise_type = self.noise_type, r=self.r,
                    root_dir=self.root_dir, transform=self.transform_ctest,
                    mode='labels_zero', noise_file=self.noise_file)      
            labels_loader = DataLoader(
                dataset=labels_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return labels_loader    

        elif mode=='unlearning':
            labels_dataset = cifar_dataset(dataset=self.dataset, 
                    noise_mode=self.noise_mode, noise_type = self.noise_type, r=self.r,
                    root_dir=self.root_dir, transform=self.transform_train,
                    mode='unlearning', noise_file=self.noise_file, 
                    unlearning_mask=unlearning_mask)      
            labels_loader = DataLoader(
                dataset=labels_dataset, 
                batch_size=self.batch_size*unlearning_batch_size,
                shuffle=True,
                num_workers=self.num_workers)          
            return labels_loader    

        elif mode=='unlearning_vit':
            labels_dataset = cifar_dataset(dataset=self.dataset, 
                    noise_mode=self.noise_mode, noise_type = self.noise_type, r=self.r,
                    root_dir=self.root_dir, transform=self.transform_cWA,
                    mode='unlearning', noise_file=self.noise_file, 
                    unlearning_mask=unlearning_mask)      
            labels_loader = DataLoader(
                dataset=labels_dataset, 
                batch_size=self.batch_size*unlearning_batch_size,
                shuffle=True,
                num_workers=self.num_workers)          
            return labels_loader    