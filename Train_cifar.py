import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import random
import os
import numpy as np
import dataloader_cifar as dataloader
import json
from train_function import *
from args_config import get_args

args = get_args()
import warnings

# Ignore UserWarning with specific message
warnings.filterwarnings("ignore", message="positional arguments and argument \"destination\" are deprecated")
# Fix random seed for reproducibility
os.environ['PYTHONHASHSEED']=str(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.cuda.set_device(args.gpu_id)

# make log files
(log_files) = setup_log_files(args)
(test_log, prob_log,
 labels_log, path) = log_files

# warmup epoch setting
if (args.warmup_epochs == -10):
    if args.dataset=='cifar10':
        args.warmup_epochs = 10
    elif args.dataset=='cifar100':
        args.warmup_epochs = 30

# loader load
loader = dataloader.cifar_dataloader(args.dataset,r=args.r,
    noise_mode=args.noise_mode,batch_size=args.batch_size,
    num_workers=args.num_workers,root_dir=args.data_path, noise_type = args.noise_type,
    noise_file='%s/%.1f_%s.json'%(args.data_path,args.r,args.noise_mode))

# Create two networks
print('| Building net')
net1 = create_model(args)
if args.net2_vit:
    net2 = create_clip(args)
    clip_models = net2
else:
    net2 = create_model(args)
    clip_models = create_clip(args)
optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer2 = torch.optim.SGD(filter(lambda p: p.requires_grad, net2.parameters()),
            lr=args.vit_learning_rate, momentum=0.9, weight_decay=5e-4)

# Variable initialization
all_loss = [[],[]] 
mask_net1 = []
mask_net2 = []
unlearning_flag = False
resumed = False

# save initial labels
eval_loader = loader.run('eval_train')  
eval_c_loader = eval_loader  
if args.net2_vit:
    eval_c_loader = loader.run('eval_vit')  
save_labels(eval_loader, labels_log)
labels_zero_loader = loader.run('labels_zero_shot')  
clip_text_inputs = torch.cat\
            ([clip.tokenize(f"a photo of a {c}") for c in loader.label_names]).cuda()
clean_labels, noisy_labels, clip_predicted =\
    zero_shot_clip(args, clip_models, labels_zero_loader, clip_text_inputs)

# train loop
with open("%s/config.json"%(path), mode="w") as f:
    json.dump(args.__dict__, f, indent=4)
for epoch in range(args.num_epochs+1):   
    lr=args.lr
    lr_2 = args.vit_learning_rate
    if epoch >= 150:
        lr /= 10 
        lr_2 /= 10     
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr       
    for param_group in optimizer2.param_groups:
        param_group['lr'] = lr_2          
    test_loader = loader.run('test')
    eval_loader = loader.run('eval_train')   
    if epoch == args.resume_epoch:
        clip_visual_training(net2)
        optimizer2 = torch.optim.SGD(filter(lambda p: p.requires_grad, net2.parameters()),
            lr=args.vit_learning_rate, momentum=0.9, weight_decay=5e-4)
    # Warmup training
    if epoch<args.warmup_epochs:      
        warmup_trainloader = loader.run('warmup')
        print('Warmup Net1')
        warmup(args,epoch,net1,optimizer1,warmup_trainloader)    
        print('\nWarmup Net2')
        if args.net2_vit:
            warmup_trainloader = loader.run('warmup_vit')
            if args.wc_target:
                warmup_vit_target(args,epoch,net2,optimizer2,
                    warmup_trainloader,clip_text_inputs) 
            else:
                warmup_vit(args,epoch,net2,optimizer2,
                    warmup_trainloader,clip_text_inputs) 
        else:
            warmup(args,epoch,net2,optimizer2,warmup_trainloader) 
    else:   
        # Evaluate training data 
        prob1,all_loss[0]=\
            eval_train(args,net1,eval_loader,all_loss[0])   
        prob2,all_loss[1]=\
            eval_train(args,net2,eval_c_loader,all_loss[1]) 
        if args.save_prob_log:
            write_files(prob_log, prob1, epoch, net=1)
            write_files(prob_log, prob2, epoch, net=2)
        # Predict clean samples
        pred1 = (prob1 > args.p_threshold)      
        pred2 = (prob2 > args.p_threshold) 
        if args.unlearning:   
            # Unlearning target sample selection
            if epoch % args.unlearning_period == 0 and\
                  epoch > args.unlearning_start and epoch != args.num_epochs: 
                unlearning_flag = True 
                mask_net1 = unlearning_sample_selection\
                    (args, torch.stack(all_loss[0]), 
                     noisy_labels, clip_predicted) 
                mask_net2 = unlearning_sample_selection\
                    (args, torch.stack(all_loss[1]),  
                     noisy_labels, clip_predicted)
                # Copy models for teacher networks
                cp_net1, cp_net2 = model_copy(net1), model_copy(net2)
                mask1_L, mask2_L = \
                    mask_net1.sum().item(), mask_net2.sum().item()
                print('unlearning sample net1 {} net2 {}\n'.format(mask1_L, mask2_L))
            # Unlearning training
            if epoch % args.unlearning_period in range(args.unlearning_duration+1)\
                  and unlearning_flag: 
                print('unlearning : {}'.format(epoch % args.unlearning_period))
                # Execute unlearning for net1
                if mask1_L != 0:
                    unlearning_trainloader1 =\
                        loader.run('unlearning',unlearning_mask=mask_net1, 
                            unlearning_batch_size=args.unlearning_batch_size)
                    train_unlearning(args, net1, cp_net1,
                        optimizer1, unlearning_trainloader1, epoch)
                # Execute unlearning for net2
                if mask2_L != 0:
                    if args.net2_vit:
                        unlearning_trainloader2 =\
                            loader.run('unlearning_vit',unlearning_mask=mask_net2, 
                                unlearning_batch_size=args.unlearning_batch_size)
                    else:
                        unlearning_trainloader2 =\
                            loader.run('unlearning',unlearning_mask=mask_net2, 
                                unlearning_batch_size=args.unlearning_batch_size)
                    train_unlearning(args, net2, cp_net2,
                        optimizer2, unlearning_trainloader2, epoch)
            else:
                unlearning_flag = False
            mask_ten1 = torch.as_tensor(mask_net1, dtype=torch.bool)
            mask_ten2 = torch.as_tensor(mask_net2, dtype=torch.bool)
            mask_or1, mask_or2 = (mask_ten1 | mask_ten2, mask_ten1 | mask_ten2)
        else:
            mask_or1, mask_or2 = ([], []) 
        # Co-training
        labeled_trainloader1, unlabeled_trainloader1 =\
            loader.run('train',pred2,prob2,mask_or2)
        print('Train Net1')
        train(args,epoch,net1,net2,optimizer1,labeled_trainloader1,\
            unlabeled_trainloader1,net_number=0)
        if args.only_clean:
            # Train only with clean samples
            labeled_trainloader2 =\
                loader.run('only_clean',pred1,prob1,mask_or1)
            print('\nTrain Net2')
            train_only_clean(args,epoch,net2,optimizer2,
                labeled_trainloader2,net_number=1)
        else: 
            # Standard co-training
            labeled_trainloader2, unlabeled_trainloader2 =\
                loader.run('train',pred1,prob1,mask_or1)
            print('\nTrain Net2')
            train(args,epoch,net2,net1,optimizer2,labeled_trainloader2,\
                unlabeled_trainloader2,net_number=1)

    # Test and log results
    test(args,epoch,net1,net2,test_loader,test_log)  

# save config
with open("%s/config.json"%(path), mode="w") as f:
    json.dump(args.__dict__, f, indent=4)