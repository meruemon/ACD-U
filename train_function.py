import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from sklearn.mixture import GaussianMixture
from utils import *

# Define loss functions
CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
criterion = SemiLoss()
conf_penalty = NegEntropy()

# Evaluate model performance on test dataset
def test(args,epoch,net1,net2,test_loader,test_log=[]):
    net1.eval()
    net2.eval()
    correct = 0
    correct_net1 = 0
    correct_net2 = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, inputs_c, targets) in enumerate(test_loader):
            inputs, inputs_c, targets =\
                  inputs.cuda(), inputs_c.cuda(), targets.cuda()
            outputs1 = net1(inputs)
            if args.net2_vit:
                outputs2 = net2(inputs_c)    
            else:
                outputs2 = net2(inputs)      

            # Calculate accuracy for ensemble and individual networks
            outputs = outputs1+outputs2
            _, predicted = torch.max(outputs, 1)               
            correct += predicted.eq(targets).cpu().sum().item() 
            _, predicted_net1 = torch.max(outputs1, 1)            
            correct_net1 += predicted_net1.eq(targets).cpu().sum().item() 
            _, predicted_net2 = torch.max(outputs2, 1)            
            correct_net2 += predicted_net2.eq(targets).cpu().sum().item() 
            total += targets.size(0)
                            
    acc = 100.*correct/total
    acc_net1 = 100.*correct_net1/total            
    acc_net2 = 100.*correct_net2/total

    # Print and log results
    print("\n| Test Epoch #%d\t Acc: %.2f\t Acc1: %.2f\t Acc2: %.2f%%\n"\
           %(epoch,acc,acc_net1,acc_net2))  
    del inputs, inputs_c, outputs, outputs1, outputs2,\
        predicted, predicted_net1, predicted_net2, correct, \
        correct_net1, correct_net2, total
    torch.cuda.empty_cache()
    with open(test_log, "a") as f:
        f.write('Epoch:%d   Acc:%.2f   Acc1:%.2f   Acc2:%.2f\n'\
                %(epoch,acc,acc_net1,acc_net2))

# Predict clean sample probability using GMM
def eval_train(args,model,eval_loader,all_loss):    
    model.eval()
    dataset_size = len(eval_loader.dataset)
    losses = torch.zeros(dataset_size)
    with torch.no_grad():
        for _, (inputs, targets, index, _) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda() 
            outputs = model(inputs) 
            loss = CE(outputs, targets)  
            for b in range(inputs.size(0)):
                losses[index[b]]=loss[b] 
    losses = (losses-losses.min())/(losses.max()-losses.min())    
    all_loss.append(losses)

    # average loss over last 5 epochs to improve convergence stability
    if args.r==0.9: 
        history = torch.stack(all_loss)
        input_loss = history[-5:].mean(0)
        input_loss = input_loss.reshape(-1,1)
    else:
        input_loss = losses.reshape(-1,1)
    
    # fit a two-component GMM to the loss
    gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss) 
    prob = prob[:,gmm.means_.argmin()]         

    del inputs, outputs, losses, gmm, input_loss
    torch.cuda.empty_cache()
    return prob, all_loss

# warmup
def warmup(args,epoch,net,optimizer,dataloader):
    net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    for batch_idx, (inputs, labels, _, _) in enumerate(dataloader):      
        inputs, labels = inputs.cuda(), labels.cuda() 
        optimizer.zero_grad()
        outputs = net(inputs)               
        loss = CEloss(outputs, labels)      
        # penalize confident prediction for asymmetric noise and clothing1M dataset
        if args.noise_mode=='asym' or args.dataset=='clothing1M':  
            penalty = conf_penalty(outputs)
            L = loss + penalty     
            del  penalty
        elif args.noise_mode=='sym' or args.noise_mode == 'cifarn':   
            L = loss

        L.backward()  
        optimizer.step() 

        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'
                %(args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, loss.item()))
        sys.stdout.flush()
    del inputs, labels, outputs, loss, L
    torch.cuda.empty_cache()

# warmup for vit
def warmup_vit(args,epoch,net,optimizer,dataloader,test_inputs):
    net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    for batch_idx, (inputs_WA, inputs_SA, _) in enumerate(dataloader):      
        inputs_WA, inputs_SA =\
              inputs_WA.cuda(), inputs_SA.cuda()
        # CLIP prediction
        with torch.no_grad():
            image_features, text_features =\
                  net(inputs_WA, test_inputs, mode='normal')
            text_features /= text_features.norm(dim=-1, keepdim=True)   
            image_features /= image_features.norm(dim=-1, keepdim=True)
            pseudo_labels = image_features.mm(text_features.T)
            probability, predicted = torch.max(pseudo_labels, 1) 
        # Loss calculation
        outputs = net(inputs_SA)
        loss = (probability * CE(outputs, predicted)).mean()    

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()  
        optimizer.step() 

        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'
                %(args.dataset, args.r, args.noise_mode, epoch, 
                  args.num_epochs, batch_idx+1, num_iter, loss.item()))
        sys.stdout.flush()
        del inputs_SA, inputs_WA, image_features,\
            text_features, pseudo_labels, outputs,\
            probability, loss
        torch.cuda.empty_cache()

# warmup for vit with target
def warmup_vit_target(args,epoch,net,optimizer,dataloader,test_inputs):
    net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
   
    for batch_idx, (inputs_WA, inputs_SA, label) in enumerate(dataloader):      
        labels = F.one_hot(label, num_classes=args.num_class).float()    
        inputs_WA, inputs_SA, labels=\
              inputs_WA.cuda(), inputs_SA.cuda(), labels.cuda()
        # CLIP prediction
        with torch.no_grad():
            image_features, text_features =\
                  net(inputs_WA, test_inputs, mode='normal')
            text_features /= text_features.norm(dim=-1, keepdim=True)   
            image_features /= image_features.norm(dim=-1, keepdim=True)
            pseudo_labels = image_features.mm(text_features.T)

        # Create pseudo labels
        targets = args.wc_alpha * labels + (1.0-args.wc_alpha)*pseudo_labels

        # Loss calculation
        outputs = net(inputs_SA)
        loss = CE_soft(outputs, targets).mean()  

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()  
        optimizer.step() 

        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'
                %(args.dataset, args.r, args.noise_mode, epoch, 
                  args.num_epochs, batch_idx+1, num_iter, loss.item()))
        sys.stdout.flush()
        del inputs_SA, inputs_WA, image_features,\
            text_features, pseudo_labels, outputs,\
            loss, labels, targets
        torch.cuda.empty_cache()
  
# train with labeled and unlabeled data
def train(args,epoch,net,net2,optimizer,labeled_trainloader, unlabeled_trainloader,net_number=0):
    net.train()
    net2.eval() 

    unlabeled_train_iter = iter(unlabeled_trainloader)    
    num_iter = (len(labeled_trainloader.dataset)//args.batch_size)+1
    for batch_idx, (inputs_x, inputs_x2, inputs_cx1, inputs_cx2, 
                    labels_x, w_x, index_x) in enumerate(labeled_trainloader):      
        try:
            inputs_u, inputs_u2, inputs_cu1, inputs_cu2, index_u =\
                  unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2, inputs_cu1, inputs_cu2, index_u =\
                  unlabeled_train_iter.next()                
        batch_size = inputs_x.size(0)
        
        # Transform label to one-hot
        labels_x = F.one_hot(labels_x, num_classes=args.num_class).float()      
        w_x = w_x.view(-1,1).type(torch.FloatTensor) 
        # Get images and labels from batch
        inputs_x, inputs_x2, labels_x, w_x, index_x =\
              inputs_x.cuda(), inputs_x2.cuda(), labels_x.cuda(), w_x.cuda(), index_x.cuda()
        inputs_u11, inputs_u12, inputs_u21, inputs_u22, index_u =\
                inputs_u.cuda(), inputs_u2.cuda(), inputs_u.cuda(),\
                inputs_u2.cuda(), index_u.cuda()
        if args.net2_vit and net_number==0:
            inputs_u21, inputs_u22 = inputs_cu1.cuda(), inputs_cu2.cuda()
        if args.net2_vit and net_number==1:        
             inputs_x, inputs_x2, inputs_u11, inputs_u12 =\
                inputs_cx1.cuda(), inputs_cx2.cuda(), inputs_cu1.cuda(), inputs_cu2.cuda()

        # Generate the pseudo-labels
        with torch.no_grad():
            outputs_u11 = net(inputs_u11)
            outputs_u12 = net(inputs_u12)
            outputs_u21 = net2(inputs_u21)
            outputs_u22 = net2(inputs_u22)       

            outputs_x = net(inputs_x)
            outputs_x2 = net(inputs_x2)      

            # Generate the pseudo-labels for unlabeled samples
            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) +\
                   torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4       
            ptu = pu**(1/args.T) 
            targets_u = ptu / ptu.sum(dim=1, keepdim=True) 
            targets_u = targets_u.detach()       
            # Generate the pseudo-labels for labeled samples
            px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
            px = w_x*labels_x + (1-w_x)*px              
            ptx = px**(1/args.T) 
            targets_x = ptx / ptx.sum(dim=1, keepdim=True)         
            targets_x = targets_x.detach()       
        
        # mixup
        l = np.random.beta(args.alpha, args.alpha)
        l = max(l, 1-l)
        all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u11, inputs_u12], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)
        idx = torch.randperm(all_inputs.size(0))
        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        mixed_input = l * input_a + (1 - l) * input_b        
        mixed_target = l * target_a + (1 - l) * target_b
    
        del all_inputs, all_targets, input_a, input_b, inputs_u11,\
            inputs_u12, inputs_u21, inputs_u22, inputs_x, inputs_x2,\
            inputs_cx1, inputs_cx2, inputs_u, inputs_u2, inputs_cu1,\
            inputs_cu2, targets_x, targets_u
        torch.cuda.empty_cache()
        # Calculate loss for labeled and unlabeled data
        logits = net(mixed_input)
        logits_x = logits[:batch_size*2]
        logits_u = logits[batch_size*2:]   
        Lx, Lu, lamb = criterion(args, logits_x, mixed_target[:batch_size*2], 
                                 logits_u, mixed_target[batch_size*2:], 
                                 epoch+batch_idx/num_iter, args.warmup_epochs)
        
        # regularization
        prior = torch.ones(args.num_class)/args.num_class
        prior = prior.cuda()        
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))

        loss = Lx + lamb * Lu  + penalty
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f  Unlabeled loss: %.2f'
                %(args.dataset, args.r, args.noise_mode, epoch, 
                  args.num_epochs, batch_idx+1, num_iter, Lx.item(), Lu.item()))
        sys.stdout.flush()
        del mixed_input, mixed_target, logits, logits_x, logits_u,\
            Lx, Lu, penalty, prior, pred_mean, loss, w_x
        torch.cuda.empty_cache()

# train on only labeled data
def train_only_clean(args,epoch,net,optimizer,labeled_trainloader,net_number=0):
    net.train()

    num_iter = (len(labeled_trainloader.dataset)//(args.batch_size*2))+1
    for batch_idx, (inputs_x, inputs_x2, inputs_cx1, inputs_cx2, 
                    labels_x, w_x, index_x) in enumerate(labeled_trainloader): 

        # Transform label to one-hot
        labels_x = F.one_hot(labels_x, num_classes=args.num_class).float()      
        w_x = w_x.view(-1,1).type(torch.FloatTensor) 

        # Get images and labels from batch
        labels_x, w_x, index_x =\
              labels_x.cuda(), w_x.cuda(), index_x.cuda()
        if args.ws_train:
            inputs_x, inputs_x2, inputs_t1, inputs_t2 =\
            inputs_x.cuda(), inputs_x2.cuda(), inputs_cx1.cuda(), inputs_cx2.cuda()
        else:
            inputs_x, inputs_x2, inputs_t1, inputs_t2 =\
                inputs_x.cuda(), inputs_x2.cuda(), inputs_x.cuda(), inputs_x2.cuda()

        # Generate the pseudo-labels
        with torch.no_grad():
            outputs_x = net(inputs_x)
            outputs_x2 = net(inputs_x2)   

            # Generate the pseudo-labels for labeled samples
            px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
            px = w_x*labels_x + (1-w_x)*px              
            ptx = px**(1/args.T)    
            targets_x = ptx / ptx.sum(dim=1, keepdim=True)          
            targets_x = targets_x.detach()       
        
        # mixup
        l = np.random.beta(args.alpha, args.alpha)
        l = max(l, 1-l)
        all_inputs = torch.cat([inputs_t1, inputs_t2], dim=0)
        all_targets = torch.cat([targets_x, targets_x], dim=0)
        idx = torch.randperm(all_inputs.size(0))
        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        mixed_input = l * input_a + (1 - l) * input_b        
        mixed_target = l * target_a + (1 - l) * target_b

        # Calculate labeled loss
        logits = net(mixed_input)
        Lx = -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * mixed_target, dim=1))

        # regularization
        prior = torch.ones(args.num_class)/args.num_class
        prior = prior.cuda()        
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))

        loss = Lx + penalty

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f'
                %(args.dataset, args.r, args.noise_mode, epoch, 
                  args.num_epochs, batch_idx+1, num_iter, Lx.item()))
        sys.stdout.flush()
        del all_inputs, all_targets, input_a, input_b,\
            inputs_x, inputs_x2, inputs_t1, inputs_t2,\
            mixed_input, mixed_target, logits, Lx,\
            penalty, prior, pred_mean, loss, w_x
        torch.cuda.empty_cache()

# Unlearning target sample selection
def unlearning_sample_selection(args, losses, noisy_labels, predicted_labels):
    diff = losses[-1] - losses[-1-args.unlearning_period]
 
    # Calculate thresholds
    threshold_diff = torch.quantile(diff, args.threshold_diff)
    threshold_current = torch.quantile(losses[-1], args.threshold_current)
    
    # Select samples with low current loss
    if args.current_condition:
        mask_current = losses[-1] < threshold_current
    else:
        mask_current = torch.zeros_like(losses[-1], dtype=torch.bool)
    
    # Select samples with decreasing loss trend
    if args.diff_condition:
        mask_diff = diff < threshold_diff
    else:
        mask_diff = torch.zeros_like(diff, dtype=torch.bool)

    # Select samples where noisy labels differ from CLIP predictions
    if args.label_condition:
        mask_labels = noisy_labels != predicted_labels
    else:
        mask_labels = torch.ones_like(losses[-1], dtype=torch.bool)
    
    # Combine conditions: (Condition 1 OR Condition 2) AND Condition 3
    mask = (mask_diff | mask_current) & mask_labels 
    return mask
    
# Train model with unlearning mechanism
def train_unlearning(args, model, cp_model, optimizer, loader, epoch):    
    model.train()
    num_iter = (len(loader.dataset)//(args.batch_size*args.unlearning_batch_size))+1
    for batch_idx, (inputs, _) in enumerate(loader):
        if inputs.size(0) == 1:
            sys.stdout.write(f"\rSkipping batch due to batch size 1")
            sys.stdout.flush()
            continue

        # Loss calculation
        inputs = inputs.cuda()
        outputs = model(inputs) 
        cp_outputs = cp_model(inputs) 
        del inputs
        torch.cuda.empty_cache()
        loss = - distill_kl_loss(outputs, cp_outputs, args.unlearning_T)
        del outputs, cp_outputs
        torch.cuda.empty_cache()

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Loss: %.4f'
                %(args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, 
                  batch_idx+1, num_iter, loss.item()))
        sys.stdout.flush()
        del loss
        torch.cuda.empty_cache()
    print('\n')