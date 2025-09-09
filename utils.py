import torch
import numpy as np
import clip
import torch.nn.functional as F
import copy
import random
from pathlib import Path
from network.PreResNet import *
from network.clip_net import *
from utils import *

# Cross Entropy with soft labels
def CE_soft(output, target):
    per_sample_loss = -torch.sum(F.log_softmax(output, dim=1) * target, dim=1)
    return per_sample_loss

# KL divergence for distillation
def distill_kl_loss(y_s, y_t, T):
    p_s = F.log_softmax(y_s / T, dim=1)
    p_t = F.softmax(y_t / T, dim=1)
    loss = F.kl_div(p_s, p_t, reduction='sum') * (T ** 2) / y_s.shape[0]
    return loss

# Semi-supervised loss
class SemiLoss(object):
    def __call__(self, args, outputs_x, targets_x, 
                outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)
        return Lx, Lu, linear_rampup(args,epoch,warm_up)

# Negative entropy regularization for confident prediction penalty
class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))

# unlabelsed data loss weight ramp-up
def linear_rampup(args, current, warm_up, rampup_length=16):
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u*float(current)

# Create ResNet18 model
def create_model(args):
    model = ResNet18(num_classes=args.num_class)
    model = model.cuda()
    return model

# Create CLIP model
def create_clip(args):
    clip_models = CLIPWithProjector(args)
    return clip_models
    
#   Model copy
def model_copy(model):
    teacher_model = copy.deepcopy(model)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False
    return teacher_model

# Enable training for CLIP visual parameters
def clip_visual_training(model):
    for param in model.clip_model.visual.parameters():
        param.requires_grad = True
    print("CLIP visual train start")

# make log files and return paths
def setup_log_files(args):
    # Create directory path
    path = f"./checkacc/ACDU/run{args.run}/ACDU_{args.dataset}_{args.noise_mode}_{args.r:.1f}"
    if args.dataset == 'clothing1M' or args.dataset == 'webvision':
        path = f"./checkacc/{args.dataset}/ACDU/run{args.run}/ACDU_{args.dataset}_{args.noise_mode}_{args.r:.1f}"
    if args.noise_mode == 'cifarn':
        path = f"./checkacc/ACDU/run{args.run}/ACDU_{args.dataset}_{args.noise_mode}_{args.noise_type}"

    Path(path).mkdir(parents=True, exist_ok=True)

    base_filename = f"{path}/ACDU_{args.dataset}_{args.noise_mode}_{args.r:.1f}"
    if args.noise_mode == 'cifarn':
        base_filename = f"{path}/ACDU_{args.dataset}_{args.noise_mode}_{args.noise_type}"

    log_files = {
        "test": "_acc.txt",
        "labels": "_labels.txt",
    }
    # Add prob-related files only when save_prob_log is enabled
    if args.save_prob_log:
        log_files["prob"] = "_prob.txt"

    # Create paths for each log file
    log_paths = {key: base_filename + suffix for key, suffix in log_files.items()}
    for file_path in log_paths.values():
        open(file_path, 'w').close()
    test_log = log_paths["test"]
    labels_log = log_paths["labels"]
    prob_log = log_paths.get("prob", None)

    return (test_log, prob_log, labels_log, path)

# Save clean and noisy labels to a file
def save_labels(loader, labels_file):
    print('Saving clean and noisy labels to file...')
    dataset_size = len(loader.dataset)
    clean_labels = torch.zeros(dataset_size, dtype=torch.int)
    noisy_labels = torch.zeros(dataset_size, dtype=torch.int)
    for _, (_, targets, index, clean) in enumerate(loader):
        for b in range(targets.size(0)):
            noisy_labels[index[b]]=targets[b]   
            clean_labels[index[b]]=clean[b]
    with open(labels_file, "a") as f:
        f.write("Clean_labels:\n")
        np.savetxt(f, clean_labels.reshape(1, -1), 
                fmt='%d', delimiter=' ', newline=' ')
        f.write('\n')
        f.write("Noisy_labels:\n")
        np.savetxt(f, noisy_labels.reshape(1, -1), 
                fmt='%d', delimiter=' ', newline=' ')
        f.write('\n')
    print(f"Labels saved successfully to {labels_file}")
     
# Zero-shot CLIP evaluation
def zero_shot_clip(args, models, loader, clip_text_inputs):
    print("Starting zero-shot CLIP evaluation...")
    dataset_size = len(loader.dataset)
    clean_labels = torch.zeros(dataset_size, dtype=torch.int)    
    noisy_labels = torch.zeros(dataset_size, dtype=torch.int)  
    predicted_labels = torch.zeros(dataset_size, dtype=torch.int)   
    models.eval()
    with torch.no_grad():
        for _, (inputs, targets, index, clean) in enumerate(loader):
            inputs = inputs.cuda()
            # Prediction by CLIP
            image_features, text_features =\
                  models(inputs, clip_text_inputs, mode='normal')
            text_features /= text_features.norm(dim=-1, keepdim=True)   
            image_features /= image_features.norm(dim=-1, keepdim=True)
            outputs = image_features.mm(text_features.T)
            _, predicted = torch.max(outputs, 1)   
            for b in range(targets.size(0)):
                noisy_labels[index[b]]=targets[b]   
                clean_labels[index[b]]=clean[b]
                predicted_labels[index[b]]=predicted[b]
    print(f"Zero-shot CLIP evaluation completed successfully")
    return clean_labels, noisy_labels, predicted_labels

# Write array data to a log file
def write_files(f_name, write_array, epoch, net=1):
    with open(f_name, "a") as f:
        f.write("Epoch {} (Net {}):\n".format(epoch, net))
        np.savetxt(f, write_array.reshape(1, -1), 
                    fmt='%0.4f', delimiter=' ', newline='\n')





