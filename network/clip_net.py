import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch
import sys
import clip
from sklearn.mixture import GaussianMixture

import torch
import torch.nn as nn
import clip

class CLIPWithProjector(nn.Module):
    def __init__(self, args):
        super(CLIPWithProjector, self).__init__()
        
        # CLIPモデルのロード
        self.clip_model, _ = clip.load(args.clip_model_name, device='cuda')
        if args.clip_float:
            self.clip_model = self.clip_model.float()
        if hasattr(self.clip_model.visual, 'output_dim'):
            in_dim = self.clip_model.visual.output_dim
        elif hasattr(self.clip_model.visual, 'proj'):
            in_dim = self.clip_model.visual.proj.out_features
        for param in self.clip_model.parameters():
            param.requires_grad = False
        # if args.clip_pro_lin:
        #     self.projector = self.create_linear(in_dim, args.num_class).cuda()
        # else:
        self.projector = self.create_classifier(in_dim, args.num_class).cuda()
    @staticmethod
    def create_classifier(in_dim, out_dim, dropout=0.25):
        return nn.Sequential(nn.Linear(in_dim, in_dim),
                             nn.BatchNorm1d(in_dim),
                             nn.ReLU(inplace=True),
                             nn.Dropout(p=dropout),
                             nn.Linear(in_dim, out_dim),
                             nn.BatchNorm1d(out_dim)
                             )
    @staticmethod
    def create_linear(in_dim, out_dim):
        return nn.Sequential(nn.Linear(in_dim, out_dim))
    
    def forward(self, image, text=[], mode='image'):
        # CLIPモデルを使用して特徴量を取得
        image_features = self.clip_model.encode_image(image)
        image_features = image_features.float()
        if mode=='image':
            projected_image_features = self.projector(image_features)
            return projected_image_features
        elif mode=='normal':
            text_features = self.clip_model.encode_text(text)
            text_features = text_features.float()
            return image_features, text_features 