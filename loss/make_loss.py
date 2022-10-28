import torch.nn.functional as F
from .softmax_loss import FocalLoss, LabelSmoothingLoss, F1Loss
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss

def make_loss(cfg,num_classes): # make loss는 class가 아닌 definition

    sampler = cfg.sampler 
    loss_type = cfg.loss_type
    num_instance = cfg.num_instance
    feat_norm = cfg.feat_norm
    feat_dim = 2048
    margin = cfg.margin
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True) 

    if "triplet" in sampler : # softmax_triplet or triplet
        if loss_type == "triplet": 
            if not margin:
                triplet = TripletLoss(feat_norm)
                print("Using soft triplet loss for training")
            else :
                triplet = TripletLoss(feat_norm,margin)
                print("Using triplet loss with margin:{}".format(margin))
    
    if cfg.label_smooth :
        xent = LabelSmoothingLoss(num_classes = num_classes)
        print(f"Label smooth on, num classes:{num_classes}")
    
    if sampler == 'softmax':
        def loss_func(score,feat,target):
            return F.cross_entropy(score,target)
    
    elif sampler == 'softmax_triplet':
        if loss_type == 'triplet':

            def loss_func(score,feat,target):
                if cfg.label_smooth:
                    ID_LOSS = xent(score, target) # LabelSmooth
                    TRI_LOSS = triplet(feat, target)[0]
                    return cfg.ID_loss_weight * ID_LOSS + cfg.triplet_loss_weight * TRI_LOSS
            
                else:
                    ID_LOSS = F.cross_entropy(score, target)
                    TRI_LOSS = triplet(feat, target)[0]
                    return cfg.ID_loss_weight * ID_LOSS + cfg.triplet_loss_weight * TRI_LOSS
                    
    return loss_func, center_criterion