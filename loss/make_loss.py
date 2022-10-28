import torch.nn.functional as F
from .softmax_loss import FocalLoss, LabelSmoothingLoss, F1Loss
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss

def make_loss(cfg,num_classes): # make loss는 class가 아닌 definition

    sampler = cfg.SAMPLER
    loss_type = cfg.METRIC_LOSS_TYPE
    loss_ratio = cfg.LOSS_RATIO
    num_instance = cfg.NUM_INSTANCE
    feat_dim = 2048
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True) 

    if "triplet" in sampler :
        if loss_type == "triplet":
            if not cfg.MARGIN:
                triplet = TripletLoss()
                print("Using soft triplet loss for training")
            else :
                triplet = TripletLoss(cfg.MARGIN)
                print("Using triplet loss with margin:{}".format(cfg.MARGIN))
    
    if cfg.LABEL_SMOOTH :
        xent = LabelSmoothingLoss(num_classes = num_classes)
        print(f"Label smooth on, num classes:{num_classes}")
    
    if sampler == 'softmax':
        def loss_func(score,feat,target):
            return F.cross_entropy(score,target)
    
    elif sampler == 'softmax_triplet':
        if loss_type == 'triplet':

            def loss_func(score,feat,target):
                if cfg.LABEL_SMOOTH:
                    ID_LOSS = xent(score, target) # LabelSmooth
                    TRI_LOSS = triplet(feat, target)[0]
                    return cfg.ID_LOSS_WEIGHT * ID_LOSS + cfg.TRIPLET_LOSS_WEIGHT * TRI_LOSS
            
                else:
                    ID_LOSS = F.cross_entropy(score, target)
                    TRI_LOSS = triplet(feat, target)[0]
                    return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS