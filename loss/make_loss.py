import torch.nn as nn
from .softmax_loss import FocalLoss, LabelSmoothingLoss, F1Loss
from .triplet_loss import TripletLoss,TripletAttentionLoss

def make_loss(cfg,num_classes): # make loss는 class가 아닌 definition

    loss_type = cfg.loss_type
    margin = cfg.margin
    feat_norm = False

    if loss_type == "triplet":
        if not margin:
            triplet = TripletLoss(feat_norm)
            print("Using soft triplet loss for training")
        else :
            triplet = TripletLoss(feat_norm,margin)
            print("Using triplet loss with margin:{}".format(margin))
    
    elif loss_type == "Attention_triplet":
        if not margin:
            triplet = TripletAttentionLoss()
            print("Using Attention triplet loss for training")
        else :
            triplet = TripletAttentionLoss(margin)
            print("Using Attention triplet loss with margin:{}".format(margin))
    
    elif loss_type == "xent":
        xent = nn.CrossEntropyLoss()
        def loss_func(score,target):
                return xent(score,target)
    
    elif loss_type == "f1":
        xent = F1Loss()
        def loss_func(score,target):
                return xent(score,target)

    elif loss_type == "xent_triplet":

        if cfg.label_smooth:
            xent = LabelSmoothingLoss()
        else:
            xent = nn.CrossEntropyLoss()
  
        def loss_func(score,feat,target):
            ID_LOSS = xent(score, target) # LabelSmooth
            TRI_LOSS = triplet(feat, target)[0]
            return cfg.ID_loss_weight * ID_LOSS + cfg.Tri_loss_weight * TRI_LOSS , cfg.ID_loss_weight * ID_LOSS, cfg.triplet_loss_weight * TRI_LOSS

    elif loss_type == "f1_triplet":
        xent = F1Loss()
        print(f"Using F1 loss, num classes:{num_classes}")
        def loss_func(score,feat,target):
            ID_LOSS = xent(score, target)
            TRI_LOSS = triplet(feat, target)[0]
            return cfg.ID_loss_weight * ID_LOSS + cfg.Tri_loss_weight * TRI_LOSS, cfg.ID_loss_weight * ID_LOSS, cfg.triplet_loss_weight * TRI_LOSS

    elif loss_type == 'xent_Attention_triplet':
        if not margin:
            triplet = TripletAttentionLoss()
            print("Using Attention triplet loss for training")
        else :
            triplet = TripletAttentionLoss(margin)
            print("Using Attention triplet loss with margin:{}".format(margin))

        if cfg.label_smooth:
            xent = LabelSmoothingLoss()
        else:
            xent = nn.CrossEntropyLoss()

        def loss_func(score,feat,target,cls_param):
            ID_LOSS = xent(score, target)
            TRI_LOSS = triplet(feat, target,cls_param)[0]
            return cfg.ID_loss_weight * ID_LOSS + cfg.triplet_loss_weight * TRI_LOSS , cfg.ID_loss_weight * ID_LOSS, cfg.triplet_loss_weight * TRI_LOSS
   
    return loss_func