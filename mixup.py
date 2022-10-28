import torch
def mixup(images,labels,lamb):
    batch_size=images.size()[0]
    ind=torch.randperm(batch_size).cuda()

    mixed_img=lamb*images+(1-lamb)*images[ind, :]
    y_a,y_b=labels,labels[ind]
    return mixed_img,y_a,y_b


def mixed_criterion(criterion,pred,y_a,y_b,lamb):
    return lamb*criterion(pred,y_a)+(1-lamb)*criterion(pred,y_b)

