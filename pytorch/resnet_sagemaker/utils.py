import torch
import torchvision as tv
import numpy as np

def mixup_data(x, y, alpha=1.0):
    
    if alpha>0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

train_preproc = tv.transforms.Compose([
                    tv.transforms.ToTensor(),
                    tv.transforms.RandomResizedCrop(224, scale=(0.8, 1.0), 
                                                    ratio=(0.75, 1.33)),
                    tv.transforms.Normalize((0.485, 0.456, 0.406), 
                                            (0.229, 0.224, 0.225)),
                    tv.transforms.RandomRotation((-5., 5.)),
                    tv.transforms.RandomHorizontalFlip(),
                ])

val_preproc = tv.transforms.Compose([
                    tv.transforms.ToTensor(),
                    tv.transforms.Normalize((0.485, 0.456, 0.406), 
                                            (0.229, 0.224, 0.225)),
                    tv.transforms.Resize(224),
                    tv.transforms.CenterCrop(224),
                ])