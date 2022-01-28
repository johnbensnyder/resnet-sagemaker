import os
import torch
import torchvision as tv
import pytorch_lightning as pl
import pl_bolts
import webdataset as wds
import torch.distributed as dist
from apex.parallel import DistributedDataParallel as DDP
from resnet_sagemaker.utils import mixup_data, mixup_criterion
from .utils import train_preproc, val_preproc
from .dist import all_gather

class ResNet(pl.LightningModule):
    
    def __init__(self, num_classes, resnet_version,
                 train_path, val_path, optimizer='adamw',
                 lr=1e-3, batch_size=64,
                 dataloader_workers=4, 
                 max_epochs=20,
                 warmup_epochs=1,
                 mixup_alpha=0.,
                 *args, **kwargs):
        super().__init__()
        self.automatic_optimization = False
        
        self.__dict__.update(locals())
        
        resnets = {
            18:tv.models.resnet18,
            34:tv.models.resnet34,
            50:tv.models.resnet50,
            101:tv.models.resnet101,
            152:tv.models.resnet152
        }
        
        optimizers = {
            'adam': torch.optim.Adam,
            'adamw': torch.optim.AdamW,
            'sgd': torch.optim.SGD
        }
        
        self.criterion = torch.nn.CrossEntropyLoss()
        
        self.model = resnets[resnet_version]()
        linear_size = list(self.model.children())[-1].in_features
        self.model.fc = torch.nn.Linear(linear_size, num_classes)
        if int(os.environ.get("WORLD_SIZE", 1))>1:
            self.model = DDP(self.model.cuda(), delay_allreduce=True)
        self.optimizer = optimizers[optimizer]
        
    def configure_optimizers(self):
        opt = self.optimizer(self.parameters(), lr=self.lr)
        self.schedule = pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR(opt,
                                                                                       self.warmup_epochs,
                                                                                       self.max_epochs)
        return opt
                                                                                       
        
    def forward(self, X):
        return self.model(X)
    
    def train_dataloader(self):
        dataset = wds.WebDataset(self.train_path).shuffle(1024) \
                        .decode("pil").to_tuple("jpeg", "cls").map_tuple(train_preproc, lambda x:x)
        
        return torch.utils.data.DataLoader(dataset, 
                                           num_workers=self.dataloader_workers, 
                                           batch_size=self.batch_size,
                                           pin_memory=True,
                                           prefetch_factor=2)
    
    def val_dataloader(self):
        dataset = wds.WebDataset(self.val_path).shuffle(1024) \
                        .decode("pil").to_tuple("jpeg", "cls").map_tuple(val_preproc, lambda x:x)
        return torch.utils.data.DataLoader(dataset, 
                                           num_workers=self.dataloader_workers, 
                                           batch_size=self.batch_size,
                                           pin_memory=True,
                                           prefetch_factor=2)
    
    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()
        x, y = batch
        if self.mixup_alpha>0.:
            mixed_x, y_a, y_b, lam = mixup_data(x, y, self.mixup_alpha)
            preds = self(mixed_x)
            loss = mixup_criterion(self.criterion, preds, y_a, y_b, lam)
        else:
            preds = self(x)
            loss = self.criterion(preds, y)
        self.manual_backward(loss)
        opt.step()
        self.schedule.step()
        acc = (y == torch.argmax(preds, 1)).type(torch.FloatTensor).mean()
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True, rank_zero_only=False, sync_dist=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=False, logger=True, rank_zero_only=False, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.criterion(preds, y)
        acc = (y == torch.argmax(preds, 1)).type(torch.FloatTensor).mean()
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True, rank_zero_only=False, sync_dist=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=False, logger=True, rank_zero_only=False, sync_dist=True)
        return acc 
