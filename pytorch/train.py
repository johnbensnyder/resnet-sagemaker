import os
import argparse
import sys
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

import torch
import torchvision as tv
import pytorch_lightning as pl
import webdataset as wds
from resnet_sagemaker.models import ResNet
from resnet_sagemaker.callbacks import PlSageMakerLogger, ProfilerCallback

import torch.distributed as dist
from apex.parallel import DistributedDataParallel as DDP

local_rank = int(os.environ.get("LOCAL_RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))
torch.cuda.set_device(local_rank)

if world_size>1:
    dist.init_process_group(
            backend="nccl", init_method="env://",
        )

def parse_args():
    cmdline = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    cmdline.add_argument('--train_file_dir', default='/opt/ml/input/data/train/',
                         help="""Path to dataset in WebDataset format.""")
    cmdline.add_argument('--validation_file_dir', default='/opt/ml/input/data/validation/',
                         help="""Path to dataset in WebDataset format.""")
    cmdline.add_argument('--max_epochs', default=20, type=int,
                         help="""Number of epochs.""")
    cmdline.add_argument('--num_classes', default=1000, type=int,
                         help="""Number of classes.""")
    cmdline.add_argument('--resnet_version', default=50, type=int,
                         help="""Resnet version.""")
    cmdline.add_argument('-lr', '--learning_rate', default=1e-2, type=float,
                         help="""Base learning rate.""")
    cmdline.add_argument('-b', '--batch_size', default=128, type=int,
                         help="""Size of each minibatch per GPU""")
    cmdline.add_argument('--warmup_epochs', default=1, type=int,
                         help="""Number of epochs for learning rate warmup""")
    cmdline.add_argument('--mixup_alpha', default=0.1, type=float,
                         help="""Extent of convex combination for training mixup""")
    cmdline.add_argument('--optimizer', default='adamw', type=str,
                         help="""Optimizer type""")
    cmdline.add_argument('--amp_backend', default='apex', type=str,
                         help="""Mixed precision backend""")
    cmdline.add_argument('--amp_level', default='O2', type=str,
                         help="""Mixed precision level""")
    cmdline.add_argument('--precision', default=16, type=int,
                         help="""Floating point precision""")
    cmdline.add_argument('--profiler_start', default=128, type=int,
                         help="""Profiler start step""")
    cmdline.add_argument('--profiler_steps', default=32, type=int,
                         help="""Profiler steps""")
    cmdline.add_argument('--dataloader_workers', default=4, type=int,
                         help="""Number of data loaders""")
    cmdline.add_argument('--profiler_type', default='smppy', type=str,
                         help="""Profiler type""")
    return cmdline
    
def main(ARGS):

    train_s3_loc = 'pipe:aws s3 cp {0}train_{{{1:04d}..{2:04d}}}.tar -'.format(ARGS.train_file_dir, 0, 2047)

    val_s3_loc = 'pipe:aws s3 cp {0}val_{{{1:04d}..{2:04d}}}.tar -'.format(ARGS.validation_file_dir, 0, 127)

    model_params = {'num_classes': ARGS.num_classes,
                    'resnet_version': ARGS.resnet_version,
                    'train_path': train_s3_loc,
                    'val_path': val_s3_loc,
                    'optimizer': ARGS.optimizer,
                    'lr': ARGS.learning_rate, 
                    'batch_size': ARGS.batch_size,
                    'dataloader_workers': ARGS.dataloader_workers,
                    'max_epochs': ARGS.max_epochs,
                    'warmup_epochs': ARGS.warmup_epochs,
                    'mixup_alpha': ARGS.mixup_alpha,
                   }

    trainer_params = {'gpus': [int(os.environ.get("LOCAL_RANK", 0))],
                      'max_epochs': ARGS.max_epochs,
                      'amp_backend': ARGS.amp_backend,
                      'amp_level': ARGS.amp_level,
                      'precision': ARGS.precision,
                      'progress_bar_refresh_rate': 0,
                      'logger': pl.loggers.TensorBoardLogger('logs/'),
                      'callbacks': [PlSageMakerLogger(), 
                                    ProfilerCallback(start_step=ARGS.profiler_start, 
                                                     num_steps=ARGS.profiler_steps, 
                                                     output_dir='logs/profiling/',
                                                     profiler_type=ARGS.profiler_type)]
                      }

    model = ResNet(**model_params)
    trainer = pl.Trainer(**trainer_params)

    trainer.fit(model)

if __name__=='__main__':
    cmdline = parse_args()
    ARGS, unknown_args = cmdline.parse_known_args()
    main(ARGS)
