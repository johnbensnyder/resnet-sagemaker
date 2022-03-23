import os
import json
from ast import literal_eval

import importlib
import importlib.util
import sys
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

def get_training_world():

    """
    Calculates number of devices in Sagemaker distributed cluster
    """
    
    # Get params of Sagemaker distributed cluster from predefined env variables
    num_gpus = int(os.environ["SM_NUM_GPUS"])
    num_cpus = int(os.environ["SM_NUM_CPUS"])
    hosts = json.loads(os.environ["SM_HOSTS"])
    current_host = os.environ["SM_CURRENT_HOST"]

    # Define PyTorch training world
    world = {}
    world["number_of_processes"] = num_gpus if num_gpus > 0 else num_cpus
    world["number_of_machines"] = len(hosts)
    world["size"] = world["number_of_processes"] * world["number_of_machines"]
    world["machine_rank"] = hosts.index(current_host)
    world["master_addr"] = hosts[0]
    world["master_port"] = "55555" # port is defined by Sagemaker

    return world

def is_sm():
    """Check if we're running inside a sagemaker training job
    """
    sm_training_env = os.environ.get('SM_TRAINING_ENV', None)
    if not isinstance(sm_training_env, dict):
        return False
    return True

def is_sm_dist():
    """Check if environment variables are set for Sagemaker Data Distributed
    This has not been tested
    """
    sm_training_env = os.environ.get('SM_TRAINING_ENV', None)
    if not isinstance(sm_training_env, dict):
        return False
    sm_training_env = literal_eval(sm_training_env)
    additional_framework_parameters = sm_training_env.get('additional_framework_parameters', None)
    if not isinstance(additional_framework_parameters, dict):
        return False
    return bool(additional_framework_parameters.get('sagemaker_distributed_dataparallel_enabled', False))

def get_herring_world():
    return {"machine_rank": 0, "number_of_processes": 8, "size": 8}
>>>>>>> 74270437b7717782c6a93252c058ab9473ea965e
