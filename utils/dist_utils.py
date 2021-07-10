import os
from ast import literal_eval

def is_sm():
    sm_training_env = os.environ.get('SM_TRAINING_ENV', None)
    if isinstance(sm_training_env, dict):
        return True
    return False

def is_sm_dist():
    """Check if environment variables are set for Sagemaker Data Distributed
    """
    sm_training_env = os.environ.get('SM_TRAINING_ENV', None)
    if not isinstance(sm_training_env, dict):
        return False
    sm_training_env = literal_eval(sm_training_env)
    additional_framework_parameters = sm_training_env.get('additional_framework_parameters', None)
    if not isinstance(additional_framework_parameters, dict):
        return False
    return bool(additional_framework_parameters.get('sagemaker_distributed_dataparallel_enabled', False))
