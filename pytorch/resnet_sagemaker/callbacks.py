import os
import pickle, gzip
import torch
import pytorch_lightning as pl

class PlSageMakerLogger(pl.Callback):
    
    def __init__(self, frequency=10):
        self.frequency=frequency
        self.step = 0
        self.epoch = 0
    
    def on_train_epoch_start(self, trainer, module, *args, **kwargs):
        self.inner_step = 0
        self.epoch += 1
    
    @pl.utilities.rank_zero_only
    def on_train_batch_end(self, trainer, module, *args, **kwargs):
        if self.inner_step%self.frequency==0:
            print("Step : {} of epoch {}".format(self.inner_step, self.epoch))
            print("Training Losses:")
            print(' '.join(["{0}: {1:.4f}".format(key, float(value)) \
                            for key,value in trainer.logged_metrics.items()]))
        self.inner_step += 1
        self.step += 1
        
    @pl.utilities.rank_zero_only
    def on_validation_end(self, trainer, module, *args, **kwargs):
        print("Validation")
        print(' '.join(["{0}: {1:.4f}".format(key, float(value)) \
                        for key,value in trainer.logged_metrics.items() if 'val' in key]))

class ProfilerCallback(pl.Callback):
    
    def __init__(self, start_step=100, num_steps=50, output_dir='logs/profiling/', profiler_type='torch'):
        super().__init__()
        self.__dict__.update(locals())
        self.step = 0
        self.profiler = torch.profiler.profile(activities=[
                                    torch.profiler.ProfilerActivity.CPU,
                                    torch.profiler.ProfilerActivity.CUDA],
                                    schedule=torch.profiler.schedule(wait=5,
                                                                     warmup=5,
                                                                     active=self.num_steps),
                                    on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(self.output_dir, "tensorboard")),
                                    with_stack=True)
    
    '''@pl.utilities.rank_zero_only
    def setup(self, trainer, module, *args, **kwargs):
        os.makedirs(self.output_dir, exist_ok=True)        
    '''
    
    def on_train_batch_start(self, trainer, module, *args, **kwargs):
        if self.step==self.start_step:
            self.profiler.__enter__()
    
    def on_train_batch_end(self, trainer, module, *args, **kwargs):
        if self.step>=self.start_step and self.step<=self.start_step + self.num_steps:
            self.profiler.step()
        if self.step==self.start_step + self.num_steps:
            self.profiler.__exit__(None, None, None)
        self.step += 1
