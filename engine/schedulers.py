import tensorflow as tf


class WarmupScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Wraps another learning rate scheduler to add a linear or exponential warmup
    """
    
    def __init__(self, scheduler, initial_learning_rate, warmup_steps, warmup_type='linear',
                 dtype=tf.float32):
        super(WarmupScheduler, self).__init__()
        self.scheduler = scheduler
        self.initial_learning_rate = tf.cast(initial_learning_rate, dtype)
        self.warmup_steps = tf.cast(warmup_steps, dtype)
        self.warmup_type = warmup_type
        self.dtype = dtype
        self.scheduler_learning_rate = scheduler(0)
        
    def compute_linear_warmup(self, step):
        return ((self.scheduler_learning_rate*step) + (self.initial_learning_rate*(self.warmup_steps-step)))/self.warmup_steps
    
    @tf.function
    def __call__(self, step):
        global_step_recomp = tf.cast(step, self.dtype)
        if global_step_recomp>=self.warmup_steps:
            return self.scheduler(global_step_recomp - self.warmup_steps)
        return self.compute_linear_warmup(global_step_recomp)
    
    def get_config(self):
        scheduler_config = self.scheduler.get_config()
        scheduler_config['initial_learning_rate'] = self.initial_learning_rate
        scheduler_config['warmup_steps'] = self.warmup_steps
        scheduler_config['warmup_type'] = self.warmup_type


