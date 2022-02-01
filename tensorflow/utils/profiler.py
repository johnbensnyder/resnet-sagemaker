import os
import tensorflow as tf

class ProfilerCallback(tf.keras.callbacks.Callback):
    
    def __init__(self, start_step, num_steps, output_dir, profiler_type='tf'):
        super().__init__()
        self.__dict__.update(locals())
    
    def on_train_batch_begin(self, batch, logs=None):
        if batch == self.start_step:
            tf.profiler.experimental.start(self.output_dir)
    
    def on_train_batch_end(self, batch, logs=None, **kwargs):
        if batch == self.start_step + self.num_steps:
            tf.profiler.experimental.stop()
