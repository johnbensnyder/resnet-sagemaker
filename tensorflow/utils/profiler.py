import os
import tensorflow as tf

class ProfilerCallback(tf.keras.callbacks.Callback):
    
    def __init__(self, start_step, num_steps, output_dir, profiler_type='smppy'):
        super().__init__()
        self.__dict__.update(locals())
    
    def on_train_begin(self, logs=None):
        if True:#self.profiler_type=='smppy':
            os.environ["SMP_ENABLE_CUDA"] = "1"
            import smppy
            self.profiler = smppy.SMProfiler.instance()
            config = smppy.Config()
            config.profiler = {
                "BucketCount": "32",
                "BucketSizeMB": "64",
                "Compression": "1000001",
                "EnableCuda": "1",
                "EnablePerf": "0",
                "EnableForkFollow": "0",
                "RecordFile": os.path.join(self.output_dir, "smpfile_%p_%t.raw")
            }
            self.profiler.configure(config)
    
    def on_train_batch_begin(self, batch, logs=None):
        if batch == self.start_step:
            if self.profiler_type == 'smppy':
                self.profiler.start_profiling()
            else:
                tf.profiler.experimental.start(self.output_dir)
    
    def on_train_batch_end(self, batch, logs=None, **kwargs):
        if batch == self.start_step + self.num_steps:
            if self.profiler_type == 'smppy':
                self.profiler.stop_profiling()
            else:
                tf.profiler.experimental.stop()
