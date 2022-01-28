import os
import tensorflow as tf
from preprocessing import resnet_preprocessing, imagenet_preprocessing, darknet_preprocessing
from utils.dist_utils import is_sm_dist, is_sm
from pathlib import Path
if is_sm_dist():
    import smdistributed.dataparallel.tensorflow as dist
else:
    import horovod.tensorflow as dist
    
# Various tuning knobs for tf.data performance
data_options = tf.data.Options()

# information about these optimizations can be found here
# https://www.tensorflow.org/api_docs/python/tf/data/Options
# this paper explains some of their implementations and shows when they provide benefit
# https://arxiv.org/pdf/2101.12127.pdf
data_options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.FILE
# slack can reduce some CPU/GPU contention
data_options.experimental_slack = True
# turning off intra_op_parallelism by setting to 1 often improves pipeline performance
data_options.experimental_threading.max_intra_op_parallelism = 1
data_options.experimental_optimization.apply_default_optimizations = True

# data_options.experimental_optimization.filter_fusion = True
# data_options.experimental_optimization.map_and_batch_fusion = True
# data_options.experimental_optimization.map_and_filter_fusion = True
# data_options.experimental_optimization.map_fusion = True
# data_options.experimental_optimization.map_parallelization = True

# map_vectorization_options = tf.data.experimental.MapVectorizationOptions()
# map_vectorization_options.enabled = True
# map_vectorization_options.use_choose_fastest = True

# data_options.experimental_optimization.map_vectorization = map_vectorization_options

# data_options.experimental_optimization.noop_elimination = True
# data_options.experimental_optimization.parallel_batch = True
# data_options.experimental_optimization.shuffle_and_repeat_fusion = True



def create_dataset(data_dir, batch_size, preprocessing='resnet', train=True, pipe_mode=False, device=None):
    if pipe_mode:
        from sagemaker_tensorflow import PipeModeDataset
        data = PipeModeDataset(channel=Path(data_dir).stem, record_format='TFRecord').shard(dist.size(), dist.rank())
    elif data_dir.startswith('s3://'):
        from s3fs import S3FileSystem
        fs = S3FileSystem()
        filenames = [os.path.join('s3://', i) for i in fs.ls(data_dir)]
        data = tf.data.TFRecordDataset(filenames).shard(dist.size(), dist.rank()).cache()
    else:
        filenames = [os.path.join(data_dir, i) for i in os.listdir(data_dir)]
        data = tf.data.TFRecordDataset(filenames).shard(dist.size(), dist.rank())
    parse_fn = lambda record: parse(record, train, preprocessing)
    data = data.shuffle(buffer_size=1000)
    data = data.map(parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # we drop remainder because we want same sized batches - XLA and because of allreduce being used to calculate
    # accuracy - validation accuracy may be slightly different than computing on all of validation data
    data = data.batch(batch_size, drop_remainder=True) #.prefetch(tf.data.experimental.AUTOTUNE)
    # prefetch to device is currently slower. Might look into Dali
    '''if device:
        prefetch = tf.data.experimental.prefetch_to_device(f'/gpu:{dist.local_rank()}', buffer_size=tf.data.experimental.AUTOTUNE)
        data = data.apply(prefetch)
    else:
        data = data.prefetch(tf.data.experimental.AUTOTUNE)'''
    data = data.prefetch(tf.data.experimental.AUTOTUNE).with_options(data_options)
    return data


@tf.function
def parse(record, is_training, preprocessing): 
    features = {'image/encoded': tf.io.FixedLenFeature((), tf.string),
                'image/class/label': tf.io.FixedLenFeature((), tf.int64),
                'image/object/bbox/xmin': tf.io.VarLenFeature(dtype=tf.float32),
                'image/object/bbox/ymin': tf.io.VarLenFeature(dtype=tf.float32),
                'image/object/bbox/xmax': tf.io.VarLenFeature(dtype=tf.float32),
                'image/object/bbox/ymax': tf.io.VarLenFeature(dtype=tf.float32),
                }
    parsed = tf.io.parse_single_example(record, features)
    image_bytes = tf.reshape(parsed['image/encoded'], shape=[])
    # bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    bbox = tf.stack([parsed['image/object/bbox/%s' % x].values for x in ['ymin', 'xmin', 'ymax', 'xmax']])
    bbox = tf.transpose(tf.expand_dims(bbox, 0), [0, 2, 1])
    if preprocessing == 'resnet':
        augmenter = None # augment.AutoAugment()
        image = resnet_preprocessing.preprocess_image(image_bytes, bbox, 224, 224, 3, is_training=is_training)
    elif preprocessing == 'imagenet': # used by hrnet
        image = imagenet_preprocessing.preprocess_image(image_bytes, bbox, 224, 224, 3, is_training=is_training)
    elif preprocessing == 'darknet':
        image = darknet_preprocessing.preprocess_image(image_bytes, bbox, 256, 256, 3, is_training=is_training)


    label = tf.cast(parsed['image/class/label'] - 1, tf.int32)
    one_hot_label = tf.one_hot(label, depth=1000, dtype=tf.float32)
    return image, one_hot_label


def parse_train(record, preprocessing):
    return parse(record, is_training=True, preprocessing=preprocessing)


def parse_validation(record, preprocessing):
    return parse(record, is_training=False, preprocessing=preprocessing)

