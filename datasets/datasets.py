import os
import tensorflow as tf
from preprocessing import resnet_preprocessing, imagenet_preprocessing, darknet_preprocessing
from utils.dist_utils import is_sm_dist, is_sm
if is_sm_dist():
    import smdistributed.dataparallel.tensorflow as dist
else:
    import horovod.tensorflow as dist

def create_dataset(data_dir, batch_size, preprocessing='resnet', train=True, pipe_mode=False):
    if pipe_mode:
        from sagemaker_tensorflow import PipeModeDataset
        data = PipeModeDataset(channel=data_dir.split('/')[-1], record_format='TFRecord').shard(dist.size(), dist.rank())
    elif data_dir.startswith('s3://'):
        from s3fs import S3FileSystem
        fs = S3FileSystem()
        filenames = [os.path.join('s3://', i) for i in fs.ls(data_dir)]
        data = tf.data.TFRecordDataset(filenames).shard(dist.size(), dist.rank())
    else:
        filenames = [os.path.join(data_dir, i) for i in os.listdir(data_dir)]
        data = tf.data.TFRecordDataset(filenames).shard(dist.size(), dist.rank())
    parse_fn = lambda record: parse(record, train, preprocessing)
    data = data.shuffle(buffer_size=1000)
    data = data.map(parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # we drop remainder because we want same sized batches - XLA and because of allreduce being used to calculate
    # accuracy - validation accuracy may be slightly different than computing on all of validation data
    data = data.batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
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

