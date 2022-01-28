import os
import argparse
import logging
from ast import literal_eval
from multiprocessing import cpu_count
import tensorflow as tf
from utils.dist_utils import is_sm_dist
from models import resnet, darknet, hrnet
from engine.schedulers import WarmupScheduler
from engine.optimizers import MomentumOptimizer
from datasets import create_dataset, parse
from engine.trainer import Trainer

if is_sm_dist():
    import smdistributed.dataparallel.tensorflow as dist
else:
    import horovod.tensorflow as dist

def parse_args():
    cmdline = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    cmdline.add_argument('--train_data_dir', default='/opt/ml/input/data/train',
                         help="""Path to dataset in TFRecord format
                             (aka Example protobufs). Files should be
                             named 'train-*' and 'validation-*'.""")
    cmdline.add_argument('--validation_data_dir', default='/opt/ml/input/data/validation',
                         help="""Path to dataset in TFRecord format
                             (aka Example protobufs). Files should be
                             named 'train-*' and 'validation-*'.""")
    cmdline.add_argument('--num_classes', default=1000, type=int,
                         help="""Number of classes.""")
    cmdline.add_argument('--train_dataset_size', default=1281167, type=int,
                         help="""Number of images in training data.""")
    cmdline.add_argument('--model_dir', default='/opt/ml/checkpoints',
                         help="""Path to save model with best accuracy""")
    cmdline.add_argument('-b', '--batch_size', default=128, type=int,
                         help="""Size of each minibatch per GPU""")
    cmdline.add_argument('--warmup_steps', default=500, type=int,
                         help="""steps for linear learning rate warmup""")
    cmdline.add_argument('--num_epochs', default=120, type=int,
                         help="""Number of epochs to train for.""")
    cmdline.add_argument('--schedule', default='cosine', type=str,
                         help="""learning rate schedule""")
    cmdline.add_argument('-lr', '--learning_rate', default=0.1, type=float,
                         help="""Start learning rate.""")
    cmdline.add_argument('--momentum', default=0.9, type=float,
                         help="""Start optimizer momentum.""")
    cmdline.add_argument('--label_smoothing', default=0.1, type=float,
                         help="""Label smoothing value.""")
    cmdline.add_argument('--mixup_alpha', default=0.2, type=float,
                        help="""Mixup beta distribution shape parameter. 0.0 disables mixup.""")
    cmdline.add_argument('--l2_weight_decay', default=1e-4, type=float,
                         help="""L2 weight decay multiplier.""")
    cmdline.add_argument('-fp16', '--fp16', default='True',
                         help="""disable mixed precision training""")
    cmdline.add_argument('-xla', '--xla', default='True',
                         help="""enable xla""")
    cmdline.add_argument('-tf32', '--tf32', default='True',
                         help="""enable tensorflow-32""")
    cmdline.add_argument('--model',
                         help="""Which model to train. Options are:
                         resnet50v1_b, resnet50v1_c, resnet50v1_d, 
                         resnet101v1_b, resnet101v1_c,resnet101v1_d, 
                         resnet152v1_b, resnet152v1_c,resnet152v1_d,
                         resnet50v2, resnet101v2, resnet152v2
                         darknet53, hrnet_w18c, hrnet_w32c""")
    cmdline.add_argument('--resume_from', 
                         help='Path to SavedModel format model directory from which to resume training')
    cmdline.add_argument('--pipe_mode', default='False',
                         help='Path to SavedModel format model directory from which to resume training')
    return cmdline


def main(FLAGS):
    dist.init()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        device = gpus[dist.local_rank()]
        tf.config.experimental.set_visible_devices(device, 'GPU')
    else:
        device = None
    # tf.config.threading.intra_op_parallelism_threads = 1 # Avoid pool of Eigen threads
    tf.config.threading.inter_op_parallelism_threads = max(2, cpu_count()//dist.local_size()-2)
    tf.config.optimizer.set_jit(FLAGS.xla)
    # tf.config.optimizer.set_experimental_options({"auto_mixed_precision": FLAGS.fp16})
    # tf.config.experimental.enable_tensor_float_32_execution(FLAGS.tf32)
    policy = tf.keras.mixed_precision.Policy('mixed_float16' if FLAGS.fp16 else 'float32')
    tf.keras.mixed_precision.set_global_policy(policy)

    preprocessing_type = 'resnet'
    if FLAGS.model == 'resnet50v1_b':
        model = resnet.ResNet50V1_b(weights=None, weight_decay=FLAGS.l2_weight_decay, classes=FLAGS.num_classes)
    elif FLAGS.model == 'resnet50v1_c':
        model = resnet.ResNet50V1_c(weights=None, weight_decay=FLAGS.l2_weight_decay, classes=FLAGS.num_classes)
    elif FLAGS.model == 'resnet50v1_d':
        model = resnet.ResNet50V1_d(weights=None, weight_decay=FLAGS.l2_weight_decay, classes=FLAGS.num_classes)
    elif FLAGS.model == 'resnet101v1_b':
        model = resnet.ResNet101V1_b(weights=None, weight_decay=FLAGS.l2_weight_decay, classes=FLAGS.num_classes)
    elif FLAGS.model == 'resnet101v1_c':
        model = resnet.ResNet101V1_c(weights=None, weight_decay=FLAGS.l2_weight_decay, classes=FLAGS.num_classes)
    elif FLAGS.model == 'resnet101v1_d':
        model = resnet.ResNet101V1_d(weights=None, weight_decay=FLAGS.l2_weight_decay, classes=FLAGS.num_classes)
    elif FLAGS.model == 'resnet152v1_b':
        model = resnet.ResNet152V1_b(weights=None, weight_decay=FLAGS.l2_weight_decay, classes=FLAGS.num_classes)
    elif FLAGS.model == 'resnet152v1_c':
        model = resnet.ResNet152V1_c(weights=None, weight_decay=FLAGS.l2_weight_decay, classes=FLAGS.num_classes)
    elif FLAGS.model == 'resnet152v1_d':
        model = resnet.ResNet152V1_d(weights=None, weight_decay=FLAGS.l2_weight_decay, classes=FLAGS.num_classes)
    elif FLAGS.model == 'resnet50v2':
        model = resnet.ResNet50V2(weights=None, weight_decay=FLAGS.l2_weight_decay, classes=FLAGS.num_classes)
    elif FLAGS.model == 'resnet101v2':
        model = resnet.ResNet101V2(weights=None, weight_decay=FLAGS.l2_weight_decay, classes=FLAGS.num_classes)
    elif FLAGS.model == 'resnet152v2':
        model = resnet.ResNet152V2(weights=None, weight_decay=FLAGS.l2_weight_decay, classes=FLAGS.num_classes)
    elif FLAGS.model == 'darknet53':
        model = darknet.Darknet(weight_decay=FLAGS.l2_weight_decay)
    elif FLAGS.model in ['hrnet_w18c', 'hrnet_w32c']:
        preprocessing_type = 'imagenet'
        model = hrnet.build_hrnet(FLAGS.model)
        model._set_inputs(tf.keras.Input(shape=(None, None, 3)))
    else:
        raise NotImplementedError('Model {} not implemented'.format(FLAGS.model))

    # model.summary()
    steps_per_epoch = FLAGS.train_dataset_size // FLAGS.batch_size
    iterations = steps_per_epoch * FLAGS.num_epochs
    batch_size_per_device = FLAGS.batch_size//dist.size()

    # 5 epochs are for warmup
    if FLAGS.schedule == 'piecewise_short':
        scheduler = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                    boundaries=[iterations//5, 
                                iterations//2, 
                                int(iterations*0.7), 
                                int(iterations*0.9)], 
                    values=[FLAGS.learning_rate, FLAGS.learning_rate * 0.1, 
                            FLAGS.learning_rate * 0.01, FLAGS.learning_rate * 0.001, 
                            FLAGS.learning_rate * 0.0001])
    elif FLAGS.schedule == 'piecewise_long':
        scheduler = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                    boundaries=[iterations//4, 
                                int(iterations*0.6), 
                                int(iterations*0.9)], 
                    values=[FLAGS.learning_rate, FLAGS.learning_rate * 0.1, FLAGS.learning_rate * 0.01, FLAGS.learning_rate * 0.001])
    elif FLAGS.schedule == 'cosine':
        scheduler = tf.keras.experimental.CosineDecayRestarts(initial_learning_rate=FLAGS.learning_rate,
                    first_decay_steps=iterations, t_mul=1, m_mul=1, alpha=1e-3)
    else:
        print('No schedule specified')


    scheduler = WarmupScheduler(scheduler=scheduler, initial_learning_rate=FLAGS.learning_rate * .01, warmup_steps=FLAGS.warmup_steps)

    # TODO support optimizers choice via config
    # opt = tf.keras.optimizers.SGD(learning_rate=scheduler, momentum=FLAGS.momentum, nesterov=True) # needs momentum correction term
    opt = MomentumOptimizer(learning_rate=scheduler, momentum=FLAGS.momentum, nesterov=True) 

    if FLAGS.fp16:
        # opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt, loss_scale=128.)
        opt = tf.keras.mixed_precision.LossScaleOptimizer(opt, dynamic=False, 
                                                          initial_scale=128., 
                                                          dynamic_growth_steps=None)
    
    # FLAGS.label_smoothing = tf.cast(FLAGS.label_smoothing, tf.float16 if FLAGS.fp16 else tf.float32)
    loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=True, 
                                                        label_smoothing=FLAGS.label_smoothing,
                                                        reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE) 

    if dist.rank() == 0:
        if FLAGS.resume_from:
            model = tf.keras.models.load_model(FLAGS.resume_from)
            print('loaded model from', FLAGS.resume_from)
        path_logs = os.path.join(os.getcwd(), FLAGS.model_dir, 'log.csv')
        os.makedirs(FLAGS.model_dir, exist_ok=True)
        logging.basicConfig(filename=path_logs,
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)
        logging.info("Training Logs")
        logger = logging.getLogger('logger')
        logger.info('Training options: %s', FLAGS)
    else:
        logger = None
    
    # barrier
    _ = dist.allreduce(tf.constant(0))
 
    train_data = create_dataset(FLAGS.train_data_dir, batch_size_per_device, 
                                preprocessing=preprocessing_type, pipe_mode=FLAGS.pipe_mode, device=device)
    validation_data = create_dataset(FLAGS.validation_data_dir, batch_size_per_device, 
                                     preprocessing=preprocessing_type, train=False, pipe_mode=FLAGS.pipe_mode, device=device)
    
    trainer = Trainer(model, opt, loss_func, scheduler, 
                      logging=logger, fp16=FLAGS.fp16, mixup_alpha=FLAGS.mixup_alpha, model_dir=FLAGS.model_dir)
    
    for epoch in range(FLAGS.num_epochs):
        if dist.rank() == 0:
            print('Starting training Epoch %d/%d' % (epoch, FLAGS.num_epochs))
        trainer.train_epoch(train_data)
        if dist.rank() == 0:
            print('Starting validation Epoch %d/%d' % (epoch, FLAGS.num_epochs))
        trainer.validation_epoch(validation_data)
        _ = dist.allreduce(tf.constant(0))
        
    if dist.rank() == 0:
        logger.info('Total Training Time: %f' % (time() - start_time))

if __name__ == '__main__':
    cmdline = parse_args()
    FLAGS, unknown_args = cmdline.parse_known_args()
    FLAGS.fp16 = literal_eval(FLAGS.fp16)
    FLAGS.xla = literal_eval(FLAGS.xla)
    FLAGS.tf32 = literal_eval(FLAGS.tf32)
    FLAGS.pipe_mode = literal_eval(FLAGS.pipe_mode)
    main(FLAGS)
