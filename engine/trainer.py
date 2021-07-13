import tensorflow as tf
from preprocessing.augmentation_utils import mixup
from time import time
from utils.dist_utils import is_sm_dist
if is_sm_dist():
    import smdistributed.dataparallel.tensorflow as dist
else:
    import horovod.tensorflow as dist

layers = tf.keras.layers

class Trainer(object):
    
    def __init__(self, model, opt, loss_func, scheduler, logging=None, 
                 fp16=True, mixup_alpha=0.0, model_dir='.',
                 start_saving_accuracy=0.7):
        self.model = model
        self.opt = opt
        self.loss_func = loss_func
        self.fp16 = fp16
        self.mixup_alpha = mixup_alpha
        self.time = time()
        self.model_dir = model_dir
        self.best_validation_accuracy = start_saving_accuracy
        self.logging = logging
        self.scheduler = scheduler
        self.iteration = tf.convert_to_tensor(0)
        self.inner_iteration = tf.convert_to_tensor(0)
        
    @tf.function
    def train_step(self, images, labels, first_batch=False):
        batch_size = tf.shape(images)[0]
        images, labels = mixup(batch_size, self.mixup_alpha, images, labels)
        images = tf.cast(images, tf.float16 if self.fp16 else tf.float32)
        with tf.GradientTape() as tape:
            logits = tf.cast(self.model(images, training=True), tf.float32)
            loss_value = self.loss_func(labels, logits)
            loss_value += tf.add_n(self.model.losses)
            if self.fp16:
                scaled_loss_value = self.opt.get_scaled_loss(loss_value)
        tape = dist.DistributedGradientTape(tape)
        if self.fp16:
            grads = tape.gradient(scaled_loss_value, self.model.trainable_variables)
            grads = self.opt.get_unscaled_gradients(grads)
        else:
            grads = tape.gradient(loss_value, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        if first_batch:
            dist.broadcast_variables(self.model.variables, root_rank=0)
            dist.broadcast_variables(self.opt.variables(), root_rank=0)
        probs = layers.Activation('softmax', dtype='float32')(logits)
        top_1_pred = tf.squeeze(tf.math.top_k(probs, k=1)[1])
        sparse_labels = tf.cast(tf.math.argmax(labels, axis=1), tf.int32)
        top_1_accuracy = tf.math.reduce_sum(tf.cast(tf.equal(top_1_pred, sparse_labels), tf.int32))/batch_size
        return loss_value, top_1_accuracy
    
    @tf.function
    def validation_step(self, images, labels):
        images = tf.cast(images, tf.float16 if self.fp16 else tf.float32)
        logits = tf.cast(self.model(images, training=False), tf.float32)
        loss = self.loss_func(labels, logits)
        top_1_pred = tf.squeeze(tf.math.top_k(logits, k=1)[1])
        sparse_labels = tf.cast(tf.math.argmax(labels, axis=1), tf.int32)
        top_1_accuracy = tf.math.reduce_sum(tf.cast(tf.equal(top_1_pred, sparse_labels), tf.int32))
        return loss, top_1_accuracy
    
    def print_train(self, interval, **kwargs):
        elapsed_time = time() - self.time
        self.time = time()
        step_time = elapsed_time/interval
        log_list = [('step', str(self.inner_iteration.numpy())), 
                    ('step time', f"{step_time:.4f}")]
        for key, value in kwargs.items():
            log_list.append((key, f"{value:.4f}"))
        info_str = ", ".join([": ".join(i) for i in log_list])
        if self.logging:
            self.logging.info(info_str)
        print(info_str)
        
    
    def train_epoch(self, dataset, print_interval=50):
        self.inner_iteration = tf.convert_to_tensor(0)
        for step, (images, labels) in enumerate(dataset):
            loss_value, top_1_accuracy = self.train_step(images, labels, first_batch=step==0)
            if dist.rank()==0 and step%print_interval==0:
                self.print_train(print_interval, 
                                 train_loss=loss_value, 
                                 top_1_accuracy=top_1_accuracy,
                                 learning_rate=self.scheduler(self.iteration))
            self.iteration+=1
            self.inner_iteration+=1
    
    def validation_epoch(self, dataset, output_name=None):
        validation_score = 0
        counter = 0
        for step, (images, labels) in enumerate(dataset):
            if step==0:
                batch_size = tf.shape(images)[0]
            loss, score = self.validation_step(images, labels)
            validation_score += score.numpy()
            counter += 1
        validation_accuracy = validation_score / (batch_size * counter)
        average_validation_accuracy = dist.allreduce(tf.constant(validation_accuracy))
        average_validation_loss = dist.allreduce(tf.constant(loss))
        if dist.rank() == 0:
            info_str = 'Validation Accuracy: {0}, Validation Loss: {1}'.format(average_validation_accuracy,
                                                                               average_validation_loss)
            if self.logging:
                self.logging.info(info_str)
            print(info_str)
            if average_validation_accuracy > self.best_validation_accuracy:
                self.best_validation_accuracy = average_validation_accuracy
                print("Found new best accuracy, saving checkpoint ...")
                self.model.save('{}/{}'.format(self.model_dir, 
                                               self.iteration if not output_name else output_name))
    
@tf.function
def train_step(model, opt, loss_func, images, labels, first_batch, batch_size, mixup_alpha=0.0, fp32=False):
    images, labels = mixup(batch_size, mixup_alpha, images, labels)
    with tf.GradientTape() as tape:
        logits = model(images, training=True)
        loss_value = loss_func(labels, tf.cast(logits, tf.float32))
        loss_value += tf.add_n(model.losses)
        if not fp32:
            scaled_loss_value = opt.get_scaled_loss(loss_value)

    tape = dist.DistributedGradientTape(tape, compression=dist.Compression.fp16)
    if not fp32:
        grads = tape.gradient(scaled_loss_value, model.trainable_variables)
        grads = opt.get_unscaled_gradients(grads)
    else:
        grads = tape.gradient(loss_value, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    if first_batch:
        dist.broadcast_variables(model.variables, root_rank=0)
        dist.broadcast_variables(opt.variables(), root_rank=0)
    
    probs = layers.Activation('softmax', dtype='float32')(logits)
    top_1_pred = tf.squeeze(tf.math.top_k(probs, k=1)[1])
    sparse_labels = tf.cast(tf.math.argmax(labels, axis=1), tf.int32)
    top_1_accuracy = tf.math.reduce_sum(tf.cast(tf.equal(top_1_pred, sparse_labels), tf.int32))
    return loss_value, top_1_accuracy


@tf.function
def validation_step(images, labels, model, loss_func):
    pred = model(images, training=False)
    loss = loss_func(labels, pred)
    top_1_pred = tf.squeeze(tf.math.top_k(pred, k=1)[1])
    sparse_labels = tf.cast(tf.math.argmax(labels, axis=1), tf.int32)
    top_1_accuracy = tf.math.reduce_sum(tf.cast(tf.equal(top_1_pred, sparse_labels), tf.int32))
    return loss, top_1_accuracy


