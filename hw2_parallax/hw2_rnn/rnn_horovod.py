"""
Example command for running this script:
mpirun --mca btl_vader_single_copy_mechanism none --allow-run-as-root -bind-to none -map-by slot -mca orte_base_help_aggregate 0 -x NCCL_DEBUG=INFO -np 2 -H localhost:2 python rnn_horovod.py --max_steps=500

Example command for examining the checkpoint file:
python <PARALLAX_HOME>/tensorflow/tensorflow/python/tools/inspect_checkpoint.py --file_name=hvd_ckpt/model.ckpt-0 --tensor_name=conv1/kernel
"""

import os
import time
import numpy as np
import tensorflow as tf
import horovod.tensorflow as hvd

# import model
# from model import lenet
from tensorflow.examples.tutorials.mnist import input_data

hvd.init()

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of iterations to run for each workers.""")
tf.app.flags.DEFINE_integer('log_frequency', 50,
                            """How many steps between two runop logs.""")
tf.app.flags.DEFINE_integer('batch_size', 32,
                            """Batch size""")

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

learning_rate = 1e-3
weight_decay = 1e-4
num_classes = 10
dropout_keep_prob = 0.5
time_step = 28

def rnn(only_logits=False):
    tf.set_random_seed(1234)
    np.random.seed(1234)

    # images_ph = tf.placeholder(tf.float32, shape=[None, 784])
    images_ph = tf.placeholder(tf.float32, [None, time_step * 28])

    # labels_ph = tf.placeholder(tf.int64, shape=[None, num_classes])
    labels_ph = tf.placeholder(tf.int64, [None, num_classes])

    is_training_ph = tf.placeholder(tf.bool, shape=())

    global_step = tf.train.get_or_create_global_step()

    # images = tf.reshape(images_ph, [-1, 28, 28, 1])
    image = tf.reshape(images_ph, [-1, time_step, 28])

    # RNN
    rnn_cell = tf.nn.rnn_cell.LSTMCell(num_units=64)
    outputs, (h_c, h_n) = tf.nn.dynamic_rnn(rnn_cell, image,
            initial_state=None, dtype=tf.float32, time_major=False)
    logits = tf.layers.dense(outputs[:, -1, :], num_classes)

    return {'logits': logits,
        'images': images_ph,
        'labels': labels_ph,
        'is_training': is_training_ph,
        'global_step': global_step}


ops = rnn(only_logits=True)
logits = ops['logits']
x = ops['images']
y = ops['labels']
is_training = ops['is_training']
global_step = ops['global_step']

loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits))
loss += weight_decay * tf.losses.get_regularization_loss()
acc = tf.reduce_mean(tf.cast(
    tf.equal(tf.argmax(logits, axis=1), tf.argmax(y, axis=1)),
    tf.float32))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
optimizer = hvd.DistributedOptimizer(optimizer)
train_op = optimizer.minimize(loss, global_step=global_step)

hooks = [hvd.BroadcastGlobalVariablesHook(0)]
if hvd.rank() == 0:
  saver = tf.train.Saver(tf.global_variables(), save_relative_paths=False,
                         allow_empty=True, max_to_keep=1000000)
  tf.add_to_collection(tf.GraphKeys.SAVERS, saver)
  scaffold = tf.train.Scaffold(saver=saver)
  ckpt_hook = tf.train.CheckpointSaverHook('hvd_ckpt',
                                           save_steps=1,
                                           scaffold=scaffold)
  hooks.append(ckpt_hook)

with tf.train.MonitoredTrainingSession(hooks=hooks) as sess:
  start = time.time()
  for i in range(FLAGS.max_steps):
    batch = mnist.train.next_batch(FLAGS.batch_size, shuffle=False)
    _, loss_ = sess.run([train_op, loss], feed_dict={x: batch[0],
                                                     y: batch[1],
                                                     is_training: True})
    if i % FLAGS.log_frequency == 0:
      end = time.time()
      throughput = float(FLAGS.log_frequency) / float(end - start)
      acc_ = sess.run(acc, feed_dict={x: mnist.test.images,
                                      y: mnist.test.labels,
                                      is_training: False})
      print("step: %d, test accuracy: %lf, throughput: %f steps/sec"
            % (i, acc_, throughput))
      start = time.time()
