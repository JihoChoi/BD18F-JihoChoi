# ================================================================================
#
#   BDDL 2018 HW 02 Distributed Deep Learning Traning
#                           with Vanilla TensorFlow, Horovod, Parallax
#
#   Author: Jiho Choi (jihochoi@snu.ac.kr)
#
#   Generative Adversarial Networks (GAN) model by Aymeric Damien
#       - https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/gan.py
#
# ================================================================================

"""
Example command for running this script:
    python gan_tf.py --ps_hosts=localhost:12345 --worker_hosts=localhost:12346,localhost:12347 --job_name=ps --task_index=0 --max_steps=100
    python gan_tf.py --ps_hosts=localhost:12345 --worker_hosts=localhost:12346,localhost:12347 --job_name=worker --task_index=0 --max_steps=100
    python gan_tf.py --ps_hosts=localhost:12345 --worker_hosts=localhost:12346,localhost:12347 --job_name=worker --task_index=1 --max_steps=100

Example command for examining the checkpoint file:
    python ~/parallax/tensorflow/tensorflow/python/tools/inspect_checkpoint.py --file_name=tf_ckpt/model.ckpt-0 --tensor_name=conv1/kernel
"""

import os
import time
import numpy as np
import tensorflow as tf
import parallax

# import model
# from model import lenet
from tensorflow.examples.tutorials.mnist import input_data

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('ps_hosts', None, """Comma-separated list of hostname:port pairs""")
tf.app.flags.DEFINE_string('worker_hosts', None, """Comma-separated list of hostname:port pairs""")
tf.app.flags.DEFINE_string('job_name', None, """One of 'ps', 'worker'""")
tf.app.flags.DEFINE_integer('task_index', 0, """Index of task within the job""")

tf.app.flags.DEFINE_integer('max_steps', 1000000, """Number of iterations to run for each workers.""")
tf.app.flags.DEFINE_integer('log_frequency', 50, """How many steps between two runop logs.""")
tf.app.flags.DEFINE_integer('batch_size', 32, """Batch size""")

ps_hosts = FLAGS.ps_hosts.split(",")
worker_hosts = FLAGS.worker_hosts.split(",")
num_workers = len(worker_hosts)

# Create a cluster from the parameter server and worker hosts.
cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

# Create and start a server for the local task.
server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

if FLAGS.job_name == "ps":

    print("========================")
    print("    Parameter Server    ")
    print("========================")
    server.join()
assert FLAGS.job_name == "worker"

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


learning_rate = 1e-3
weight_decay = 1e-4
num_classes = 10
dropout_keep_prob = 0.5

def lenet():
    np.random.seed(0)
    images_ph = tf.placeholder(tf.float32, shape=[None, 784])
    labels_ph = tf.placeholder(tf.int64, shape=[None, num_classes])
    is_training_ph = tf.placeholder(tf.bool, shape=())

    global_step = tf.train.get_or_create_global_step()

    images = tf.reshape(images_ph, [-1, 28, 28, 1])

    net = tf.layers.conv2d(images, 10, [5, 5],
                        activation=tf.nn.relu,
                        kernel_initializer=tf.constant_initializer(np.random.randn(5, 5, 1, 10) * 1e-2),
                        kernel_regularizer=tf.nn.l2_loss,
                        name='conv1')
    net = tf.layers.max_pooling2d(net, [2, 2], 2, name='pool1')
    net = tf.layers.conv2d(net, 20, [5, 5],
                        activation=tf.nn.relu,
                        kernel_initializer=tf.constant_initializer(np.random.randn(5, 5, 10, 20) * 1e-2),
                        kernel_regularizer=tf.nn.l2_loss,
                        name='conv2')
    net = tf.layers.max_pooling2d(net, [2, 2], 2, name='pool2')
    net = tf.layers.flatten(net)

    net = tf.layers.dense(net, 50,
                        activation=tf.nn.relu,
                        kernel_initializer=tf.constant_initializer(np.random.randn(320, 50) * 1e-2),
                        kernel_regularizer=tf.nn.l2_loss,
                        name='fc3')
    net = tf.layers.dropout(net, 1 - dropout_keep_prob, training=is_training_ph, name='dropout3')
    logits = tf.layers.dense(net, num_classes,
                        activation=None,
                        kernel_initializer=tf.constant_initializer(np.random.randn(50, 10) * 1e-2),
                        kernel_regularizer=tf.nn.l2_loss,
                        name='fc4')

    return {'logits': logits,
            'images': images_ph,
            'labels': labels_ph,
            'is_training': is_training_ph,
            'global_step': global_step}


with tf.device(tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d" % FLAGS.task_index,
            cluster=cluster)):
    ops = lenet()
    logits = ops['logits']
    x = ops['images']
    y = ops['labels']
    is_training = ops['is_training']
    global_step = ops['global_step']

    print("========================")
    print("    Device: %d" % FLAGS.task_index)
    print("========================")

    loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits))
    loss += weight_decay * tf.losses.get_regularization_loss()
    acc = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(logits, axis=1), tf.argmax(y, axis=1)),
            tf.float32))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    optimizer = tf.train.SyncReplicasOptimizer(
        optimizer,
        replicas_to_aggregate=num_workers,
        total_num_replicas=num_workers)
    train_op = optimizer.minimize(loss, global_step=global_step)

    is_chief = (FLAGS.task_index == 0)
    sync_replicas_hook = optimizer.make_session_run_hook(is_chief)
    # sync_replicas_hook = optimizer.make_session_run_hook(is_chief, num_tokens=0)

    saver = tf.train.Saver(tf.global_variables(), save_relative_paths=False, allow_empty=True, max_to_keep=1000000)
    tf.add_to_collection(tf.GraphKeys.SAVERS, saver)
    scaffold = tf.train.Scaffold(saver=saver)
    ckpt_hook = tf.train.CheckpointSaverHook('tf_ckpt', save_steps=1, scaffold=scaffold)

with tf.train.MonitoredTrainingSession(master=server.target,
                                is_chief=is_chief,
                                hooks=[sync_replicas_hook],
                                chief_only_hooks=[ckpt_hook]) as sess:
    start = time.time()
    for i in range(FLAGS.max_steps):
        batch = mnist.train.next_batch(FLAGS.batch_size, shuffle=False)
        _, loss_ = sess.run([train_op, loss], feed_dict={x: batch[0], y: batch[1], is_training: True})

        if i % FLAGS.log_frequency == 0:
            end = time.time()
            throughput = float(FLAGS.log_frequency) / float(end - start)
            acc_ = sess.run(acc, feed_dict={x: mnist.test.images, y: mnist.test.labels, is_training: False})
            print("step: %d, test accuracy: %lf, throughput: %f steps/sec" % (i, acc_, throughput))
            start = time.time()
