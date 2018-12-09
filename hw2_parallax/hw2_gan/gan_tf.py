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
    python gan_tf.py --ps_hosts=localhost:12345 --worker_hosts=localhost:12346,localhost:12347 --job_name=ps --task_index=0 --max_steps=1000
    python gan_tf.py --ps_hosts=localhost:12345 --worker_hosts=localhost:12346,localhost:12347 --job_name=worker --task_index=0 --max_steps=1000
    python gan_tf.py --ps_hosts=localhost:12345 --worker_hosts=localhost:12346,localhost:12347 --job_name=worker --task_index=1 --max_steps=1000

Example command for examining the checkpoint file:
    python ~/parallax/tensorflow/tensorflow/python/tools/inspect_checkpoint.py --file_name=tf_ckpt/model.ckpt-0 --tensor_name=conv1/kernel
"""

from __future__ import division, print_function, absolute_import
# MatPlotLib without GUI
# https://stackoverflow.com/questions/4706451/how-to-save-a-figure-remotely-with-pylab
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt

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

tf.app.flags.DEFINE_integer('max_steps', 10000, """Number of iterations to run for each workers.""")
tf.app.flags.DEFINE_integer('log_frequency', 1, """How many steps between two runop logs.""")
# tf.app.flags.DEFINE_integer('batch_size', 32, """Batch size""")

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

'''

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
'''

# Training Params
# num_steps = 10000 --> max_steps
# batch_size = 128
batch_size = 32 # ORG: 128
learning_rate = 0.0002

# Network Params
image_dim = 784 # 28*28 pixels
gen_hidden_dim = 256
disc_hidden_dim = 256
noise_dim = 100 # Noise data points


# A custom initialization (see Xavier Glorot init)
def glorot_init(shape):
    tf.set_random_seed(1234)
    return tf.random_normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.))

# Store layers weight & bias
weights = {
    'gen_hidden1': tf.Variable(glorot_init([noise_dim, gen_hidden_dim])),
    'gen_out': tf.Variable(glorot_init([gen_hidden_dim, image_dim])),
    'disc_hidden1': tf.Variable(glorot_init([image_dim, disc_hidden_dim])),
    'disc_out': tf.Variable(glorot_init([disc_hidden_dim, 1])),
}
biases = {
    'gen_hidden1': tf.Variable(tf.zeros([gen_hidden_dim])),
    'gen_out': tf.Variable(tf.zeros([image_dim])),
    'disc_hidden1': tf.Variable(tf.zeros([disc_hidden_dim])),
    'disc_out': tf.Variable(tf.zeros([1])),
}

# Generator
def generator(x):
    hidden_layer = tf.matmul(x, weights['gen_hidden1'])
    hidden_layer = tf.add(hidden_layer, biases['gen_hidden1'])
    hidden_layer = tf.nn.relu(hidden_layer)
    out_layer = tf.matmul(hidden_layer, weights['gen_out'])
    out_layer = tf.add(out_layer, biases['gen_out'])
    out_layer = tf.nn.sigmoid(out_layer)
    return out_layer

# Discriminator
def discriminator(x):
    hidden_layer = tf.matmul(x, weights['disc_hidden1'])
    hidden_layer = tf.add(hidden_layer, biases['disc_hidden1'])
    hidden_layer = tf.nn.relu(hidden_layer)
    out_layer = tf.matmul(hidden_layer, weights['disc_out'])
    out_layer = tf.add(out_layer, biases['disc_out'])
    out_layer = tf.nn.sigmoid(out_layer)
    return out_layer


with tf.device(tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d" % FLAGS.task_index,
            cluster=cluster)):
    # ops = lenet()
    # logits = ops['logits']
    # x = ops['images']
    # y = ops['labels']
    # is_training = ops['is_training']
    # global_step = ops['global_step']

    np.random.seed(0)

    # Build Networks
    # Network Inputs
    gen_input = tf.placeholder(tf.float32, shape=[None, noise_dim], name='input_noise')
    disc_input = tf.placeholder(tf.float32, shape=[None, image_dim], name='disc_input')

    global_step = tf.train.get_or_create_global_step()

    # Build Generator Network
    gen_sample = generator(gen_input)

    # Build 2 Discriminator Networks (one from noise input, one from generated samples)
    disc_real = discriminator(disc_input)
    disc_fake = discriminator(gen_sample)

    print("========================")
    print("    Device: %d" % FLAGS.task_index)
    print("========================")

    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits))
    # loss += weight_decay * tf.losses.get_regularization_loss()
    # acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=1), tf.argmax(y, axis=1)), tf.float32))

    # Build Loss
    gen_loss = -tf.reduce_mean(tf.log(disc_fake))
    # disc_loss = -tf.reduce_mean(tf.log(disc_real) + tf.log(1. - disc_fake))
    disc_loss = gen_loss -tf.reduce_mean(tf.log(disc_real) + tf.log(1. - disc_fake))


    # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    # optimizer = tf.train.SyncReplicasOptimizer(
    #     optimizer,
    #     replicas_to_aggregate=num_workers,
    #     total_num_replicas=num_workers)

    # Build Optimizers
    optimizer_gen = tf.train.AdamOptimizer(learning_rate=learning_rate)
    optimizer_gen = tf.train.SyncReplicasOptimizer(
        optimizer_gen,
        replicas_to_aggregate=num_workers,
        total_num_replicas=num_workers)

    optimizer_disc = tf.train.AdamOptimizer(learning_rate=learning_rate)
    optimizer_disc = tf.train.SyncReplicasOptimizer(
        optimizer_disc,
        replicas_to_aggregate=num_workers,
        total_num_replicas=num_workers)

    # Generator Network Variables
    gen_vars = [weights['gen_hidden1'], weights['gen_out'],
                biases['gen_hidden1'], biases['gen_out']]
    # Discriminator Network Variables
    disc_vars = [weights['disc_hidden1'], weights['disc_out'],
                biases['disc_hidden1'], biases['disc_out']]

    # train_op = optimizer.minimize(loss, global_step=global_step)
    train_gen = optimizer_gen.minimize(gen_loss, var_list=gen_vars, global_step=global_step)
    train_disc = optimizer_disc.minimize(disc_loss, var_list=disc_vars, global_step=global_step)


    is_chief = (FLAGS.task_index == 0)
    # sync_replicas_hook_gen = optimizer_gen.make_session_run_hook(is_chief)
    sync_replicas_hook_disc = optimizer_disc.make_session_run_hook(is_chief)
    # sync_replicas_hook = optimizer.make_session_run_hook(is_chief, num_tokens=0)

    saver = tf.train.Saver(tf.global_variables(), save_relative_paths=False, allow_empty=True, max_to_keep=10000)
    tf.add_to_collection(tf.GraphKeys.SAVERS, saver)
    scaffold = tf.train.Scaffold(saver=saver)
    ckpt_hook = tf.train.CheckpointSaverHook('tf_ckpt', save_steps=1, scaffold=scaffold)

                                # hooks=[sync_replicas_hook_gen, sync_replicas_hook_disc],
with tf.train.MonitoredTrainingSession(master=server.target,
                                is_chief=is_chief,
                                hooks=[sync_replicas_hook_disc],
                                chief_only_hooks=[ckpt_hook]) as sess:
    start = time.time()
    for i in range(FLAGS.max_steps):
        # batch = mnist.train.next_batch(FLAGS.batch_size, shuffle=False)
        batch_x, _ = mnist.train.next_batch(batch_size, shuffle=False)


        # Generate noise to feed to the generator
        z = np.random.uniform(-1., 1., size=[batch_size, noise_dim])

        # _, loss_ = sess.run([train_op, loss],
        #             feed_dict={x: batch[0], y: batch[1], is_training: True})

        # Train
        _, _, gl, dl = sess.run([train_gen, train_disc, gen_loss, disc_loss],
                                feed_dict={disc_input: batch_x, gen_input: z})


        if i % FLAGS.log_frequency == 0:
            end = time.time()
            throughput = float(FLAGS.log_frequency) / float(end - start)
            # acc_ = sess.run(acc, feed_dict={x: mnist.test.images, y: mnist.test.labels, is_training: False})
            # print("step: %d, test accuracy: %lf, throughput: %f steps/sec" % (i, acc_, throughput))
            print('step %i: Generator Loss: %f, Discriminator Loss: %f, throughput: %f steps/sec' % (i, gl, dl, throughput))

            start = time.time()

        # if i % 1000 == 0 or i == 1:
        #     print('Step %i: Generator Loss: %f, Discriminator Loss: %f' % (i, gl, dl))


    # Generate images from noise, using the generator network.
    f, a = plt.subplots(4, 10, figsize=(10, 4))
    for i in range(10):
        # Noise input.
        z = np.random.uniform(-1., 1., size=[4, noise_dim])
        g = sess.run([gen_sample], feed_dict={gen_input: z})
        g = np.reshape(g, newshape=(4, 28, 28, 1))
        # Reverse colours for better display
        g = -1 * (g - 1)
        for j in range(4):
            # Generate image from noise. Extend to 3 channels for matplot figure.
            img = np.reshape(np.repeat(g[j][:, :, np.newaxis], 3, axis=2),
                             newshape=(28, 28, 3))
            a[j][i].imshow(img)

    # f.show()
    # plt.draw()

    print("==========================")
    print("SAVE IMAGE")
    print("==========================")

    # f.savefig('out1.png')
    plt.savefig('./out/out_single_itr-' + str(FLAGS.max_steps) + '_dev-' + str(FLAGS.task_index) + '.png')
    # plt.waitforbuttonpress()










