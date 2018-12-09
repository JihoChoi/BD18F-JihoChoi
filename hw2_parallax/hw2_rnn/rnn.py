# ================================================================================
#
#   BDDL 2018 HW 02 Distributed Deep Learning Traning
#                           with Vanilla TensorFlow, Horovod, Parallax
#
#   Author: Jiho Choi (jihochoi@snu.ac.kr)
#
#   RNN / LSTM
#       - https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/gan.py
#       - https://ratsgo.github.io/natural%20language%20processing/2017/03/09/rnnlstm/
# ================================================================================

"""
Example command for running this script:
    python rnn_tf.py --ps_hosts=localhost:12345 --worker_hosts=localhost:12346,localhost:12347 --job_name=ps --task_index=0 --max_steps=1000
    python rnn_tf.py --ps_hosts=localhost:12345 --worker_hosts=localhost:12346,localhost:12347 --job_name=worker --task_index=0 --max_steps=1000
    python rnn_tf.py --ps_hosts=localhost:12345 --worker_hosts=localhost:12346,localhost:12347 --job_name=worker --task_index=1 --max_steps=1000

Example command for examining the checkpoint file:
    python ~/parallax/tensorflow/tensorflow/python/tools/inspect_checkpoint.py --file_name=tf_ckpt/model.ckpt-0 --tensor_name=conv1/kernel
"""


import tensorflow as tf
import time

from tensorflow.examples.tutorials.mnist import input_data
from model import lenet

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('max_steps', 1000,
                            """Number of iterations to run for each workers.""")
tf.app.flags.DEFINE_integer('log_frequency', 50,
                            """How many steps between two runop logs.""")
tf.app.flags.DEFINE_integer('batch_size', 32,
                            """Batch size""")

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

ops = lenet()
train_op = ops['train_op']
loss = ops['loss']
acc = ops['acc']
x = ops['images']
y = ops['labels']
is_training = ops['is_training']


with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  start = time.time()
  for i in range(FLAGS.max_steps):
    batch = mnist.train.next_batch(FLAGS.batch_size)
    _, loss_ = sess.run([train_op, loss], feed_dict={x: batch[0],
                                                     y: batch[1],
                                                     is_training: True})
    if i % FLAGS.log_frequency == 0:
      end = time.time()
      throughput = float(FLAGS.log_frequency) / float(end - start)
      acc_ = sess.run(acc, feed_dict={x: mnist.test.images,
                                      y: mnist.test.labels,
                                      is_training: False})
      print("step %d, test accuracy %lf, throughput: %f steps/sec" % (i, acc_, throughput))
      start = time.time()
