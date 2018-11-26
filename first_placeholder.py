import tensorflow as tf
import os
import sys
import numpy as np
import read_cifar

file_names = {'train':['data_batch_%d' % i for i in range(1, 6)], 'eval':['test_batch']}
input_dir = '/ssd3/open_dataset/cifar-10-data/cifar-10-batches-py'

BATCH_SIZE = 64


train_data, train_labels, eval_data, eval_labels = read_cifar.read_numpy(input_dir=input_dir, file_names=file_names)
train_size = train_data.shape[0]


with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [BATCH_SIZE, 32, 32, 3], name='x-input')
    y_ = tf.placeholder(tf.float32, [BATCH_SIZE,], name='y-input')
    tf.summary.image('input', x, 10)

merged = tf.summary.merge_all()

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    writer = tf.summary.FileWriter('./tmp/summary', sess.graph)
    for step in range(3):
        offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
        batch_data = train_data[offset:(offset + BATCH_SIZE), ...]
        batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
        summary = sess.run(merged, feed_dict={x:batch_data, y_:batch_labels})
        writer.add_summary(summary, step)
        print('%s steps' % step)
