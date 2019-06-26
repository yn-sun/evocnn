'''
Length:6, Num:386037, Mean:1.00, Std:0.01,
conv[14,14,9,-0.18,0.68],
conv[18,18,13,-0.26,0.93],
pool[2,2,0.94],
pool[4,4,0.57],
full[1801,0.6176560743803923,0.9038750398909305],
full[2,0.19075779624924527,0.47152749823228046]


'''

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import control_flow_ops
from datetime import datetime
import numpy as np
import os
import get_data as data
import tensorflow as tf

batch_size = 100
total_epochs = 100

def model():
    is_training = tf.placeholder(tf.bool, [])
    train_images, train_label = data.get_train_data(batch_size)
    test_images, test_label = data.get_test_data(batch_size)
    x = tf.cond(is_training, lambda:train_images, lambda:test_images)
    y_ = tf.cond(is_training, lambda:train_label, lambda:test_label)
    y_ = tf.cast(y_, tf.int64)
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.crelu,
                        normalizer_fn=slim.batch_norm,
                        weights_regularizer=slim.l2_regularizer(0.005),
                        normalizer_params={'is_training': is_training, 'decay': 0.95}
                        ):
        conv1 =slim.conv2d(x, 9, [14,14], weights_initializer=tf.truncated_normal_initializer(mean=-0.18, stddev=0.68))
        conv2 =slim.conv2d(conv1, 13, [18,18], weights_initializer=tf.truncated_normal_initializer(mean=-0.26, stddev=0.93))
        pool1 = slim.avg_pool2d(conv2, [2,2], stride=2, padding='SAME')
        pool2 = slim.avg_pool2d(pool1, [4,4], stride=4, padding='SAME')
        flatten = slim.flatten(pool2)
        full1 = slim.fully_connected(flatten, 1801, weights_initializer=tf.truncated_normal_initializer(mean=0.6176560, stddev=0.90), biases_initializer=tf.constant_initializer(0.1, dtype=tf.float32))
        logits = slim.fully_connected(full1, 2, activation_fn=None, weights_initializer=tf.truncated_normal_initializer(mean=0.190, stddev=0.4715), biases_initializer=tf.constant_initializer(0.1, dtype=tf.float32))
        correct_prediction = tf.equal(tf.argmax(logits, 1), y_)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        regularization_loss = tf.add_n(slim.losses.get_regularization_losses())
        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=logits))+ regularization_loss
        step = tf.get_variable("step", [], initializer=tf.constant_initializer(0.0), trainable=False)

#         lr = tf.train.exponential_decay(0.1,
#                                   step,
#                                   550*30,
#                                   0.9,
#                                   staircase=True)
#
#
#         optimizer = tf.train.GradientDescentOptimizer(lr)
        optimizer = tf.train.AdamOptimizer(0.001)
#         lr_summary = tf.summary.scalar('lr', lr)
        train_step = slim.learning.create_train_op(cross_entropy, optimizer, global_step=step)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if update_ops:
            updates = tf.group(*update_ops)
            cross_entropy = control_flow_ops.with_dependencies([updates], cross_entropy)

        loss_summary = tf.summary.scalar('loss', cross_entropy)
        accuracy_summary = tf.summary.scalar('accuracy', accuracy)
        merge_summary = tf.summary.merge([loss_summary, accuracy_summary])
        return is_training, train_step, step, accuracy, cross_entropy, merge_summary

def train():
    gpu_options = tf.GPUOptions(allow_growth=True)
    is_training, train_step, _, accuracy, loss, merge_summary = model()

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        train_data_length = 1000
        test_data_length = 50000
        steps_in_each_epoch = (train_data_length//batch_size)
        total_steps = int(total_epochs*steps_in_each_epoch)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        try:
            for i in range(total_steps):
                if coord.should_stop():
                    break
                _, accuracy_str, loss_str, _ = sess.run([train_step, accuracy,loss, merge_summary], {is_training:True})
                if i % steps_in_each_epoch == 0:
                    test_total_step = test_data_length//batch_size
                    test_accuracy_list = []
                    test_loss_list = []
                    for _ in range(test_total_step):
                        test_accuracy_str, test_loss_str = sess.run([accuracy, loss], {is_training:False})
                        test_accuracy_list.append(test_accuracy_str)
                        test_loss_list.append(test_loss_str)
                    print('{}, {}, Step:{}/{}, train_loss:{}, acc:{}, test_loss:{}, accu:{}'.format(datetime.now(), i // steps_in_each_epoch, i, total_steps, loss_str, accuracy_str, np.mean(test_loss_list), np.mean(test_accuracy_list)))

        except tf.errors.OutOfRangeError:
            print('done')
        finally:
            coord.request_stop()
            coord.join(threads)
if __name__ =='__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    tf.reset_default_graph()
    train()#99.993
