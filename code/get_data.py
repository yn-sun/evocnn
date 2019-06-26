import tensorflow as tf
import scipy.io as io
import numpy as np
import sklearn.preprocessing as pre
import os

def get_general_image(path, name, num):
    data = io.loadmat(path)
    data = data[name].astype(np.float32)
    data = np.reshape(data, [num, 28, 28, 1], order='F')
    return data
def get_general_label(path, name):
    label = io.loadmat(path)
    label = label[name]
    label = np.squeeze(label.astype(np.int32))
    return label

def get_mnist_train_data():
    train_images_path = '/am/lido/home/yanan/training_data/rectangles_images/train_images.mat'
    train_label_path = '/am/lido/home/yanan/training_data/rectangles_images/train_label.mat'

    train_data = get_general_image(train_images_path, 'train_images', 10000)
    train_label = get_general_label(train_label_path, 'train_label')


    return train_data, train_label


def get_mnist_test_data():
    test_images_path = '/am/lido/home/yanan/training_data/rectangles_images/test_images.mat'
    test_label_path = '/am/lido/home/yanan/training_data/rectangles_images/test_label.mat'


    test_data = get_general_image(test_images_path, 'test_images', 50000)
    test_label = get_general_label(test_label_path, 'test_label')

    return test_data, test_label


def get_mnist_validate_data():
    validate_images_path = '/am/lido/home/yanan/training_data/rectangles_images/validate_images.mat'
    validate_label_path = '/am/lido/home/yanan/training_data/rectangles_images/validate_label.mat'

    validate_data = get_general_image(validate_images_path, 'validate_images', 2000)
    validate_label = get_general_label(validate_label_path, 'validate_label')

    return  validate_data, validate_label


def get_standard_train_data(name):
    data_path = '/am/lido/home/yanan/training_data/back-{}/train_images.npy'.format(name)
    label_path = '/am/lido/home/yanan/training_data/back-{}/train_label.npy'.format(name)
    data = np.load(data_path)
    label = np.load(label_path)
    return data, label

def get_standard_validate_data(name):
    data_path = '/am/lido/home/yanan/training_data/back-{}/validate_images.npy'.format(name)
    label_path = '/am/lido/home/yanan/training_data/back-{}/validate_label.npy'.format(name)
    data = np.load(data_path)
    label = np.load(label_path)
    return data, label

def get_standard_test_data(name):
    data_path = '/am/lido/home/yanan/training_data/back-{}/test_images.npy'.format(name)
    label_path = '/am/lido/home/yanan/training_data/back-{}/test_label.npy'.format(name)
    data = np.load(data_path)
    label = np.load(label_path)
    return data, label



def get_train_data(batch_size):
    t_image, t_label = get_mnist_train_data()
    train_image = tf.cast(t_image, tf.float32)
    train_label = tf.cast(t_label, tf.int32)
    single_image, single_label  = tf.train.slice_input_producer([train_image, train_label], shuffle=True)
    single_image = tf.image.per_image_standardization(single_image)
    image_batch, label_batch = tf.train.batch([single_image, single_label], batch_size=batch_size, num_threads=2, capacity=batch_size*3)
    return image_batch, label_batch

def get_validate_data(batch_size):
    t_image, t_label = get_mnist_validate_data()
    validate_image = tf.cast(t_image, tf.float32)
    validate_label = tf.cast(t_label, tf.int32)
    single_image, single_label  = tf.train.slice_input_producer([validate_image, validate_label], shuffle=False)
    single_image = tf.image.per_image_standardization(single_image)
    image_batch, label_batch = tf.train.batch([single_image, single_label], batch_size=batch_size, num_threads=2, capacity=batch_size*3)
    return image_batch, label_batch


def get_test_data(batch_size):
    t_image, t_label = get_mnist_test_data()
    test_image = tf.cast(t_image, tf.float32)
    test_label = tf.cast(t_label, tf.int32)
    single_image, single_label  = tf.train.slice_input_producer([test_image, test_label], shuffle=False)
    single_image = tf.image.per_image_standardization(single_image)
    image_batch, label_batch = tf.train.batch([single_image, single_label], batch_size=batch_size, num_threads=2, capacity=batch_size*3)
    return image_batch, label_batch

def tf_standalized(data):
    image = tf.placeholder(tf.float32, shape=[28,28,1])
    scale_data = tf.image.per_image_standardization(image)
    data_list = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        data_length = data.shape[0]
        for i in range(data_length):
            standard_data = sess.run(scale_data, {image:data[i]})
            print(i, data_length)
            data_list.append(standard_data)
    return np.array(data_list)


if __name__ =='__main__':
    name = 'random'
    data, label = get_standard_test_data(name)
    print(data.shape, label.shape, data.dtype, label.dtype)








# def get_mnist_train_batch(batch_size, capacity=1000):
#     mnist = input_data.read_data_sets('MNIST_data', reshape=False, one_hot=True)
#     train_images = tf.cast(mnist.train.images, tf.float32)
#     train_labels = tf.cast(mnist.train.labels, tf.int32)
#     input_queue = tf.train.slice_input_producer([train_images, train_labels],shuffle=True, capacity=capacity, name='input_queue')
#     images_batch, labels_batch = tf.train.batch(input_queue, batch_size=batch_size, num_threads=3, capacity=capacity)
#     return images_batch, labels_batch
# batch_images, batch_labels = get_mnist_train_batch(100)
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         coord = tf.train.Coordinator()
#         threads = tf.train.start_queue_runners(sess, coord)
#         i = 0
#         try:
#             while not coord.should_stop():
#
#                 batch_images_v, batch_labels_v = sess.run([batch_images, batch_labels])
#                 print(i)
#                 i += 1
#
#         except tf.errors.OutOfRangeError:
#             print('done')
#         finally:
#             coord.request_stop()
#             coord.join(threads)