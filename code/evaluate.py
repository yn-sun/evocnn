import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from population import Population
from individual import Individual
import numpy as np
import collections
import timeit
import os
import pickle
import utils
from datetime import datetime
import get_data




class Evaluate:

    def __init__(self, pops, train_data, train_label, validate_data, validate_label, number_of_channel, epochs, batch_size, train_data_length, validate_data_length):
        self.pops = pops
        self.train_data = train_data # train or test data. data[0] is images and data[1] are label
        self.train_label = train_label
        self.validate_data = validate_data
        self.validate_label = validate_label
        self.number_of_channel = number_of_channel
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_data_length = train_data_length
        self.validate_data_length = validate_data_length
    '''
    Parse the chromosome in the population to the information which can be directly employed by TensorFLow
    '''
    def parse_population(self, gen_no):
        save_dir = os.getcwd() + '/save_data/gen_{:03d}'.format(gen_no)
        tf.gfile.MakeDirs(save_dir)
        history_best_score = 0
        for i in range(self.pops.get_pop_size()):
            indi = self.pops.get_individual_at(i)
            rs_mean, rs_std, num_connections, new_best = self.parse_individual(indi, self.number_of_channel, i, save_dir, history_best_score)
            #rs_mean, rs_std, num_connections, new_best = np.random.random(), np.random.random(), np.random.random_integers(1000, 100000), -1
            indi.mean = rs_mean
            indi.std = rs_std
            indi.complxity = num_connections
            history_best_score = new_best
            list_save_path = os.getcwd() + '/save_data/gen_{:03d}/pop.txt'.format(gen_no)
            utils.save_append_individual(str(indi), list_save_path)

        pop_list = self.pops
        list_save_path = os.getcwd() + '/save_data/gen_{:03d}/pop.dat'.format(gen_no)
        with open(list_save_path, 'wb') as file_handler:
            pickle.dump(pop_list, file_handler)


    def build_graph(self, indi_index, num_of_input_channel, indi, train_data, train_label, validate_data, validate_label):
        is_training = tf.placeholder(tf.bool, [])
        X = tf.cond(is_training, lambda:train_data, lambda:validate_data)
        y_ = tf.cond(is_training, lambda:train_label, lambda:validate_label)
        true_Y = tf.cast(y_, tf.int64)

        name_preffix = 'I_{}'.format(indi_index)
        num_of_units = indi.get_layer_size()

        ################# variable for convolution operation#######################################
        last_output_feature_map_size = num_of_input_channel
        ############### state the connection numbers################################################
        num_connections = 0
        ############################################################################################
        output_list = []
        output_list.append(X)
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.crelu,
                        normalizer_fn=slim.batch_norm,
                        #weights_regularizer=slim.l2_regularizer(0.005),
                        normalizer_params={'is_training': is_training, 'decay': 0.99}):

            for i in range(num_of_units):
                current_unit = indi.get_layer_at(i)
                if current_unit.type == 1:
                    name_scope = '{}_conv_{}'.format(name_preffix, i)
                    with tf.variable_scope(name_scope):
                        filter_size = [current_unit.filter_width, current_unit.filter_height]
                        mean=current_unit.weight_matrix_mean
                        stddev=current_unit.weight_matrix_std
                        conv_H = slim.conv2d(output_list[-1], current_unit.feature_map_size, filter_size, weights_initializer=tf.truncated_normal_initializer(mean=mean, stddev=stddev), biases_initializer=init_ops.constant_initializer(0.1, dtype=tf.float32))
                        output_list.append(conv_H)
                        # update for next usage
                        last_output_feature_map_size = current_unit.feature_map_size
                        num_connections += current_unit.feature_map_size*current_unit.filter_width*current_unit.filter_height+current_unit.feature_map_size
                elif current_unit.type == 2:
                    with tf.variable_scope('{}_pool_{}'.format(name_preffix, i)):
                        kernel_size = [current_unit.kernel_width, current_unit.kernel_height]
                        if current_unit.kernel_type < 0.5:
                            pool_H = slim.max_pool2d(output_list[-1], kernel_size=kernel_size, stride=kernel_size, padding='SAME')
                        else:
                            pool_H = slim.avg_pool2d(output_list[-1], kernel_size=kernel_size, stride=kernel_size, padding='SAME')
                        output_list.append(pool_H)
                        # pooling operation does not change the number of channel size, but channge the output size
                        last_output_feature_map_size = last_output_feature_map_size
                        num_connections += last_output_feature_map_size
                elif current_unit.type == 3:
                    with tf.variable_scope('{}_full_{}'.format(name_preffix, i)):
                        last_unit = indi.get_layer_at(i-1)
                        if last_unit.type != 3: # use the previous setting to calculate this input dimension
                            input_data =  slim.flatten(output_list[-1])
                            input_dim = input_data.get_shape()[1].value
                        else: # current input dim should be the number of neurons in the previous hidden layer
                            input_data = output_list[-1]
                            input_dim = last_unit.hidden_neuron_num
                        mean=current_unit.weight_matrix_mean
                        stddev=current_unit.weight_matrix_std
                        if i < num_of_units - 1:
                            full_H = slim.fully_connected(input_data, num_outputs=current_unit.hidden_neuron_num, weights_initializer=tf.truncated_normal_initializer(mean=mean, stddev=stddev), biases_initializer=init_ops.constant_initializer(0.1, dtype=tf.float32))
                        else:
                            full_H = slim.fully_connected(input_data, num_outputs=current_unit.hidden_neuron_num, activation_fn=None, weights_initializer=tf.truncated_normal_initializer(mean=mean, stddev=stddev), biases_initializer=init_ops.constant_initializer(0.1, dtype=tf.float32))
                        output_list.append(full_H)
                        num_connections += input_dim*current_unit.hidden_neuron_num + current_unit.hidden_neuron_num
                else:
                    raise NameError('No unit with type value {}'.format(current_unit.type))


            with tf.name_scope('{}_loss'.format(name_preffix)):
                logits = output_list[-1]
                #regularization_loss = tf.add_n(tf.losses.get_regularization_losses())
                cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_Y, logits=logits))
            with tf.name_scope('{}_train'.format(name_preffix)):
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                if update_ops:
                    updates = tf.group(*update_ops)
                    cross_entropy = control_flow_ops.with_dependencies([updates], cross_entropy)
                #global_step = tf.get_variable("global_step", [], initializer=tf.constant_initializer(0.0), trainable=False)
                #self.train_data_length//self.batch_size
#                 lr = tf.train.exponential_decay(0.1, step, 550*30, 0.9, staircase=True)
#                 optimizer = tf.train.GradientDescentOptimizer(lr)
                optimizer = tf.train.AdamOptimizer()
                train_op = slim.learning.create_train_op(cross_entropy, optimizer)
            with tf.name_scope('{}_test'.format(name_preffix)):
                accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), true_Y), tf.float32))

            tf.summary.scalar('loss', cross_entropy)
            tf.summary.scalar('accuracy', accuracy)
            merge_summary = tf.summary.merge_all()

            return is_training, train_op, accuracy, cross_entropy, num_connections, merge_summary



    def parse_individual(self, indi, num_of_input_channel, indi_index, save_path, history_best_score):
        tf.reset_default_graph()
        train_data, train_label = get_data.get_train_data(self.batch_size)
        validate_data, validate_label = get_data.get_validate_data(self.batch_size)
        is_training, train_op, accuracy, cross_entropy, num_connections, merge_summary = self.build_graph(indi_index, num_of_input_channel, indi, train_data, train_label, validate_data, validate_label)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            steps_in_each_epoch = (self.train_data_length//self.batch_size)
            total_steps = int(self.epochs*steps_in_each_epoch)
            coord = tf.train.Coordinator()
            #threads = tf.train.start_queue_runners(sess, coord)
            try:
                threads = []
                for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                    threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))
                for i in range(total_steps):
                    if coord.should_stop():
                        break
                    _, accuracy_str, loss_str, _ = sess.run([train_op, accuracy,cross_entropy, merge_summary], {is_training:True})
                    if i % (2*steps_in_each_epoch) == 0:
                        test_total_step = self.validate_data_length//self.batch_size
                        test_accuracy_list = []
                        test_loss_list = []
                        for _ in range(test_total_step):
                            test_accuracy_str, test_loss_str = sess.run([accuracy, cross_entropy], {is_training:False})
                            test_accuracy_list.append(test_accuracy_str)
                            test_loss_list.append(test_loss_str)
                        mean_test_accu = np.mean(test_accuracy_list)
                        mean_test_loss = np.mean(test_loss_list)
                        print('{}, {}, indi:{}, Step:{}/{}, train_loss:{}, acc:{}, test_loss:{}, acc:{}'.format(datetime.now(), i // steps_in_each_epoch, indi_index, i, total_steps, loss_str, accuracy_str, mean_test_loss, mean_test_accu))
                        #print('{}, test_loss:{}, acc:{}'.format(datetime.now(), loss_str, accuracy_str))
                #validate the last epoch
                test_total_step = self.validate_data_length//self.batch_size
                test_accuracy_list = []
                test_loss_list = []
                for _ in range(test_total_step):
                    test_accuracy_str, test_loss_str = sess.run([accuracy, cross_entropy], {is_training:False})
                    test_accuracy_list.append(test_accuracy_str)
                    test_loss_list.append(test_loss_str)
                mean_test_accu = np.mean(test_accuracy_list)
                mean_test_loss = np.mean(test_loss_list)
                print('{}, test_loss:{}, acc:{}'.format(datetime.now(), mean_test_loss, mean_test_accu))
                mean_acc = mean_test_accu
                if mean_acc > history_best_score:
                    save_mean_acc = tf.Variable(-1, dtype=tf.float32, name='save_mean')
                    save_mean_acc_op = save_mean_acc.assign(mean_acc)
                    sess.run(save_mean_acc_op)
                    saver0 = tf.train.Saver()
                    saver0.save(sess, save_path +'/model')
                    saver0.export_meta_graph(save_path +'/model.meta')
                    history_best_score = mean_acc

            except Exception as e:
                print(e)
                coord.request_stop(e)
            finally:
                print('finally...')
                coord.request_stop()
                coord.join(threads)

            return mean_test_accu, np.std(test_accuracy_list), num_connections, history_best_score


