import tensorflow as tf
import random
import numpy as np
from DataProcess import data_save_read
def linear( input, output_dim, scope=None, stddev=1.0):
    norm = tf.random_normal_initializer(stddev=stddev)
    const = tf.constant_initializer(0.0)
    with tf.variable_scope(scope or 'linear'):
        w = tf.get_variable('w', [input.shape()[1], output_dim], initializer=norm)
        b = tf.get_variable('b', [output_dim], initializer=const)
        return tf.matmul(input, w) + b
with tf.Session() as sessions:
    data = data_save_read.read_data('input_data.xlsx')
    random.shuffle(data)
    data_obs = []
    data_label = []
    # argsmain = Params().get_main_args()
    for i in range(len(data)):
        # Get Train Data And Label
        print('i {}, data {}, label {}'.format(i, data[i][0:-1], data[i][-1:]))
        data_obs.append(data[i][0:-1])
        data_label.append(data[i][-1:])
    tf.global_variables_initializer().run()
    in_data = np.hstack((data_obs, data_label))[0:1]
    in_data = list[in_data]
    out = linear(in_data, 1)