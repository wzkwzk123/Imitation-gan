import tensorflow as tf
import numpy as np
import random
from DataProcess import data_save_read
from param import Params
from network import linear

args = Params().get_main_args()



class Gan(object):
    def __init__(self, real_obs, real_action, num_steps, log_every, args):
        self.real_obs = real_obs
        self.real_action = real_action
        self.num_steps = num_steps
        self.pre_batch_size = args.pre_batch_size
        self.batch_size = args.batch_size
        self.d_input_dimension = args.D_input_dimension
        self.g_input_dimension = args.G_input_dimension
        self.g_output_dimension = args.G_output_dimension
        self.d_output_dimension = args.D_output_dimension
        self.log_every = log_every
        self.mlp_hidden_size = 5
        self.learning_rate = 0.0003
        self._create_model()

    def optimizer(self, loss, var_list, initial_learning_rate):
        """
        The Learning Rate is gradually decreasing
        :param loss:
        :param var_list:
        :param initial_learning_rate:
        :return:
        """
        decay = 0.95
        num_decay_step = 150
        batch = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(
            initial_learning_rate,
            batch,
            num_decay_step,
            decay,
            staircase=True
        )
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(
            loss,
            global_step=batch,
            var_list=var_list
        )
        return optimizer

    def discriminator(self, input, h_dim):
        """
        The Discriminator Network
        :return:
        """
        h0 = linear(input, h_dim * 2, 'd0')
        h1 = tf.nn.relu(linear(h0, h_dim * 2, 'd1'))
        h3 = tf.nn.sigmoid(linear(h1, args.D_output_dimension, 'd2'))
        return h3

    def generator(self, input, h_dim):
        """
        The Genarate Network
        :return:
        """
        h0 = tf.nn.softplus(linear(input, h_dim), 'g0')
        h1 = linear(h0, args.G_output_dimension, 'g1')
        return h1

    def _create_model(self):
        with tf.variable_scope('D_pre'):
            """
            The Real Data initial The Discriminant network
            """
            self.pre_input = tf.placeholder(tf.float32, shape=(self.pre_batch_size, self.d_input_dimension))
            self.pre_labels = tf.placeholder(tf.float32, shape=(self.pre_batch_size, self.d_output_dimension))
            # self.pre_d_out = self.linear(self.pre_input, args.D_output_dimension)
            self.D_pre = self.discriminator(self.pre_input, self.mlp_hidden_size)
            self.pre_loss = tf.reduce_mean(tf.square(self.D_pre - self.pre_labels))
            self.pre_opt = self.optimizer(self.pre_loss, None, self.learning_rate)

        with tf.variable_scope('Gen'):
            self.z = tf.placeholder(tf.float32, shape=(self.batch_size, self.g_input_dimension))
            self.G = self.generator(self.z, self.mlp_hidden_size)

        with tf.variable_scope('Disc') as scope:
            self.x = tf.placeholder(tf.float32, shape=(self.batch_size, self.d_input_dimension)) # Real data
            self.D1 = self.discriminator(self.x, self.mlp_hidden_size)
            scope.reuse_variables()
            self.D2 = self.discriminator(tf.concat([self.z, self.G], 1), self.mlp_hidden_size)

        self.loss_d = tf.reduce_mean(-tf.log(self.D1) - tf.log(1 - self.D2))
        self.loss_g = tf.reduce_mean(-tf.log(self.D2))

        self.d_pre_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D_pre')
        self.d_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Disc')
        self.g_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Gen')

        self.opt_d = self.optimizer(self.loss_d, self.d_params, self.learning_rate)
        self.opt_g = self.optimizer(self.loss_g, self.g_params, self.learning_rate)

    def train(self):
        real_obs_data = self.real_obs[0:self.pre_batch_size]
        real_action_data = self.real_action[0:self.pre_batch_size]

        real_obs_train_data = self.real_obs[self.batch_size:self.batch_size*2]
        real_action_train_data = self.real_action[self.batch_size:self.batch_size*2]

        real_all_data = np.hstack((real_obs_data, real_action_data))
        real_all_train_data = np.hstack((real_obs_train_data, real_action_train_data))
        with tf.Session() as session:
            tf.global_variables_initializer().run()
            num_pretrain_step = 2

            for step in range(num_pretrain_step):
                d = real_all_data
                # self.pre_labels = tf.ones_like(self.D1)

                pretrain_loss, _ = session.run([self.pre_loss, self.pre_opt], {
                    self.pre_input:np.reshape(d, (self.pre_batch_size, self.d_input_dimension)),
                    self.pre_labels:np.ones((self.pre_batch_size, self.d_output_dimension))
                })
                # print("")
            self.weightsD = session.run(self.d_pre_params)
            # copy weights from pre-training over to new D network
            for i, v in enumerate(self.d_params):
                session.run(v.assign(self.weightsD[i]))
            self.weightsD = session.run(self.d_params)
            for step in range(self.num_steps):
                # update discriminator
                d = session.run([self.D1], {self.x: real_all_train_data})
                # print("d",d)
                loss_d, _ = session.run([self.loss_d, self.opt_d],{
                    self.x: real_all_train_data,
                    self.z: real_obs_train_data
                })
                # update generator
                loss_g, _ = session.run([self.loss_g, self.opt_g], {
                    self.z:real_obs_train_data
                })

                if step % self.log_every == 0:
                    print('{}: {}\t{}'.format(step, loss_d, loss_g))

if __name__ == "__main__":
    data = data_save_read.read_data('input_data.xlsx')
    random.shuffle(data)
    data_obs = []
    data_label = []
    argsmain = Params().get_main_args()
    for i in range(len(data)):
        # Get Train Data And Label
        print('i {}, data {}, label {}'.format(i, data[i][0:-1], data[i][-1:]))
        data_obs.append(data[i][0:-1])
        data_label.append(data[i][-1:])
    # real_obs, real_action, num_steps, log_every, args
    model = Gan(data_obs, data_label, 1000, 10, argsmain)
    model.train()














