import os
import tensorflow as tf
import fcn
from tensorflow.examples.tutorials.mnist import input_data

BATCH_SIZE = 4

LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULA_RATE = 0.0001
TRAING_STEP = 30000
MOVING_AVERAGE_DECAY = 0.99


MODEL_SAVE_PATH = "model/"
MODEL_NAME = "model.ckpt"


# def cost_fun():
#
#
#     return

class train_config():
    def __init__(self):
        self.x = tf.placeholder(tf.float32, [None, fcn.INPUT_NODE], name='x-input')
        self.y_ = tf.placeholder(tf.float32, [None, fcn.OUTPUT_NODE], name='y-input')

        self.regularizer = tf.contrib.layers.l2_regularizer(REGULA_RATE)

        self.y = fcn.interface(self.x, self.regularizer)
        self.global_step = tf.Variable(0, trainable=False)

        self.variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY, self.global_step)
        self.variable_averages_op = self.variable_averages.apply(tf.trainable_variables())

        self.cost_function = tf.reduce_sum(tf.pow(self.y_ - self.y))
        # cost = cost_fun(x, y)
        self.loss = self.cost_function + tf.add_n(tf.get_collection('losses'))
        self.learning_rate = tf.train.exponential_decay(
            LEARNING_RATE_BASE, self.global_step, 50, LEARNING_RATE_DECAY)
        self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate) \
            .minimize(self.loss, global_step=self.global_step)
        with tf.control_dependencies([self.train_step, self.variable_averages_op]):
            self.train_op = tf.no_op(name='train')
        self.saver = tf.train.Saver(max_to_keep=1)
        # self.sess = tf.Session()
        print('train config load ')

    def get_batch(self, dataset):
        xs = []
        ys = []

        for i in range(BATCH_SIZE):
            xs.append(dataset[i][0])
            ys.append(dataset[i][1])
        return xs, ys


if __name__ == '__main__':

    pass