import tensorflow as tf
import fcn

MODEL_SAVE_PATH = "model/"
MODEL_NAME = "model.ckpt"
MOVING_AVERAGE_DECAY = 0.99


class eval_config():
    def __init__(self):

        # self.x = tf.placeholder(tf.float32, [None, fcn.INPUT_NODE], name='x-input')
        # self.y = fcn.interface(self.x, None)
        #
        # self.variable_averages = tf.train.ExponentialMovingAverage(
        #     MOVING_AVERAGE_DECAY)
        # self.variable_to_restore = self.variable_averages.variables_to_restore()
        # self.saver = tf.train.Saver(self.variable_to_restore)
            # self.sess = tf.Session()
        print('eval config load ')

    def eval(self, data):
        with tf.Graph().as_default() as g:
            x = tf.placeholder(tf.float32, [None, fcn.INPUT_NODE], name='x-input')
            y_ = tf.placeholder(tf.float32, [None, fcn.OUTPUT_NODE], name='y-input')
            y = fcn.interface(x, None)

            variable_averages = tf.train.ExponentialMovingAverage(
                MOVING_AVERAGE_DECAY)
            variable_to_restore = variable_averages.variables_to_restore()
            saver = tf.train.Saver(variable_to_restore)
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    print(ckpt.model_checkpoint_path)
                    saver.restore(sess, ckpt.model_checkpoint_path)
                # for i in range(4):
                #     for j in range(10):
                #     print('test eval------------')
                    print(sess.run(y, feed_dict={x: [data[0]], y_: [data[1]]}))

    # def get_batch(self, dataset):
    #     xs = []
    #     ys = []
    #
    #     for i in range(BATCH_SIZE):
    #         xs.append(dataset[0])
    #         ys.append(dataset[i][1])
    #     return xs, ys