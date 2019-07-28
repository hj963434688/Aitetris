import tensorflow as tf

#              矩阵   形状  形态 坐标
INPUT_NODE = 10 * 20 + 7 + 4 + 10
#            高度  空白
OUTPUT_NODE = 1 + 1

HIDLAYER_NODE = 100


def get_weight_variable(shape, regularizer):
    weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.8))
    if regularizer != None:
        tf.add_to_collection('losses', regularizer(weights))
    return weights


def interface(input_tensor, regularizer):
    with tf.variable_scope('hidden_layer'):
        weights = get_weight_variable([INPUT_NODE, HIDLAYER_NODE], regularizer)
        biases = tf.get_variable("biases", [HIDLAYER_NODE], initializer=tf.constant_initializer(0.5))
        hidelayer = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    with tf.variable_scope('out_layer'):
        weights = get_weight_variable([HIDLAYER_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.5))
        outlayer = tf.nn.relu(tf.matmul(hidelayer, weights) + biases)

    return outlayer
