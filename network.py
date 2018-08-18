import tensorflow as tf
import tensorflow.contrib.slim as slim

def prelu(inputs):
    alphas = tf.get_variable("alphas", shape=inputs.get_shape()[-1], dtype=tf.float32, initializer=tf.constant_initializer(0.25))
    pos = tf.nn.relu(inputs)
    neg = alphas * (inputs-abs(inputs))*0.5
    return pos + neg

def gen_graph(inputs, label_batch, num_of_class):
    with tf.name_scope('Network'):
        net = tf.layers.dense(inputs, 128, activation=prelu, name='dense_0')    
        net = tf.layers.dense(net, 64, activation=prelu, name='dense_1')
        net = tf.layers.dense(net, 32, activation=prelu, name='dense_2')
        net = tf.layers.dense(net, num_of_class, activation=None, name='dense_3')
        
        loss_visible = tf.nn.softmax(net, name='softmax_loss_visible')
        loss = tf.losses.softmax_cross_entropy(label_batch, net) 
        return loss, net, loss_visible