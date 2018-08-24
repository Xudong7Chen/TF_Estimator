import tensorflow as tf
import numpy as np
from network import gen_graph
from utils import cal_acc
import os 
from config import get_config
from data_input import get_data 

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
para, _ = get_config()


def my_model_fn(features, labels, mode, params):

    def prelu(inputs):
        alphas = tf.get_variable("alphas", shape=inputs.get_shape()[-1], dtype=tf.float32, initializer=tf.constant_initializer(0.25))
        pos = tf.nn.relu(inputs)
        neg = alphas * (inputs-abs(inputs))*0.5
        return pos + neg

    inputs = tf.feature_column.input_layer(features, params['feature_columns'])
    with tf.name_scope('Network'):
        net = tf.layers.dense(inputs, 128, activation=prelu, name='dense_0')    
        net = tf.layers.dense(net, 64, activation=prelu, name='dense_1')
        net = tf.layers.dense(net, 32, activation=prelu, name='dense_2')
        logits = tf.layers.dense(net, params['num_of_classes'], activation=None, name='dense_3')
        predicted_classes = tf.argmax(logits, 1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits}
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    
    loss = tf.losses.softmax_cross_entropy(labels, logits) 
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
    
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=logits)
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)


tf.logging.set_verbosity(tf.logging.INFO)
feature_columns = [tf.feature_column.numeric_column(key='data', shape=(16,))]
classifier = tf.estimator.Estimator(model_fn=my_model_fn,
                                    model_dir='./models',
                                    params={'feature_columns': feature_columns,
                                            'num_of_classes': 2})
a = classifier.predict(input_fn=lambda:get_data(para.file_train, para.batch_size))
for i in a:
    print(i)