import tensorflow as tf
import numpy as np
from network import gen_graph
from utils import cal_acc
import os 
from config import get_config
from data_input import get_data 

tf.logging.set_verbosity(tf.logging.INFO)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
para, _ = get_config()
111
def my_model(features, labels, mode, params):

    def prelu(inputs):
        alphas = tf.get_variable("alphas", shape=inputs.get_shape()[-1], dtype=tf.float32, initializer=tf.constant_initializer(0.25))
        pos = tf.nn.relu(inputs)
        neg = alphas * (inputs-abs(inputs))*0.5
        return pos + neg
        
    labels = features['label']
    inputs = tf.feature_column.input_layer(features, params['feature_columns'])
    net = tf.layers.dense(inputs, 128, activation=prelu, name='dense_0')    
    net = tf.layers.dense(net, 64, activation=prelu, name='dense_1')
    net = tf.layers.dense(net, 32, activation=prelu, name='dense_2')
    logits = tf.layers.dense(net, params["num_of_class"], activation=None, name='dense_3')
    predicted_class = tf.argmax(logits, 1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_class,
            'probabilities': tf.nn.softmax(logits),
            'logits': logits
        }
        return tf.estimator.EstimatorSpec(mode, predictions)

    loss = tf.losses.softmax_cross_entropy(labels, logits) 
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predicted_class)
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
    


train = get_data(para.file_train, para.batch_size)
valid = get_data(para.file_valid, para.batch_size)
my_feature_columns = [tf.feature_column.numeric_column(key='data')]

classifier = tf.estimator.Estimator(
    model_fn=my_model,
    params={
        'feature_columns': my_feature_columns,
        'num_of_class': 2
        })

classifier.train(
    input_fn=lambda:get_data(para.file_train, para.batch_size),
    steps=100000)

eval_result = classifier.evaluate(
    input_fn=lambda:get_data(para.file_valid, para.batch_size))
print(eval_result)
