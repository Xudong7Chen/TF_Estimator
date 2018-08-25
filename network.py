import tensorflow as tf
import tensorflow.contrib.slim as slim


def prelu(inputs):
    alphas = tf.get_variable("alphas", shape=inputs.get_shape()[-1], dtype=tf.float32, initializer=tf.constant_initializer(0.25))
    pos = tf.nn.relu(inputs)
    neg = alphas * (inputs-abs(inputs))*0.5
    return pos + neg

def Network(inputs, params):
    with tf.name_scope('Network'):
        net = tf.layers.dense(inputs, 128, activation=prelu, name='dense_0')    
        net = tf.layers.dense(net, 64, activation=prelu, name='dense_1')
        net = tf.layers.dense(net, 32, activation=prelu, name='dense_2')
        logits = tf.layers.dense(net, params['num_of_classes'], activation=None, name='dense_3')
    return logits

def my_model_fn(features, labels, mode, params):
    inputs = tf.feature_column.input_layer(features, params['feature_columns'])
    logits = Network(inputs, params)
    predicted_classes = tf.argmax(logits, 1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits}
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    loss = tf.losses.softmax_cross_entropy(labels, logits) 

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    result = tf.one_hot(predicted_classes, 2, 1, 0)
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=result)
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)