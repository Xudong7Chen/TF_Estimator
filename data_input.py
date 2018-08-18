import numpy as np
import tensorflow as tf

def get_data(name, batch_size, data_shape=16, train=True):
    with tf.name_scope('input_pipeline'):
        filename_queue = tf.train.string_input_producer([name],shuffle=True)
        # read tfrecord
        reader = tf.TextLineReader()
        key, value = reader.read(filename_queue)
        record_defaults = [['0'], ['1 111 186 105 83 122 228 239 16 50268 80 506 1056 39265 1559430 7']]
        label, feature = tf.decode_csv(value, field_delim=',', record_defaults=record_defaults)

        feature = tf.string_split([feature], ' ').values
        feature = tf.string_to_number(feature)
        feature = tf.reshape(feature, [data_shape])
        label = tf.cast(tf.string_to_number(label), tf.uint8)
        label = tf.one_hot(label, 2, 1 ,0)
        label = tf.cast(label, tf.float32)
        feature, label = tf.train.shuffle_batch(
            [feature, label], batch_size=batch_size,
            capacity=1000+3*batch_size, min_after_dequeue=2*batch_size,
            num_threads=32
        )

        feature = tf.reshape(feature, [batch_size, data_shape])
        label = tf.reshape(label, [batch_size, 2])

    return feature, label