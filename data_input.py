import numpy as np
import tensorflow as tf

def get_data(name, batch_size=1, data_shape=16, train=True):
    def parse_function(example_proto):
        parse_dict = {'data':tf.FixedLenFeature(shape=(16) , dtype=tf.int64, default_value=None),\
                      'label':tf.FixedLenFeature(shape=(1), dtype=tf.int64, default_value=None)}
        parse_example = tf.parse_single_example(example_proto, parse_dict)
        parse_example['label'] = tf.one_hot(parse_example['label'], 2)
        return parse_example

    with tf.name_scope('input_pipeline'):
        dataset = tf.data.TFRecordDataset([name])
        dataset = dataset.map(parse_function)
        dataset = dataset.shuffle(buffer_size=1500)
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()
    return next_element

# next_element = get_data('./data/tfrecord/data_train.tfrecord')
# sess = tf.InteractiveSession()
# while True:
#     try:
#         data, label = sess.run([next_element['data'], next_element['label']])
#         print(data, label)
#     except tf.errors.OutOfRangeError:
#         print('Finish...')
#         break