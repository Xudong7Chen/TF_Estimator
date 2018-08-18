import tensorflow as tf
import os
import sys

def int_feature(x):
    if not isinstance(x, list):
        raise Exception('Input must be list!')
    return tf.train.Feature(int64_list=tf.train.Int64List(value=x))

def float_feature(x):
    if not isinstance(x, list):
        raise Exception('Input must be list!')
    return tf.train.Feature(float_list=tf.train.FloatList(value=x))

def string_feature(x):
    if not isinstance(x, list):
        raise Exception('Input must be list!')
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=x))

base_dir = './data/tfrecord'
data_dir = './data/process/data_train.txt'
tfrecord_name = 'data_train.tfrecord'

if not tf.gfile.Exists(base_dir):
    tf.gfile.MakeDirs(base_dir)

file_dir = os.path.join(base_dir, tfrecord_name)
rd_writer = tf.python_io.TFRecordWriter(file_dir)

data_train = open(data_dir).readlines()

trans_num = 0
for i in data_train:
    label, data = i.strip().split(',')
    data = [int(s) for s in data.split(' ')]
    label = [int(label)]
    features = {}
    # features['data'] = string_feature([data.encode()])
    # features['label'] = string_feature([label.encode()])
    features['data'] = int_feature(data)
    features['label'] = int_feature(label)
    tf_features = tf.train.Features(feature=features)
    tf_example = tf.train.Example(features=tf_features)
    tf_serialized = tf_example.SerializeToString()
    rd_writer.write(tf_serialized)

    trans_num = trans_num + 1
    sys.stdout.write('\r>> Successfully transform %d examples.' % trans_num)
    sys.stdout.flush()

rd_writer.close()
