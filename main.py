import tensorflow as tf
import numpy as np
from network import my_model_fn
from utils import cal_acc
import os 
from config import get_config
from data_input import get_data 

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
para, _ = get_config()

tf.logging.set_verbosity(tf.logging.INFO)
feature_columns = [tf.feature_column.numeric_column(key='data', shape=(16,))]
classifier = tf.estimator.Estimator(model_fn=my_model_fn,
                                    model_dir='./models',
                                    params={'feature_columns': feature_columns,
                                            'num_of_classes': 2})
# a = classifier.train(input_fn=lambda:get_data(para.file_train, para.batch_size), steps=100000)
a = classifier.evaluate(input_fn=lambda:get_data(para.file_valid, para.batch_size))
