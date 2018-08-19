import tensorflow as tf
import numpy as np
from network import gen_graph
from utils import cal_acc
import os 
from config import get_config
from data_input import get_data 

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
para, _ = get_config()


print('Creating input pipeline...') #数据以#号结束
train_feature, train_label = get_data(para.file_train, para.batch_size)
valid_feature, valid_label = get_data(para.file_valid, para.batch_size)
print('Success.')    

train_data_num = para.train_data_num
valid_data_num = para.valid_data_num

total_steps = int(para.total_epochs * train_data_num / para.batch_size)
epoch_learning_rate = para.ini_learning_rate

# creat graph---------------------------------------------------------------------------------------------------------------------------------------
inputs_placeholder = tf.placeholder(tf.float32, shape=[None, para.data_shape])
label_placeholder = tf.placeholder(tf.float32, shape=[None, 2])
global_step = tf.Variable(0, trainable=False, name='global_step')

loss, network_output, loss_visible = gen_graph(inputs_placeholder, label_placeholder, 2)
optimizer = tf.train.AdamOptimizer(epoch_learning_rate)
train_op = optimizer.minimize(loss, global_step)
saver = tf.train.Saver(max_to_keep=4)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
epoch = 0
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
# sess = tf_debug.LocalCLIDebugWrapperSession(sess)
# sess = tf_debug.TensorBoardDebugWrapperSession(sess, 'localhost:6064')

#visualize some variables
tf.summary.scalar("cls_loss", loss)#cls_loss
summary_op = tf.summary.merge_all()

writer = tf.summary.FileWriter('./logs', sess.graph)

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

if para.is_train == True:
    sess.run(tf.global_variables_initializer())
    for step in range(1, total_steps):

        if (step * para.batch_size) % train_data_num == 0:
            epoch = epoch + 1
        training_array, training_label = sess.run([train_feature, train_label])
        train_feed_dict = {
            inputs_placeholder: training_array,
            label_placeholder: training_label}

        batch_loss, batch_output, _, summary = sess.run([loss, network_output, train_op, summary_op], feed_dict=train_feed_dict)

        if epoch == (para.total_epochs * 0.5) or epoch == (para.total_epochs * 0.7):
            epoch_learning_rate = epoch_learning_rate / 10

        if step % 2000 == 0:
            print('Validing----------------------------------')
            for _ in range(5):
                validing_array, validing_label = sess.run([valid_feature, valid_label])
                valid_feed_dict = {
                    inputs_placeholder: validing_array,
                    label_placeholder: validing_label}
                valid_batch_loss, valid_batch_output = sess.run([loss, network_output], feed_dict=valid_feed_dict)
                test_accuracy = cal_acc(validing_label, valid_batch_output, para.batch_size)
                print('batch_accuracy: %f' % test_accuracy)
                print('batch_loss: %f' % valid_batch_loss)
            print('Finish Validing---------------------------')

        if step % 20 == 0:
            batch_accuracy = cal_acc(training_label, batch_output, para.batch_size)
            print('steps: %d / %d' % (step, total_steps))
            print('accuracy: %f' % batch_accuracy)
            print('loss: %f' % batch_loss)
            print('cur_lr: %f' % epoch_learning_rate)
            print('------------------------------------------')
            
        if step % 2000 == 0:
            #store model
            print('storing model')
            saver.save(sess=sess, save_path='./models/zzc_%d' % step)
        
        writer.add_summary(summary,global_step=step)

coord.request_stop()
coord.join(threads)
sess.close()