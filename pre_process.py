import numpy as np
from utils import array2onehot

def data_process(file_list, split=True):
    for name in file_list:
        data_class = name.split('.')[0]
        f1 = open('./data/origin/%s' % name)
        data = f1.readlines()[1:]
        num_data = len(data)
        f2 = open('./data/process/data_train.txt', 'a')
        f3 = open('./data/process/data_valid.txt', 'a')

        train_num = int(num_data * 0.7) if split else num_data
        print('train_num: %d' % train_num)
        print('valid_num: %d' % (num_data-train_num))

        for index, line in enumerate(data):
            line = line.strip().split('\t')
            line[1], line[2] = ' '.join(line[1].split('.')), ' '.join(line[2].split('.'))
            line = ' '.join(line)
            line = data_class + ',' + line + '\n'
            if index < train_num:
                f2.write(line)
            else:
                f3.write(line)

data_process(['0.txt','1.txt'])