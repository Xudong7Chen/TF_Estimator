import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--split', type=float, default=True)
parser.add_argument('--is_train', type=bool, default=True)
parser.add_argument('--ini_learning_rate', type=float, default=0.001)
parser.add_argument('--total_epochs', type=int, default=300)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--data_shape', type=int, default=16)
parser.add_argument('--train_data_num', type=int, default=1550)
parser.add_argument('--valid_data_num', type=int, default=666)
parser.add_argument('--file_train', type=str, default='./data/tfrecord/data_train.tfrecord')
parser.add_argument('--file_valid', type=str, default='./data/tfrecord/data_valid.tfrecord')

def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed