import numpy as np

def array2onehot(array):
    # one hot encode
    onehot_encoded = list()
    for value in array:
           letter = [0 for _ in range(5)]
           letter[value] = 1
           onehot_encoded.append(letter)
    return np.asarray(onehot_encoded)

def cal_acc(label, output, batch_size):
    output = np.argmax(output, axis=1)
    label = np.argmax(label, axis=1)
    temp = output - label
    total = 0
    for i in temp:
        if i == 0:
            total = total + 1
    return total / batch_size