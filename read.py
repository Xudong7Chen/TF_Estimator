import numpy as np

f = open('bbr.txt')

lines = f.readlines()
database = []
single_data = []

i = 0
for line in lines:
    if '#' not in line:
        line = line.strip('\n').split(' ')
        single_data.append(line)
    else:
        database.append(single_data)
        single_data = []
    #     i = i + 1
    # if i == 100 :
    #     break

database = np.asarray(database).astype(np.float)
database = np.reshape(database, (-1,2751,7))