import torch
import os

train_set = torch.load('no_name/train_data_free_train', map_location='cpu')

test_set = torch.load('no_name/train_data_free_test', map_location='cpu')

pos = 0
for i in range(1000):
    mylist = []
    if len(train_set) - 1 >= 100*i - 99:
        mylist = train_set[100*(i):100*(i + 1)]
        torch.save(mylist, os.path.join('predictor/4099', "piece_{}".format(i)))

torch.save(test_set, os.path.join('predictor/4099', "test"))