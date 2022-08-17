import os
import torch
reward_list = []
for i in range(1000):
    if os.path.exists(os.path.join('predictor/4099', 'piece_{}'.format(i))):
        reward_list.extend(torch.load(os.path.join('predictor/4099', 'piece_{}'.format(i)), map_location='cpu'))
    else:
        break

torch.save(reward_list, os.path.join('predictor/4099', 'train_data_free_train'))
from shutil import copy
copy(os.path.join('predictor/4099', 'test'), os.path.join('predictor/4099', 'train_data_free_test'))
# torch.save(reward_list, os.path.join('predictor/4099', 'train_data_free_train'))