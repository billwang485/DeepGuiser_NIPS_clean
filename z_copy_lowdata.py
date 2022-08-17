import os
from shutil import copy
import torch
dir1 = ['predictor/train_set/May  5 05 45 01',
'predictor/train_set/May  5 05 46 20',
'predictor/train_set/May  5 05 47 01',
'predictor/train_set/May  5 05 47 29',
# 'predictor/train_set/May  5 05 48 04',
'/home/ChenYu_Wang/nfs/projects/HANAG_nat_search_space/predictor/train_set/May  5 05 48 53',
'predictor/train_set/May  5 05 49 35',
'predictor/train_set/May  5 05 50 51',
'predictor/train_set/May  5 10 54 27',
'predictor/train_set/May  5 10 55 05']
dir = [
'predictor/train_set/May  5 10 56 23',
'predictor/train_set/May  5 10 57 01',
'predictor/train_set/May  5 10 58 10',
'predictor/train_set/May  5 10 59 25'
]
name = 'train_data_constrain_'
count = 0
path1 = 'z/low_data_free'
data = []
for i, path in enumerate(dir):
    if os.path.exists(os.path.join(path, 'train_data_constrain_')):
        count = count + 1
        # import ipdb; ipdb.set_trace()
        # copy(os.path.join(path, name), os.path.join(path1, str(i)))
        tmp = torch.load(os.path.join(path, 'train_data_constrain_'), map_location='cpu')
        data.extend(tmp) 

for path in dir1:
    if os.path.exists(os.path.join(path, 'train_data_free_')):
        count = count + 1
        # copy(os.path.join(path, name), os.path.join(path1, str(i)))
        tmp = torch.load(os.path.join(path, 'train_data_free_'), map_location='cpu')
        data.extend(tmp) 

import random
random.shuffle(data)

torch.save(data, 'no_name/lf_data')
# for i in range(count):
#     tmp = torch.load(os.path.join(path1, str(i)), map_location= 'cpu')
#     for x in tmp:
#         if 'absolulte_reward' in x.keys():
#             x['absolute_reward'] = x['absolulte_reward']
#             x.pop('absolulte_reward')
#     data.extend(tmp)

# torch.save(data, os.path.join(path1, name))