from typing import OrderedDict
import torch
import os

data = torch.load('predictor/hf_data/train_data_constrain_4', map_location='cpu')

data1 = torch.load('no_name/train_data_free_train', map_location='cpu')
import numpy as np
split = int(np.floor(len(data) * 0.9 / 4) * 4)
train_data = data[:split]
test_data = data[split:]
target_list = []
surrogate_list = []
for dp in test_data:
    target_list.append(dp['target_arch'])
    surrogate_list.append(dp['surrogate_arch']) 
for i, d1 in enumerate(data1):
    if not d1['target_arch'] in target_list and not d1['surrogate_arch'] in surrogate_list:
        train_data.append(d1)

torch.save(train_data, 'predictor/hf_data/train_data_enhanced')
torch.save(test_data, 'predictor/hf_data/my_test_data')

new_data = []

for td in train_data:
    t = OrderedDict()
    t['target_arch'] = td['target_arch']
    t['surrogate_arch'] = td['surrogate_arch']
    t['absolute_reward'] = float(td['absolute_reward'])
    new_data.append(t)

torch.save(new_data, 'predictor/hf_data/train_data_enhanced')
new_data = []

for td in test_data:
    t = OrderedDict()
    t['target_arch'] = td['target_arch']
    t['surrogate_arch'] = td['surrogate_arch']
    t['absolute_reward'] = float(td['absolute_reward'])
    new_data.append(t)

torch.save(new_data, 'predictor/hf_data/my_test_data')