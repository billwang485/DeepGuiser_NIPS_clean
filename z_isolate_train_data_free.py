dirs1 = ['./predictor/finetune_dataset/Apr 24 05 51 48',
    './predictor/finetune_dataset/Apr 24 05 52 17',
    './predictor/finetune_dataset/Apr 24 05 53 20',
    './predictor/finetune_dataset/Apr 24 05 53 57',
    './predictor/finetune_dataset/Apr 24 05 54 32',
    './predictor/finetune_dataset/Apr 24 05 54 59',
    './predictor/finetune_dataset/Apr 24 05 55 18',
    './predictor/finetune_dataset/Apr 24 05 56 25',
    './predictor/finetune_dataset/Apr 24 05 56 42',
    './predictor/finetune_dataset/Apr 24 05 57 02',
    './predictor/finetune_dataset/Apr 24 05 57 30',
    './predictor/finetune_dataset/Apr 24 05 57 45',
    './predictor/finetune_dataset/Apr 24 05 58 06',
    './predictor/finetune_dataset/Apr 24 05 58 22',
    './predictor/finetune_dataset/Apr 24 06 06 58',]


dirs2 = ['./predictor/finetune_dataset/Apr 24 06 09 24']

import os
import torch
target_dir = []
target_arch_list = []
for i, dir in enumerate(dirs1):
    for j in range(1000):
        if os.path.exists(os.path.join(dir, str(j), 'train_data_free')):
            target_dir.extend(torch.load(os.path.join(dir, str(j), 'train_data_free'), map_location = 'cpu'))
            target_arch_list.append(target_dir[-1]['surrogate_arch'])
        else:
            break

surrogate_dir = []
surrogate_arch_list = []
for i, dir in enumerate(dirs2):
    for j in range(1000):
        if os.path.exists(os.path.join(dir, str(j), 'train_data_free')):
            surrogate_dir.extend(torch.load(os.path.join(dir, str(j), 'train_data_free'), map_location = 'cpu'))
            surrogate_arch_list.append(surrogate_dir[-1]['surrogate_arch'])
        else:
            break

target_data = []
for i, data in enumerate(target_dir):
    arch = data['surrogate_arch']
    if not arch in surrogate_arch_list:
        target_data.append(data)

surrogate_data = []
for i, data in enumerate(surrogate_dir):
    arch = data['surrogate_arch']
    if not arch in target_arch_list:
        surrogate_data.append(data)

torch.save(target_data, os.path.join('no_name', 'train_data_free_train'))
torch.save(surrogate_data, os.path.join('no_name', 'train_data_free_test'))
