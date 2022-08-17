import torch
import os

x = torch.load('predictor/hf_data/train_data_constrain_4', map_location='cpu')

for i in range(int(len(x) / 4)):
    # mean = x[4 * i]['acc_adv_baseline'] + x[4 * i + 1]['acc_adv_baseline'] + x[4 * i + 2]['acc_adv_baseline'] + x[4 * i + 3]['acc_adv_baseline']
    # mean = mean / 4
    x[4 * i]['relative_reward'] = x[4 * i]['absolute_reward'] / ( x[4 * i]['acc_clean_target']/100 - x[4 * i]['acc_adv_baseline'])
    x[4 * i + 1]['relative_reward'] = x[4 * i + 1]['absolute_reward']  / ( x[4 * i]['acc_clean_target']/100 - x[4 * i]['acc_adv_baseline'])
    x[4 * i + 2]['relative_reward'] = x[4 * i + 2]['absolute_reward']  / ( x[4 * i]['acc_clean_target']/100 - x[4 * i]['acc_adv_baseline'])
    x[4 * i + 3]['relative_reward'] = x[4 * i + 3]['absolute_reward'] / ( x[4 * i]['acc_clean_target']/100 - x[4 * i]['acc_adv_baseline'])

torch.save(x, 'predictor/hf_data/train_data_constrain_4_n1')