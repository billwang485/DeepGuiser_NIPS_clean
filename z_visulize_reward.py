import os
import matplotlib.pyplot as plt
import numpy as np
import torch
# dirs = [
#     './predictor/finetune_dataset/May  1 14 22 19',
#     './predictor/finetune_dataset/May  1 14 26 12',
#     './predictor/finetune_dataset/May  1 14 27 31',
#     './predictor/finetune_dataset/May  1 14 29 14',
#     './predictor/finetune_dataset/May  1 14 30 12'
# ]
# train_data = []
# for dir in dirs:
#     tmp = torch.load(os.path.join(dir, 'train_data_constrain_'), map_location='cpu')
#     for tmp1 in tmp:
#         train_data.append(tmp1[0])

# torch.save(train_data, 'predictor/finetune_dataset/train_data_constrain_')
reward_list = []
# for data_point in train_data:
#     reward_list.append(data_point['relative_reward'])

y = torch.load('predictor/lf_data/train_data_free_', map_location='cpu')

for tmp in y:
    reward_list.append(tmp['absolute_reward'].item())

reward_list = np.array(reward_list, dtype=float)

variance = np.var(reward_list)
mean = np.mean(reward_list)
min_reward = np.min(reward_list)
max_reward = np.max(reward_list)
total_step = 100
step = (max_reward - min_reward) / total_step
freqs = [0] * total_step
fig, ax = plt.subplots()  

plt.hist(reward_list, bins=total_step, density=True)
plt.text(0.5, 1.1,'var = {:.3f} mean = {:.3f} min={:.3f} max={:.3f}'.format(variance, mean, min_reward, max_reward),
    horizontalalignment='center',
    verticalalignment='center',
    transform = ax.transAxes)
ax.set_ylabel('num')
ax.set_xlabel('absolute_reward')

plt.savefig(os.path.join('visualization', 'lf_absolute_reward_free.png'))