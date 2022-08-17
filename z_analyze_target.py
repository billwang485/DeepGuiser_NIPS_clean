import os
import torch
import matplotlib.pyplot as plt
import numpy as np
train_data = torch.load('no_name/train_data_free_train', map_location='cpu')
target_list = []
for i, data_point in enumerate(train_data):
    if not data_point['target_arch'] in target_list:
        target_list.append(data_point['target_arch'])

all_data_list = []
for i, target in enumerate(target_list):
    data_list = []
    for j, data_point in enumerate(train_data):
        if data_point['target_arch'] == target:
            data_list.append(data_point)
    torch.save(data_list, os.path.join('z/analysis', 'target_{}'.format(i)))
    all_data_list.append(data_list)

all_reward_list = []
reward_mean_one_target = []
reward_var_one_target = []
for i,data_list in enumerate(all_data_list):
    tmp = []
    for data_point in data_list:
        tmp.append(data_point['absolute_reward'])
    reward_mean_one_target.append(np.mean(tmp))
    reward_var_one_target.append(np.var(tmp))
    all_reward_list.append(tmp)
print(np.mean(reward_var_one_target))
print(np.var(reward_mean_one_target))
torch.save(reward_mean_one_target, 'z/analysis/reward_mean_one_target')
torch.save(reward_var_one_target, 'z/analysis/reward_var_one_target')
torch.save(all_reward_list, 'z/analysis/all_reward_list')
torch.save(all_data_list, 'z/analysis/all_data_list')
torch.save(np.var(reward_mean_one_target), 'z/analysis/var_of_different_target_mean')

variance = np.var(reward_mean_one_target)
mean = np.mean(reward_mean_one_target)
min_reward = np.min(reward_mean_one_target)
max_reward = np.max(reward_mean_one_target)
total_step = 10
step = (max_reward - min_reward) / total_step
freqs = [0] * total_step
fig, ax = plt.subplots()  

plt.hist(reward_mean_one_target, bins=total_step, density=True)
plt.text(0.5, 1.1,'var = {:.3f} mean = {:.3f} min={:.3f} max={:.3f}'.format(variance, mean, min_reward, max_reward),
    horizontalalignment='center',
    verticalalignment='center',
    transform = ax.transAxes)
ax.set_ylabel('num')
ax.set_xlabel('absolute_reward')

plt.savefig(os.path.join('visualization', 'reward_mean_one_target.png'))

import torch
# from math import mean
reward_list_ = []
for reward_list in all_reward_list:
    reward_list = reward_list - np.mean(reward_list)
    reward_list_.extend(reward_list)
# x = torch.Tensor(all_reward_list)
# reward_list_ = x.view(-1).tolist()

variance = np.var(reward_list_)
mean = np.mean(reward_list_)
min_reward = np.min(reward_list_)
max_reward = np.max(reward_list_)
total_step = 10
step = (max_reward - min_reward) / total_step
freqs = [0] * total_step
fig, ax = plt.subplots()  

plt.hist(reward_list_, bins=total_step, density=True)
plt.text(0.5, 1.1,'var = {:.3f} mean = {:.3f} min={:.3f} max={:.3f}'.format(variance, mean, min_reward, max_reward),
    horizontalalignment='center',
    verticalalignment='center',
    transform = ax.transAxes)
ax.set_ylabel('num')
ax.set_xlabel('absolute_reward')

plt.savefig(os.path.join('visualization', 'reward_list_minus_one_target_mean.png'))
