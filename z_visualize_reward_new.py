import os
import matplotlib.pyplot as plt
import numpy as np
import torch
# dirs = ['./predictor/finetune_dataset/Apr 24 05 51 48',
#     './predictor/finetune_dataset/Apr 24 05 52 17',
#     './predictor/finetune_dataset/Apr 24 05 53 20',
#     './predictor/finetune_dataset/Apr 24 05 53 57',
#     './predictor/finetune_dataset/Apr 24 05 54 32',
#     './predictor/finetune_dataset/Apr 24 05 54 59',
#     './predictor/finetune_dataset/Apr 24 05 55 18',
#     './predictor/finetune_dataset/Apr 24 05 56 25',
#     './predictor/finetune_dataset/Apr 24 05 56 42',
#     './predictor/finetune_dataset/Apr 24 05 57 02',
#     './predictor/finetune_dataset/Apr 24 05 57 30',
#     './predictor/finetune_dataset/Apr 24 05 57 45',
#     './predictor/finetune_dataset/Apr 24 05 58 06',
#     './predictor/finetune_dataset/Apr 24 05 58 22',
#     './predictor/finetune_dataset/Apr 24 06 06 58',]
#     #  './predictor/finetune_dataset/Apr 24 06 09 24']
# # train_data = []
# # for dir in dirs:
# #     tmp = torch.load(os.path.join(dir, 'train_data_constrain_'), map_location='cpu')
# #     for tmp1 in tmp:
# #         train_data.append(tmp1[0])

# # torch.save(train_data, 'predictor/finetune_dataset/train_data_constrain_')
# reward_list = []
# train_data_constrain = []
# # target_info = torch.load('no_name/rr_target_dir', map_location='cpu')
# # surrogate_info = torch.load('no_name/rr_surrogate_path', map_location='cpu')

# # max_index = len(surrogate_info) - 1

# for dir in dirs:
#     for j in range(1000):
#         if not os.path.exists(os.path.join(dir, str(j))):
#             break
#         if not os.path.exists(os.path.join(dir, str(j), 'train_data_{}'.format(j))):
#             print('{} 不存在'.format(os.path.join(dir, str(j), 'train_data_{}'.format(j))))
#             continue
#         rewards = torch.load(os.path.join(dir, str(j), 'train_data_{}'.format(j)), map_location='cpu')
#         # if len(rewards) != 4:
#         #     print('{} 长度不对'.format(os.path.join(dir, str(j), 'train_data_{}'.format(j))))
#         #     s_arch_list = []
#         #     for reward in rewards:
#         #         s_arch_list.append(reward['surrogate_arch'])
#         #     for k in range(4):
#         #         error_pos = []
#         #         if os.path.exists(os.path.join(dir, str(j), 'surrogate_arch_{}'.format(k))):
#         #             surrogate_arch = torch.load(os.path.join(dir, str(j), 'surrogate_arch_{}'.format(k)))
#         #             if surrogate_arch not in s_arch_list:
#         #                 error_pos.append(k)
#         #         else:
#         #             error_pos.append(k)
#         #     if len(error_pos) > 0:
#         #         print(error_pos)
#         #     if len(rewards) > 4:
#         #         print(len(rewards))
#         #         for a, s_arch in enumerate(s_arch_list):
#         #             for b, s1_arch in enumerate(s_arch_list):
#         #                 if s_arch == s1_arch and a != b:
#         #                     print('{} and {} 重合'.format(a, b))

#         #     continue
#         if len(rewards) < 4:
#             continue
#         if len(rewards) > 4:
#             rewards = rewards[0:4]
#         for reward in rewards:
#             reward_list.append(reward['absolute_reward'])
#             train_data_constrain.append(reward)
# torch.save(train_data_constrain, 'no_name/train_data_constrain_')
x = torch.load('final_train_data3', map_location = 'cpu')
reward_list = [y['absolute_reward'] for y in x]
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

plt.savefig(os.path.join('visualization', 'hf_absolute_reward_null.png'))