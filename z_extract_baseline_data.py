f=open('z/baseline/May 14 18 22 26/log.txt')
txt=[]
for line in f:
    txt.append(line.strip())
print(txt)

supernet_reward_list = []
supernet_acc_adv_list = []
supernet_acc_clean_list = []
for x in txt:
    if 'absolute_reward' in x:
        tmp = x.index('absolute_reward') + 16
        supernet_reward_list.append(str(x[tmp: tmp + 4]))
        tmp = x.index('surrogate_acc_adv') + 18
        supernet_acc_adv_list.append(str(x[tmp: tmp + 4]))
        tmp = x.index('target_acc_clean') + 17
        supernet_acc_clean_list.append(str(x[tmp: tmp + 4]))

import torch
import os
a = torch.load('final_test_data514', map_location='cpu')
reward_list = []
# target_acc_clean_list = []
acc_adv_list = []
for x in a:
    reward_list.append(str(x['absolute_reward']))
    acc_adv_list.append(str(x['acc_adv_surrogate']))
    # target_acc_clean_list.append(str(x['acc_adv_surrogate']))

# torch.save(supernet_reward_list, 'supernet_reward_list')
# torch.save(supernet_acc_adv_list, 'supernet_acc_adv_list')
# torch.save(reward_list, 'reward_list')
# torch.save(acc_adv_list, 'acc_adv_list')
str1 = '\n'
f=open("supernet_reward_list.txt","w")
f.write(str1.join(supernet_reward_list))
f.close()


f=open("supernet_acc_adv_list.txt","w")
f.write(str1.join(supernet_acc_adv_list))
f.close()


f=open("reward_list.txt","w")
f.write(str1.join(reward_list))
f.close()


f=open("acc_adv_list.txt","w")
f.write(str1.join(acc_adv_list))
f.close()

f=open("supernet_acc_clean_list.txt","w")
f.write(str1.join(supernet_acc_clean_list))
f.close()