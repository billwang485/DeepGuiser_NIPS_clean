
import torch
import os

acc_clean_target = []
predictor_acc_clean_surrogate = []
predictor_acc_adv_surrogate = []
acc_adv_baseline = []
# random_acc_clean_surrogate = []
# random_acc_adv_surrogate = []

def mean(a):
    return sum(a) / len(a)

for dir in os.listdir('z/0519'):
    # if step >= 15:
    #     break
    # tmp = torch.load(os.path.join(mypath, 'random_surrogate_0_result'))
    if os.path.exists(os.path.join('z/0519', dir, 'surrogate_gates_result')):
        tmp = torch.load(os.path.join('z/0519', dir, 'surrogate_gates_result'))
    else:
        assert 0
    # else:
    #     tmp = torch.load(os.path.join('z/0519', dir, 'save_dict'))
    acc_clean_target.append(tmp['target_acc_clean'])
    acc_adv_baseline.append(tmp['adv_acc_baseline'])
    # random_acc_adv_surrogate.append(tmp['adv_acc_baseline'] + tmp['reward'])
    # random_acc_clean_surrogate.append(tmp['surrogate_acc_clean'])
    predictor_acc_adv_surrogate.append(tmp['adv_acc_baseline'] + tmp['reward'])
    predictor_acc_clean_surrogate.append(tmp['surrogate_acc_clean'])

result = []
result.append(str(mean(acc_clean_target)))
result.append(str(mean(acc_adv_baseline)))
# result.append(str(mean(random_acc_adv_surrogate)))
# result.append(str(mean(random_acc_clean_surrogate)))
result.append(str(mean(predictor_acc_adv_surrogate)))
result.append(str(mean(predictor_acc_clean_surrogate)))
str1 = '\n'
f=open("supernet_random_average.txt","w")
f.write(str1.join(result))
f.close()