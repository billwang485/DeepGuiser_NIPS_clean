dirs = ['z/transformer_random_test/May 18 07 41 07',
    'z/transformer_random_test/May 18 07 41 17',
    'z/transformer_random_test/May 18 07 41 52',
    'z/transformer_random_test/May 18 07 42 32',
    'z/transformer_random_test/May 18 07 44 18',
    'z/transformer_random_test/May 18 07 44 49',
    'z/transformer_random_test/May 18 07 45 50',
    'z/transformer_random_test/May 18 07 52 32',
    'z/transformer_random_test/May 18 07 53 02',
    'z/transformer_random_test/May 18 07 54 10',
    'z/transformer_random_test/May 18 07 54 58',
    'z/transformer_random_test/May 18 07 55 26',
    'z/transformer_random_test/May 18 07 55 54',
    'z/transformer_random_test/May 18 07 56 22',
    'z/transformer_random_test/May 18 07 56 43',
    'z/transformer_random_test/May 18 07 57 28',
    'z/transformer_random_test/May 18 08 14 24',
    'z/transformer_random_test/May 18 08 15 15',
    'z/transformer_random_test/May 18 08 15 34',
    'z/transformer_random_test/May 18 08 16 15',
    ]
import torch
import os

acc_clean_target = []
predictor_acc_clean_surrogate = []
predictor_acc_adv_surrogate = []
acc_adv_baseline = []
random_acc_clean_surrogate = []
random_acc_adv_surrogate = []

def mean(a):
    return sum(a) / len(a)

for step, mypath in enumerate(dirs):
    if step >= 15:
        break
    tmp = torch.load(os.path.join(mypath, 'random_surrogate_0_result'))
    acc_clean_target.append(tmp['target_acc_clean'])
    acc_adv_baseline.append(tmp['adv_acc_baseline'])
    random_acc_adv_surrogate.append(tmp['adv_acc_baseline'] + tmp['reward'])
    random_acc_clean_surrogate.append(tmp['surrogate_acc_clean'])
    tmp = torch.load(os.path.join(mypath, 'surrogate_predictor_result'))
    predictor_acc_adv_surrogate.append(tmp['adv_acc_baseline'] + tmp['reward'])
    predictor_acc_clean_surrogate.append(tmp['surrogate_acc_clean'])

result = []
result.append(str(mean(acc_clean_target)))
result.append(str(mean(acc_adv_baseline)))
result.append(str(mean(random_acc_adv_surrogate)))
result.append(str(mean(random_acc_clean_surrogate)))
result.append(str(mean(predictor_acc_adv_surrogate)))
result.append(str(mean(predictor_acc_clean_surrogate)))
str1 = '\n'
f=open("predictor_random_average.txt","w")
f.write(str1.join(result))
f.close()