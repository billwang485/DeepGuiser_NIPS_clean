dir = ['./predictor/finetune_dataset/Apr 24 05 51 48',
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
    './predictor/finetune_dataset/Apr 24 06 06 58',
    './predictor/finetune_dataset/Apr 24 06 09 24'
    ]
import torch
import os
target_dir = []
surrogate_dir = []
for i, tdir in enumerate(dir):
    for j in range(10000):
        target_dict = {}
        if not os.path.exists(os.path.join(tdir, str(j))):
            break
        if not (os.path.exists(os.path.join(tdir, str(j), 'target.pt')) and os.path.exists(os.path.join(tdir, str(j), 'target_baseline.pt')) and os.path.exists(os.path.join(tdir, str(j), 'surrogate_model.pt'))):
            continue
        target_dict['target_dir'] = os.path.join(tdir, str(j))
        target_dict['surrogate_path'] = [os.path.join(tdir, str(j), 'surrogate_model.pt')]
        surrogate_dir.append(os.path.join(tdir, str(j), 'surrogate_model.pt'))
        for k in range(1, 10000):
            if not os.path.exists(os.path.join(tdir, str(j), 'surrogate_model_{}.pt'.format(k))):
                break
            target_dict['surrogate_path'].append(os.path.join(tdir, str(j), 'surrogate_model_{}.pt'.format(k)))
            surrogate_dir.append(os.path.join(tdir, str(j), 'surrogate_model_{}.pt'.format(k)))
        target_dir.append(target_dict)
torch.save(target_dir, os.path.join('no_name', 'rr_target_dir'))
torch.save(surrogate_dir, os.path.join('no_name', 'rr_surrogate_path'))
