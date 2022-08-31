import torch
path_ = 'final_test_data514'
name = 'train'
import utils
a = torch.load(path_)
reshape_data = []

def tuple_2_list(mylist):
    tmp_list = []
    for x in mylist:
        tmp_list.append(list(x))
    return tmp_list

for _,x in enumerate(a):
    t_genotype = utils.arch_to_genotype(x['target_arch'][0], x['target_arch'][1], 4, "LOOSE_END_PRIMITIVES", [5], [5])
    s_genotype = utils.arch_to_genotype(x['surrogate_arch'][0], x['surrogate_arch'][1], 4, "LOOSE_END_PRIMITIVES", [5], [5])
    tmp = {}
    # tmp['target_arch'] = [tuple_2_list(x['target_arch'][0]), tuple_2_list(x['target_arch'][1])]
    tmp['target_genotype'] = "{}".format(t_genotype)
    # tmp['surrogate_arch'] = [tuple_2_list(x['surrogate_arch'][0]), tuple_2_list(x['surrogate_arch'][1])]
    tmp['surrogate_genotype'] = "{}".format(s_genotype)
    tmp["adversarial_accuracy"] = {"surrogate": x['acc_adv_surrogate'], "baseline": x['acc_adv_surrogate'] - x['absolute_reward']}
    reshape_data.append(tmp)

utils.save_yaml(reshape_data, "transbench/data/high_fidelity/test.yaml")

