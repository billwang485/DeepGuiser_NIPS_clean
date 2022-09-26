import os
import shutil
import sys
STEM_WORK_DIR = os.path.join(os.getcwd(), "..")
sys.path.append(STEM_WORK_DIR)
import glob
import random
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
import utils
import genotypes
from final_test.supernet_based.models import LooseEndModel
'''
This files tests the transferbility isotonicity on supernets and trained-from-scratch models
'''
# sys.stdout = open(os.devnull, 'w'
parser = argparse.ArgumentParser("DeepGuiser")
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')#
parser.add_argument('--seed', type=int, default=1234, help='random seed')#
parser.add_argument('--num', type=int, default=0, help='random seed')#
parser.add_argument('--genotype', type=str, default='ResBlock', help='genotype')#
args = parser.parse_args()
def uniform_random_transform(arch):
    transformed_arch = []
    for i, (op, f, t) in enumerate(arch):
        opname = genotypes.LOOSE_END_PRIMITIVES[op]
        availavble_choices = genotypes.LooseEnd_Transition_Dict[opname]
        transformed_op = random.randint(0, len(availavble_choices) - 1)
        transformed_opname = availavble_choices[transformed_op]
        transformed_arch.append((genotypes.LOOSE_END_PRIMITIVES.index(transformed_opname), f, t))

    return transformed_arch

CIFAR_SIZE = [1, 3, 32, 32]
CIFAR_CLASSES = 10
NUM = args.num
args.seed = 1234 * (args.num + 1)
def main(): 
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed(args.seed)
        cudnn.benchmark = True
        cudnn.enabled = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    result_dict = {}
    result_dict["target"] = {}
    result_dict["surrogate"] = {}

    genotype = eval("genotypes.{}".format(args.genotype))

    result_dict["target"]["genotype"] = "{}".format(genotype)
        

    arch_normal, arch_reduce = utils.genotype_to_arch(genotype)

    transformed_normal = uniform_random_transform(arch_normal)
    transformed_reduce = uniform_random_transform(arch_reduce)

    surrogate_genotpye = utils.arch_to_genotype(transformed_normal, transformed_reduce, 4, "LOOSE_END_PRIMITIVES",
                     utils.parse_loose_end_concat(transformed_normal), utils.parse_loose_end_concat(transformed_reduce))

    result_dict["surrogate"]["genotpye"] = "{}".format(surrogate_genotpye)

    target_model = LooseEndModel(device, CIFAR_CLASSES, genotype)
    target_model.to(device)
    surrogate_model = LooseEndModel(device, CIFAR_CLASSES, surrogate_genotpye)
    surrogate_model.to(device)


    target_nparam = utils.compute_nparam(target_model, CIFAR_SIZE , 'null')
    surrogate_nparam = utils.compute_nparam(surrogate_model, CIFAR_SIZE , 'null')

    target_flops = utils.compute_flops(target_model, CIFAR_SIZE , 'null')
    surrogate_flops = utils.compute_flops(surrogate_model, CIFAR_SIZE , 'null')

    target_latency = utils.compute_latency(target_model, STEM_WORK_DIR)
    surrogate_latency = utils.compute_latency(surrogate_model, STEM_WORK_DIR)

    result_dict["surrogate"]["latency"] = surrogate_latency
    result_dict["surrogate"]["flops"] = surrogate_flops
    result_dict["surrogate"]["nparam"] = surrogate_nparam

    result_dict["target"]["latency"] = target_latency
    result_dict["target"]["flops"] = target_flops
    result_dict["target"]["nparam"] = target_nparam

    utils.save_yaml(result_dict, "{}_{}.yaml".format(args.genotype, NUM))




    # train_queue, test_queue = utils.get_final_test_data(args, CIFAR_CLASSES)

    # assert os.path.exists(args.arch_info)

    # arch_info = utils.load_yaml(args.arch_info)
    # shutil.copy(args.arch_info, os.path.join(args.save, "exp_config.yaml"))

    # if arch_info["target"][0]["pretrained_weight"] != "None" and arch_info["surrogate"][0]["pretrained_weight"] != "None" and arch_info["target"][0]["pretrained_baseline_weight"] != "None":
    #     target_weight_path = os.path.join(STEM_WORK_DIR, arch_info["target"][0]["pretrained_weight"])
    #     baseline_weight_path = os.path.join(STEM_WORK_DIR, arch_info["target"][0]["pretrained_baseline_weight"])
    #     surrogate_weight_path = os.path.join(STEM_WORK_DIR, arch_info["surrogate"][0]["pretrained_weight"])
        
    #     assert os.path.exists(target_weight_path)
    #     assert os.path.exists(baseline_weight_path)
    #     assert os.path.exists(surrogate_weight_path)
    #     bypass_training = True
    #     load_pretrained_weight = True
    # else:
    #     bypass_training = False
    #     load_pretrained_weight = False

    # target_genotype = eval(arch_info['target'][0]['genotype'])
    # surrogate_genotpye = eval(arch_info['surrogate'][0]['genotype'])

    # target_model = LooseEndModel(device, CIFAR_CLASSES, target_genotype)

    # # assert 0

    # if load_pretrained_weight:
    #     utils.load_supernet_based(target_model, target_weight_path)
    # target_model.to(device)

    # if bypass_training:
    #     logging.info('Bypassing Target Model Training')
    # else:

    #     logging.info('training Target Model')

    #     target_optimizer = torch.optim.SGD(
    #         target_model.model_parameters(),
    #         args.learning_rate,
    #         momentum=args.momentum,
    #         weight_decay=args.weight_decay
    #     )

    #     target_model.to(device)

    #     if args.scheduler == "naive_cosine":
    #         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #         target_optimizer, float(args.epochs), eta_min=args.learning_rate_min
    #         )
    #     else:
    #         assert False, "unsupported scheduler type: %s" % args.scheduler

    #     utils.train_model(target_model, train_queue, device, criterion, target_optimizer, scheduler, args.epochs, logger)

    # utils.save_supernet_based(target_model, os.path.join(args.save, 'target_model.pt'))
    # acc_clean_target, _ = utils.test_clean_accuracy(target_model, test_queue, logger)

    # logging.info('training Surrogate Model')

    # surrogate_model = LooseEndModel(device, CIFAR_CLASSES, surrogate_genotpye)

    # if load_pretrained_weight:
    #     utils.load_supernet_based(surrogate_model, surrogate_weight_path)
    # surrogate_model.to(device)

    # if bypass_training:
    #     logging.info("Bypassing Surrogate Model Training")
    # else:
    #     logging.info('training Surrogate Model')

    
    #     surrogate_optimizer = torch.optim.SGD(
    #         surrogate_model.model_parameters(),
    #         args.learning_rate,
    #         momentum=args.momentum,
    #         weight_decay=args.weight_decay
    #     )

    #     surrogate_model.to(device)

    #     if args.scheduler == "naive_cosine":
    #         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #         surrogate_optimizer, float(args.epochs), eta_min=args.learning_rate_min
    #         )
    #     else:
    #         assert False, "unsupported scheduler type: %s" % args.scheduler

    #     utils.train_model(surrogate_model, train_queue, device, criterion, surrogate_optimizer, scheduler, args.epochs, logger)

    # utils.save_supernet_based(surrogate_model, os.path.join(args.save, 'surrogate_model.pt'))
    # acc_clean_surrogate, _ = utils.test_clean_accuracy(surrogate_model, test_queue, logger)


    # logging.info('training Target Baseline Model')

    # baseline_model = LooseEndModel(device, CIFAR_CLASSES, target_genotype)

    
   

    # if load_pretrained_weight:
    #     utils.load_compiled_based(baseline_model, baseline_weight_path)

    # baseline_model.to(device)

    # if bypass_training:
    #     logging.info("Bypassing Surrogate Model Training")
    # else:
    #     logging.info('training Target Baseline Model')

    #     target_baseline_optimizer = torch.optim.SGD(
    #         baseline_model.model_parameters(),
    #         args.learning_rate,
    #         momentum=args.momentum,
    #         weight_decay=args.weight_decay
    #     )

    #     if args.scheduler == "naive_cosine":
    #         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #         target_baseline_optimizer, float(args.epochs), eta_min=args.learning_rate_min
    #         )
    #     else:
    #         assert False, "unsupported scheduler type: %s" % args.scheduler

    #     utils.train_model(baseline_model, train_queue, device, criterion, target_baseline_optimizer, scheduler, args.epochs, logger)

    # utils.save_supernet_based(baseline_model, os.path.join(args.save, 'baseline_model.pt'))
    # acc_clean_baseline, _ = utils.test_clean_accuracy(baseline_model, test_queue, logger)

    # logging.info('training completed')

    # logging.info('testing adversarial attack transferbitlity')

    # attack_info = utils.load_yaml(args.attack_info)

    # # acc_adv_baseline, acc_adv_surrogate = utils.compiled_pgd_test(target_model, surrogate_model, baseline_model, test_queue, attack_info, logger)
    # acc_adv_baseline, acc_adv_surrogate = attacks.adversarial_test(target_model, surrogate_model, baseline_model, test_queue, attack_info, logger)

    # save_dict = {}
    # save_dict['target_genotype'] = '{}'.format(target_genotype)
    # save_dict['surrogate_arch'] = '{}'.format(surrogate_genotpye)
    # save_dict['acc_clean'] = {'target': acc_clean_target, 'surrogate': acc_clean_surrogate, 'baseline': acc_clean_baseline}
    # save_dict['acc_adv'] = {'surrogate': acc_adv_surrogate, 'baseline': acc_adv_baseline}
    # utils.save_yaml(save_dict, os.path.join(args.save, 'result.yaml'))

    # utils.draw_clean(target_genotype, args.save, 'target')
    # utils.draw_clean(surrogate_genotpye, args.save, 'surrogate')


if __name__ == '__main__':
    main()
