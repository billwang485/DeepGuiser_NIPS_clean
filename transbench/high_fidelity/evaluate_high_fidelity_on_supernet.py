import os
import re
import sys
STEM_WORK_DIR = os.path.join(os.getcwd(), "../..")
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
from torch.utils.tensorboard import SummaryWriter
import utils
from genotypes import Genotype
from basic_parts.basic_integrated_model import NASNetwork as Network


'''
This file trains supernet and twin supernet for 50 epochs and save them
You can bypass the first 50 epochs of training by loading pretrained models
'''

parser = argparse.ArgumentParser("DeepGuiser")
parser.add_argument('--data', type=str, default=os.path.join(STEM_WORK_DIR, '../data'), help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.05, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=int, default=50, help='report frequency')
parser.add_argument('--test_freq', type=int, default=10, help='test frequency')
parser.add_argument('--test_archs', type=int, default=100, help='how many archs to test supernet')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=1, help='number of training epochs')
parser.add_argument('--init_channels', type=int, default=20, help='number of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--save', type=str, default=utils.localtime_as_dirname(), help='experiment name')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--n_archs', type=int, default=1, help='number of candidate archs')
parser.add_argument('--prefix', type=str, default='.', help='parent save path')
parser.add_argument('--controller_hid', type=int, default=100, help='controller hidden dimension')
parser.add_argument('--entropy_coeff', nargs='+', type=float, default=[0.003, 0.003], help='coefficient for entropy: [normal, reduce]')
parser.add_argument('--gamma', type=float, default=0.99, help='time decay for baseline update')
parser.add_argument('--controller_start_training', type=int, default=0, help='Epoch that the training of controller starts')
parser.add_argument('--scheduler', type=str, default='naive_cosine', help='type of LR scheduler')
parser.add_argument('--learning_rate_min', type=float, default=0.01, help='min learning rate')
parser.add_argument('--store', type=int, default=1, help='Whether to store the model')
parser.add_argument('--edge_hid', type=int, default=100, help='edge hidden dimension')
parser.add_argument('--transformer_learning_rate', type=float, default=3e-4, help='learning rate for pruner')
parser.add_argument('--transformer_weight_decay', type=float, default=5e-4, help='learning rate for pruner')
parser.add_argument('--transformer_nfeat', type=int, default=1024, help='feature dimension of each node')
parser.add_argument('--transformer_nhid', type=int, default=100, help='hidden dimension')
parser.add_argument('--transformer_dropout', type=float, default=0, help='dropout rate for transformer')
parser.add_argument('--transformer_normalize', action='store_true', default=False, help='use normalize in GCN')
parser.add_argument('--num_nodes', type=int, default=4, help='number of intermediate nodes')
parser.add_argument('--op_type', type=str, default='LOOSE_END_PRIMITIVES', help='LOOSE_END_PRIMITIVES | FULLY_CONCAT_PRIMITIVES')
parser.add_argument('--transfer', action='store_true', default=True, help='eval the transferability')
parser.add_argument('--debug', action='store_true', default=False, help='debud mode')
parser.add_argument('--evaluate_dataset', "-ed", type=str, default=' ', help='parent save path')
args = parser.parse_args()

if args.op_type=='LOOSE_END_PRIMITIVES':
    args.loose_end = True
else:
    args.loose_end = False
args.cutout = False

NAME = args.evaluate_dataset.split(".")[-2]
args.save = NAME + "_" + args.save 
utils.preprocess_exp_dir(args)
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py')+glob.glob('*.sh')+glob.glob('*.yml'))

utils.initialize_logger(args)

CIFAR_CLASSES = 10
STEPS = 10
EPS = 0.031
summaryWriter = SummaryWriter(os.path.join(args.save, "runs"))

def main():
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed(args.seed)
        cudnn.benchmark = True
        cudnn.enabled = True
        logging.info('GPU device = %d' % args.gpu)
    else:
        logging.info('no GPU available, use CPU!!')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    logging.info("args = %s" % args)

    _, valid_queue, _ = utils.get_cifar_data_queue(args)
    
    model = Network(
        args.init_channels, CIFAR_CLASSES, args.layers, None, device, steps=args.num_nodes, controller_hid=args.controller_hid, entropy_coeff=args.entropy_coeff, edge_hid = args.edge_hid, transformer_nfeat = args.transformer_nfeat, transformer_nhid = args.transformer_nhid, transformer_dropout = args.transformer_dropout, transformer_normalize = args.transformer_normalize, loose_end = args.loose_end, op_type=args.op_type
    )

    model_twin = model.get_twin_model()

    utils.load_supernet(model, os.path.join(STEM_WORK_DIR, 'supernet/selected_supernets/supernet.pt'))
    utils.load_supernet(model_twin, os.path.join(STEM_WORK_DIR, 'supernet/selected_supernets/supernet_twin.pt'))

    model.to(device)
    model_twin.to(device)

    old_dataset = utils.load_yaml(args.evaluate_dataset)

    for num in range(len(old_dataset)):

        logging.info("Evalute Architecture NO.{}".format(num))

        optimized_normal, optimized_reduce = utils.genotype_to_arch(eval(old_dataset[num]["surrogate_genotype"]))
        arch_normal, arch_reduce = utils.genotype_to_arch(eval(old_dataset[num]["target_genotype"]))
        acc_adv_baseline = utils.AvgrageMeter()
        acc_adv_surrogate = utils.AvgrageMeter()
        for step, (input, target) in enumerate(valid_queue):
            n = input.size(0)
            input = input.to(device)
            target = target.to(device)
            if step >= 10:
                break
            input_adv = utils.linf_pgd(model_twin, optimized_normal, optimized_reduce, input, target, eps=EPS, alpha=EPS / STEPS, steps=STEPS, rand_start=False)
            input_adv_ = utils.linf_pgd(model_twin, arch_normal, arch_reduce, input, target, eps=EPS, alpha=EPS / STEPS, steps=STEPS, rand_start=False)

            logits = model._inner_forward(input_adv, arch_normal, arch_reduce)
            acc_adv_surrogate_ = utils.accuracy(logits, target, topk=(1, 5))[0] / 100.0
            # print(utils.accuracy(logits, target, topk=(1, 5))[0])
            acc_adv_surrogate.update(acc_adv_surrogate_.item(), n)

            logits = model._inner_forward(input_adv_, arch_normal, arch_reduce)
            acc_adv_baseline_ = utils.accuracy(logits, target, topk=(1, 5))[0] / 100.0
            acc_adv_baseline.update(acc_adv_baseline_.item(), n)

        old_dataset[num]["supernet"] = {}
        old_dataset[num]["supernet"]["adversarial_accuracy"] = {}
        old_dataset[num]["supernet"]["adversarial_accuracy"]["baseline"] = acc_adv_baseline.avg
        old_dataset[num]["supernet"]["adversarial_accuracy"]["surrogate"] = acc_adv_surrogate.avg
        logging.info(old_dataset[num]["supernet"]["adversarial_accuracy"])
    
    utils.save_yaml(old_dataset, "{}_new.yaml".format(NAME))

    # save model
   

if __name__ == '__main__':
    main()






