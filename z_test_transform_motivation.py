import os
from shutil import copy
from matplotlib import collections
import torch
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
    './predictor/finetune_dataset/Apr 24 05 58 22']


import os
from pickle import TRUE
import sys
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import time
import re
from single_model import NASNetwork as Network
from nat_learner_twin import Transformer
import random
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
localtime = time.asctime( time.localtime(time.time()))
x = re.split(r"[\s,(:)]",localtime)
default_EXP = " ".join(x[1:-1])
parser = argparse.ArgumentParser("NAT")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--save_freq', type=int, default=100, help='save frequency')
parser.add_argument('--lr', type=float, default=0.05, help='init learning rate')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--num', type=int, default=10000, help='number of training iteration')
parser.add_argument('--init_channels', type=int, default=20, help='number of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--save', type=str, default=default_EXP, help='experiment name')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--prefix', type=str, default='.', help='parent save path')
parser.add_argument('--controller_hid', type=int, default=100, help='controller hidden dimension')
parser.add_argument('--entropy_coeff', nargs='+', type=float, default=[0.000, 0.000], help='coefficient for entropy: [normal, reduce]')
parser.add_argument('--gamma', type=float, default=0.99, help='time decay for baseline update')
parser.add_argument('--controller_start_training', type=int, default=0, help='Epoch that the training of controller starts')
parser.add_argument('--scheduler', type=str, default='naive_cosine', help='type of LR scheduler')
parser.add_argument('--lr_min', type=float, default=0.01, help='min learning rate')
parser.add_argument('--store', type=int, default=1, help='Whether to store the model')
parser.add_argument('--edge_hid', type=int, default=100, help='edge hidden dimension')
parser.add_argument('--transformer_weight_decay', type=float, default=5e-4, help='learning rate for pruner')
parser.add_argument('--transformer_nfeat', type=int, default=1024, help='feature dimension of each node')
parser.add_argument('--transformer_nhid', type=int, default=100, help='hidden dimension')
parser.add_argument('--transformer_dropout', type=float, default=0, help='dropout rate for transformer')
parser.add_argument('--transformer_normalize', action='store_true', default=False, help='use normalize in GCN')
parser.add_argument('--num_nodes', type=int, default=4, help='number of intermediate nodes')
parser.add_argument('--op_type', type=str, default='LOOSE_END_PRIMITIVES', help='LOOSE_END_PRIMITIVES | FULLY_CONCAT_PRIMITIVES')
parser.add_argument('--pw', type=str, default='LOOSE_END_supernet', help='The path to pretrained weight if there is(dir)')
parser.add_argument('--debug', action='store_true', default=False, help=' ')
parser.add_argument('--constrain', action='store_true', default=False, help=' ')
parser.add_argument('--accu_batch', type=int, default=10, help=' ')
parser.add_argument('--rt', type=str, default='a', help='reward_type')       
args = parser.parse_args()

utils.parse_args_(args)
assert args.pw != ' '

if args.op_type=='LOOSE_END_PRIMITIVES':
    args.loose_end = True
else:
    args.loose_end = False

if not os.path.exists(args.prefix):
    os.makedirs(args.prefix)
tmp = 'z/test_transform_motivation'
if args.debug:
    tmp = os.path.join(tmp, 'debug')

args.save = os.path.join(args.prefix, tmp, args.save)
args.cutout = False
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py')+glob.glob('*.sh')+glob.glob('*.yml'))


log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logger = logging.getLogger()
CIFAR_CLASSES = 10

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

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    target_model = Network( \
        args.init_channels, CIFAR_CLASSES, args.layers, criterion, device, steps=args.num_nodes, controller_hid=args.controller_hid, entropy_coeff=args.entropy_coeff, edge_hid = args.edge_hid, transformer_nfeat = args.transformer_nfeat, transformer_nhid = args.transformer_nhid, transformer_dropout = args.transformer_dropout, transformer_normalize = args.transformer_normalize, loose_end = args.loose_end, op_type=args.op_type\
    )

    target_model_baseline = Network( \
        args.init_channels, CIFAR_CLASSES, args.layers, criterion, device, steps=args.num_nodes, controller_hid=args.controller_hid, entropy_coeff=args.entropy_coeff, edge_hid = args.edge_hid, transformer_nfeat = args.transformer_nfeat, transformer_nhid = args.transformer_nhid, transformer_dropout = args.transformer_dropout, transformer_normalize = args.transformer_normalize, loose_end = args.loose_end, op_type=args.op_type\
    )

    surrogate_model = Network( \
        args.init_channels, CIFAR_CLASSES, args.layers, criterion, device, steps=args.num_nodes, controller_hid=args.controller_hid, entropy_coeff=args.entropy_coeff, edge_hid = args.edge_hid, transformer_nfeat = args.transformer_nfeat, transformer_nhid = args.transformer_nhid, transformer_dropout = args.transformer_dropout, transformer_normalize = args.transformer_normalize, loose_end = args.loose_end, op_type=args.op_type\
    )

    train_queue, valid_queue, test_queue = utils.get_train_queue(args)

    target_dirs = []
    surrogate_dirs = []

    for i, path_ in enumerate(dir):
        for j in range(100):
            if os.path.exists(os.path.join(path_, str(j), 'target.pt')) and os.path.exists(os.path.join(path_, str(j), 'target_baseline.pt')):
                target_dirs.append(os.path.join(path_, str(j)))
            if os.path.exists(os.path.join(path_, str(j), 'surrogate_model.pt')):
                surrogate_dirs.append(os.path.join(path_, str(j)))
    
    all_data = []
    
    for i in range(20):
        data = []
        utils.load(target_model, os.path.join(target_dirs[i], 'target.pt'))
        target_model.to(device)
        target_model.single = True
        utils.load(target_model_baseline, os.path.join(target_dirs[i], 'target_baseline.pt'))
        target_model_baseline.to(device)
        target_model_baseline.single = True
        indices = list(range(len(surrogate_dirs)))
        random.shuffle(indices)
        for j in range(20):
            reward = utils.AvgrageMeter()
            acc_clean = utils.AvgrageMeter()
            acc_adv = utils.AvgrageMeter()
            optimized_acc_adv = utils.AvgrageMeter()
            optimized_acc_clean = utils.AvgrageMeter()
            baseline_acc_clean = utils.AvgrageMeter()
            utils.load(surrogate_model, os.path.join(surrogate_dirs[indices[j]], 'surrogate_model.pt'))
            surrogate_model.to(device)
            surrogate_model.single = True
            top1 = surrogate_model._test_acc(valid_queue, surrogate_model.arch_normal, surrogate_model.arch_reduce)
            logging.info('top1 = %.3f', top1)
            for step, (input, target) in enumerate(valid_queue):
                if step > args.accu_batch:
                    break
                n = target.size(0)
                input = input.to(device)
                target = target.to(device)
                _ , acc_clean_, _, _ , optimized_acc_adv_ = target_model.evaluate_transfer(surrogate_model, (target_model.arch_normal, target_model.arch_reduce), (surrogate_model.arch_normal, surrogate_model.arch_reduce), input, target)
                _ , _ , _ , _ , acc_adv_baseline_ = target_model.evaluate_transfer(target_model_baseline, (target_model.arch_normal, target_model.arch_reduce), (target_model_baseline.arch_normal, target_model_baseline.arch_reduce), input, target)
                reward_ = optimized_acc_adv_ - acc_adv_baseline_
                reward.update(reward_, n)
                acc_clean.update(acc_clean_, n)
                acc_adv.update(acc_adv_baseline_, n)
                optimized_acc_adv.update(optimized_acc_adv_, n)
            data_point = OrderedDict()
            
            data_point["target_arch"] = (target_model.arch_normal, target_model.arch_reduce)
            data_point["surrogate_arch"] = (surrogate_model.arch_normal, surrogate_model.arch_reduce)
            # data_point["target_acc_clean"] = target_acc_clean
            # data_point["surrogate_acc_clean"] = surrogate_acc_clean
            data_point["surrogate_indice"] = indices[j]
            data_point["target_indice"] = i
            # data_point["train_info"] = {"index": its, "info": args.save}

            # data_point["target_arch"] = (target_normal, target_reduce)
            # data_point["surrogate_arch"] = (surrogate_normal, surrogate_reduce)
            data_point["absolute_reward"] = reward.avg
            data_point["relative_reward"] = (optimized_acc_adv.avg - acc_adv.avg) / (acc_clean.avg - acc_adv.avg)
            # data_point["target_acc_clean"] = acc_clean.avg
            # data_point["surrogate_acc_clean"] = optimized_acc_clean.avg
            data_point["acc_adv"] = acc_adv.avg
            data_point["surrogate_acc_adv"] = optimized_acc_adv.avg
            data_point["constrain"] = True
            # data_point["train_info"] = {"index": iteration, "info": args.save}

            data.append(data_point)
            logging.info("target: %d surrogate: %d absolute_reward=%.2f, relative_reward=%.2f, acc_adv_baseline=%.2f, surrogate_acc_adv=%.2f", i, indices[j],\
                reward.avg, (optimized_acc_adv.avg - acc_adv.avg) / (acc_clean.avg - acc_adv.avg), acc_adv.avg, optimized_acc_adv.avg)
        all_data.append(data)
        torch.save(all_data, os.path.join(args.save, 'result'))
    torch.save(all_data, os.path.join(args.save, 'result'))
            



if __name__ == '__main__':
    main()