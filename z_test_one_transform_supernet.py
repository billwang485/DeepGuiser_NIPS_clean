import os
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
from search_model_twin import NASNetwork as Network
from single_model import FinalNetwork as FinalNet
import random
from copy import deepcopy
from scipy.stats import kendalltau
import shutil
'''
This files tests the transferbility isotonicity on supernets and trained-from-scratch models
'''

localtime = time.asctime( time.localtime(time.time()))
x = re.split(r"[\s,(:)]",localtime)
default_EXP = " ".join(x[1:-1])
parser = argparse.ArgumentParser("DeepGuiser")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.05, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=int, default=50, help='report frequency')
parser.add_argument('--test_freq', type=int, default=10, help='test frequency')
parser.add_argument('--gpu', type=int, default=6, help='gpu device id')#
parser.add_argument('--epochs', type=int, default=50, help='number of signle model training epochs')
parser.add_argument('--init_channels', type=int, default=20, help='number of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--save', type=str, default=default_EXP, help='experiment name')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--train_portion', type=float, default=0.8, help='data portion for training weights')
parser.add_argument('--prefix', type=str, default='.', help='parent save path')
parser.add_argument('--gamma', type=float, default=0.99, help='time decay for baseline update')
parser.add_argument('--scheduler', type=str, default='naive_cosine', help='type of LR scheduler')
parser.add_argument('--learning_rate_min', type=float, default=0.01, help='min learning rate')
parser.add_argument('--store', type=int, default=1, help='Whether to store the model')
parser.add_argument('--edge_hid', type=int, default=100, help='edge hidden dimension')
parser.add_argument('--num_nodes', type=int, default=4, help='number of intermediate nodes')
parser.add_argument('--op_type', type=str, default='LOOSE_END_PRIMITIVES', help='LOOSE_END_PRIMITIVES | FULLY_CONCAT_PRIMITIVES')#
parser.add_argument('--transfer', action='store_true', default=True, help='eval the transferability')
parser.add_argument('--arch', type=str, default=' ', help='The path to best archs')#
parser.add_argument('--pw', type=str, default='pretrain_models/LOOSE', help='The path to best archs')#
parser.add_argument('--controller_hid', type=int, default=100, help='controller hidden dimension')
parser.add_argument('--transformer_learning_rate', type=float, default=3e-4, help='learning rate for pruner')
parser.add_argument('--transformer_weight_decay', type=float, default=5e-4, help='learning rate for pruner')
parser.add_argument('--transformer_nfeat', type=int, default=1024, help='feature dimension of each node') 
parser.add_argument('--accu_batch', type=int, default=10, help='feature dimension of each node') 
parser.add_argument('--transformer_nhid', type=int, default=100, help='hidden dimension')
parser.add_argument('--transformer_dropout', type=float, default=0, help='dropout rate for transformer')
parser.add_argument('--transformer_normalize', action='store_true', default=False, help='use normalize in GCN')
parser.add_argument('--entropy_coeff', nargs='+', type=float, default=[0.003, 0.003], help='coefficient for entropy: [normal, reduce]')
parser.add_argument('--pwt', type=str, default='final_train/LOOSE_RES/target.pt', help='The path to pretrained weight')
parser.add_argument('--pwtb', type=str, default=' ', help='The path to pretrained weight')
parser.add_argument('--pws', type=str, default='final_train/LOOSE_RES/surrogate_model_0.pt', help='The path to pretrained weight')
args = parser.parse_args()

if args.op_type=='LOOSE_END_PRIMITIVES':
    args.loose_end = True
else:
    args.loose_end = False

if not os.path.exists(args.prefix):
    os.makedirs(args.prefix)

args.save = os.path.join(args.prefix, 'z/test_one_transform_supernet', args.save)
args.cutout = False
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py')+glob.glob('*.sh')+glob.glob('*.yml'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logger = logging.getLogger()
CIFAR_CLASSES = 10


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

    network_args_ = 'FinalNet(\
            args.init_channels, CIFAR_CLASSES, args.layers, criterion, device, steps=args.num_nodes, controller_hid=args.controller_hid, entropy_coeff=args.entropy_coeff, edge_hid = args.edge_hid, transformer_nfeat = args.transformer_nfeat, transformer_nhid = args.transformer_nhid, transformer_dropout = args.transformer_dropout, transformer_normalize = args.transformer_normalize, loose_end = args.loose_end, op_type=args.op_type\
        )'

    network_args = (args.init_channels, CIFAR_CLASSES, args.layers, criterion, device, args.num_nodes, 3, args.edge_hid, args.loose_end, None, None, args.op_type)

    _, test_queue, _ = utils.get_train_queue(args)

    target_model=FinalNet(
            args.init_channels, CIFAR_CLASSES, args.layers, criterion, device, steps=args.num_nodes, controller_hid=args.controller_hid, entropy_coeff=args.entropy_coeff, edge_hid = args.edge_hid, transformer_nfeat = args.transformer_nfeat, transformer_nhid = args.transformer_nhid, transformer_dropout = args.transformer_dropout, transformer_normalize = args.transformer_normalize, loose_end = args.loose_end, op_type=args.op_type
        )
    utils.load(target_model, args.pwt, only_arch=True)
    target_normal = target_model.arch_normal
    target_reduce = target_model.arch_reduce
    utils.load(target_model, args.pws, only_arch=True)
    surrogate_normal = target_model.arch_normal
    surrogate_reduce = target_model.arch_reduce

    model = Network( \
        args.init_channels, CIFAR_CLASSES, args.layers, criterion, device, steps=args.num_nodes, controller_hid=args.controller_hid, entropy_coeff=args.entropy_coeff, edge_hid = args.edge_hid, transformer_nfeat = args.transformer_nfeat, transformer_nhid = args.transformer_nhid, transformer_dropout = args.transformer_dropout, transformer_normalize = args.transformer_normalize, loose_end = args.loose_end, op_type=args.op_type\
    )

    model_twin = Network( \
        args.init_channels, CIFAR_CLASSES, args.layers, criterion, device, steps=args.num_nodes, controller_hid=args.controller_hid, entropy_coeff=args.entropy_coeff, edge_hid = args.edge_hid, transformer_nfeat = args.transformer_nfeat, transformer_nhid = args.transformer_nhid, transformer_dropout = args.transformer_dropout, transformer_normalize = args.transformer_normalize, loose_end = args.loose_end, op_type=args.op_type\
    )

    utils.load(model, os.path.join(args.pw, 'model.pt'))
    model.to(device)

    utils.load(model_twin, os.path.join(args.pw, 'model_twin.pt'))
    model_twin.to(device)

    reward = utils.AvgrageMeter()
    acc_clean = utils.AvgrageMeter()
    acc_adv = utils.AvgrageMeter()
    optimized_acc_adv = utils.AvgrageMeter()
    optimized_acc_clean = utils.AvgrageMeter()
    

    for step, (input, target) in enumerate(test_queue):
        n = input.size(0)
        if step >= args.accu_batch:
            break
        input = input.to(device)
        target = target.to(device)
        reward_, acc_clean_, optimized_acc_clean_, acc_adv_, optimized_acc_adv_ = model.evaluate_transfer(model_twin, (target_normal, target_reduce), (surrogate_normal, surrogate_reduce), input, target)
        reward.update(reward_, n)
        acc_clean.update(acc_clean_, n)
        acc_adv.update(acc_adv_, n)
        optimized_acc_adv.update(optimized_acc_adv_, n)
        optimized_acc_clean.update(optimized_acc_clean_, n)
    logging.info("absolute_reward=%.2f, relative_reward=%.2f, target_acc_clean=%.2f, surrogate_acc_clean=%.2f, acc_adv=%.2f, surrogate_acc_adv=%.2f",\
        reward.avg, (optimized_acc_adv.avg - acc_adv.avg) / (acc_clean.avg - acc_adv.avg), acc_clean.avg, optimized_acc_clean.avg, acc_adv.avg, optimized_acc_adv.avg)



    
    


if __name__ == '__main__':
    main()