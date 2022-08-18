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
from distillation import Linf_PGD
'''
This files tests the transferbility isotonicity on supernets finetune models and trained-from-scratch models
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
parser.add_argument('--gpu', type=int, default=1, help='gpu device id')
parser.add_argument('--num', type=int, default=5000, help=' ')
parser.add_argument('--init_channels', type=int, default=20, help='number of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--save', type=str, default=default_EXP, help='experiment name')
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
parser.add_argument('--surrogate_num', type=int, default=50, help='number of surrogate model to test')
parser.add_argument('--target_num', type=int, default=5, help='number of target model to test')
parser.add_argument('--pw', type=str, default='', help='The path to pretrained weight if there is(dir)')
args = parser.parse_args()

eps = 0.031

assert args.pw != ' '

if args.op_type=='LOOSE_END_PRIMITIVES':
    args.loose_end = True
else:
    args.loose_end = False

if not os.path.exists(args.prefix):
    os.makedirs(args.prefix)

args.save = os.path.join(args.prefix, 'Trans_Iso_surrogate_finetune', args.save)
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

    model = Network(
        args.init_channels, CIFAR_CLASSES, args.layers, criterion, device, steps=args.num_nodes, controller_hid=args.controller_hid, entropy_coeff=args.entropy_coeff, edge_hid = args.edge_hid, transformer_nfeat = args.transformer_nfeat, transformer_nhid = args.transformer_nhid, transformer_dropout = args.transformer_dropout, transformer_normalize = args.transformer_normalize, loose_end = args.loose_end, op_type=args.op_type
    )

    model_twin = Network(
        args.init_channels, CIFAR_CLASSES, args.layers, criterion, device, steps=args.num_nodes, controller_hid=args.controller_hid, entropy_coeff=args.entropy_coeff, edge_hid = args.edge_hid, transformer_nfeat = args.transformer_nfeat, transformer_nhid = args.transformer_nhid, transformer_dropout = args.transformer_dropout, transformer_normalize = args.transformer_normalize, loose_end = args.loose_end, op_type=args.op_type
    )

    model_optimizer = torch.optim.SGD(
        model.model_parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    model_twin_optimizer = torch.optim.SGD(
        model_twin.model_parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    test_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    random.shuffle(indices)
    split = int(np.floor(num_train))
    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        shuffle = True,
        pin_memory=True, num_workers=2
    )

    test_queue = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size,
        shuffle = True,
        pin_memory=True, num_workers=2
    )

    model.to(device)
    model_twin.to(device)

    model._model_optimizer = model_optimizer

    model_twin._model_optimizer = model_twin_optimizer

    logging.info('Copying Pretrained Models')

    pretrained_prefix = os.path.join(args.pw)
    shutil.copy(os.path.join(pretrained_prefix, 'supernet.pt'), os.path.join(args.save, 'supernet.pt'))
    shutil.copy(os.path.join(pretrained_prefix, 'supernet_twin.pt'), os.path.join(args.save, 'supernet_twin.pt'))
    shutil.copy(os.path.join(pretrained_prefix, 'surrogate_archs'), os.path.join(args.save, 'surrogate_archs'))
    shutil.copy(os.path.join(pretrained_prefix, 'target_archs'), os.path.join(args.save, 'target_archs'))
    surrogate_archs = torch.load(os.path.join(args.save, 'surrogate_archs'))
    target_archs = torch.load(os.path.join(args.save, 'target_archs'))
    logging.info('save to %s', args.save)
    for i in range(args.surrogate_num):
        shutil.copy(os.path.join(pretrained_prefix, 'surrogate_{}.pt'.format(i)), os.path.join(args.save, 'surrogate_{}.pt'.format(i)))
    for i in range(args.target_num):
        shutil.copy(os.path.join(pretrained_prefix, 'target_{}.pt'.format(i)), os.path.join(args.save, 'target_{}.pt'.format(i)))

    logging.info('Testing Transferbility Isotonicity')

    Testing_Transfer_Isotonicity(args, test_queue, criterion, device, surrogate_archs, target_archs)

def Testing_Transfer_Isotonicity(args, valid_queue, criterion, device, surrogate_archs, target_archs):

    attack_acc_clean_supernet = []
    attack_acc_adv_supernet = []
    attack_acc_clean_single = []
    attack_acc_adv_single = []
    for i in range(args.surrogate_num):
        attack_acc_clean_supernet.append(utils.AvgrageMeter())
        attack_acc_adv_supernet.append(utils.AvgrageMeter())
        attack_acc_clean_single.append(utils.AvgrageMeter())
        attack_acc_adv_single.append(utils.AvgrageMeter())

    logging.info('Loading Pretrained Supernets')
    model = Network(
        args.init_channels, CIFAR_CLASSES, args.layers, criterion, device, steps=args.num_nodes, controller_hid=args.controller_hid, entropy_coeff=args.entropy_coeff, edge_hid = args.edge_hid, transformer_nfeat = args.transformer_nfeat, transformer_nhid = args.transformer_nhid, transformer_dropout = args.transformer_dropout, transformer_normalize = args.transformer_normalize, loose_end = args.loose_end, op_type=args.op_type
    )
    model_twin = Network(
        args.init_channels, CIFAR_CLASSES, args.layers, criterion, device, steps=args.num_nodes, controller_hid=args.controller_hid, entropy_coeff=args.entropy_coeff, edge_hid = args.edge_hid, transformer_nfeat = args.transformer_nfeat, transformer_nhid = args.transformer_nhid, transformer_dropout = args.transformer_dropout, transformer_normalize = args.transformer_normalize, loose_end = args.loose_end, op_type=args.op_type
        )
    utils.load(model, os.path.join(args.save, 'supernet.pt'))
    utils.load(model_twin, os.path.join(args.save, 'supernet_twin.pt'))
    model.to(device)
    model_twin.to(device)
    for i in range(args.target_num):
        logging.info('save to %s', args.save)
        save_ = os.path.join(args.save, str(i))
        os.mkdir(save_)
        logging.info('Testing Target Model {}'.format(i))
        logging.info('Testing on Supernet')
        for step, (input, target) in enumerate(valid_queue):
            n = input.size(0)
            input = input.to(device)
            target = target.to(device)
            for j, (arch_normal, arch_reduce) in enumerate(surrogate_archs):
                input_adv = Linf_PGD(model_twin, arch_normal, arch_reduce, input, target, eps, alpha=eps/10, steps=10)
                (acc_clean, acc_adv) = model.eval_transfer(input_adv, input, target, target_archs[i][0], target_archs[i][1])
                attack_acc_clean_supernet[j].update(acc_clean.item(), n)
                attack_acc_adv_supernet[j].update(acc_adv.item(), n)
        logging.info('Complete Testing Target Model %d on Supernet', i)
        logging.info('Surrogate Acc When Generating Adversial Inputs')
        logging.info('Surrogate Acc Attacking Target Model')
        tmp_1 = []
        tmp_2 = []
        logging.info('save to %s', args.save)
        for j in range(len(surrogate_archs)):
            logging.info('Surrogate Model %d on Supernet: attack_acc_clean=%.2f, attack_acc_adv=%.2f', j, attack_acc_clean_supernet[j].avg, attack_acc_adv_supernet[j].avg)
            tmp_1.append(attack_acc_clean_supernet[j].avg)
            tmp_2.append(attack_acc_adv_supernet[j].avg)
        logging.info('attack_acc_clean_supernet:')
        logging.info(tmp_1)
        logging.info('attack_acc_adv_supernet:')
        logging.info(tmp_2)
        logging.info('Testing on Supernet Completed')
        
        
        logging.info('Loading Target Model')
        logging.info('save to %s', args.save)
        assert os.path.exists(os.path.join(args.save, 'target_{}.pt'.format(i)))
        target_model = FinalNet(
        args.init_channels, CIFAR_CLASSES, args.layers, criterion, device, steps=args.num_nodes, controller_hid=args.controller_hid, entropy_coeff=args.entropy_coeff, edge_hid = args.edge_hid, transformer_nfeat = args.transformer_nfeat, transformer_nhid = args.transformer_nhid, transformer_dropout = args.transformer_dropout, transformer_normalize = args.transformer_normalize, loose_end = args.loose_end, op_type=args.op_type
    )
        utils.load(target_model, os.path.join(args.save, 'target_{}.pt'.format(i)))
        target_model.to(device)
        for j, (arch_normal, arch_reduce) in enumerate(surrogate_archs):
            logging.info('save to %s', args.save)
            surrogate_model = FinalNet(
        args.init_channels, CIFAR_CLASSES, args.layers, criterion, device, steps=args.num_nodes, controller_hid=args.controller_hid, entropy_coeff=args.entropy_coeff, edge_hid = args.edge_hid, transformer_nfeat = args.transformer_nfeat, transformer_nhid = args.transformer_nhid, transformer_dropout = args.transformer_dropout, transformer_normalize = args.transformer_normalize, loose_end = args.loose_end, op_type=args.op_type
    )
            utils.load(surrogate_model, os.path.join(args.save, 'surrogate_{}.pt'.format(j)))
            surrogate_model.to(device)
            surrogate_model.arch_normal = arch_normal
            surrogate_model.arch_reduce = arch_reduce
            for step, (input, target) in enumerate(valid_queue):
                n = input.size(0)
                input = input.to(device)
                target = target.to(device)
                # logging.info('Testing Acc of Surrogate Model')
                # acc_clean = surrogate_model._test_acc(valid_queue, surrogate_model.arch_normal, surrogate_model.arch_reduce)
                input_adv = Linf_PGD(surrogate_model, surrogate_model.arch_normal, surrogate_model.arch_reduce, input, target, eps, alpha=eps/10, steps=10)
                # surrogate_acc_clean_single[j].update(acc_clean.item(), n)
                # surrogate_acc_adv_single[j].update(acc_adv.item(), n)
                (acc_clean, acc_adv) = target_model.eval_transfer(input_adv, input, target)
                attack_acc_clean_single[j].update(acc_clean.item(), n)
                attack_acc_adv_single[j].update(acc_adv.item(), n)
        tmp_1 = []
        tmp_2 = []
        for j in range(args.surrogate_num):
            logging.info('Surrogate Model %d Single: target_acc_clean=%.2f, target_acc_adv=%.2f', i, attack_acc_clean_single[j].avg, attack_acc_adv_single[j].avg)
            tmp_1.append(attack_acc_clean_single[j].avg)
            tmp_2.append(attack_acc_adv_single[j].avg)
        logging.info('save to %s', args.save)
        logging.info('attack_acc_clean_single:')
        logging.info(tmp_1)
        logging.info('attack_acc_adv_single:')
        logging.info(tmp_2)
        logging.info('Testing Single Completed')

        logging.info('Computing Kendall')
        supernet_adv = []
        single_adv = []
        for j in range(args.surrogate_num):
            supernet_adv.append(attack_acc_adv_supernet[j].avg)
            single_adv.append(attack_acc_adv_single[j].avg)
            logging.info('Adversial Acc %d: Supernet: %.2f Single: %.2f', j, supernet_adv[-1], single_adv[-1])
        kendalltau_, _ = kendalltau(supernet_adv, single_adv)
        logging.info('Kendalltau = {}'.format(kendalltau_))
        

def Training_Single_Models(name, num, args, device, train_queue, valid_arch_queue, *network):
    logging.info('Sampling and Training %s Models', name)
    archs = []
    for i in range(num):
        model_=FinalNet(*network, name)

        # Checking identical models
        tmp = (deepcopy(model_.arch_normal), deepcopy(model_.arch_reduce))
        while tmp in archs:
            model_ = FinalNet(*network, name)
            tmp = (deepcopy(model_.arch_normal), deepcopy(model_.arch_reduce))
        archs.append(tmp)
        
        optimizer = torch.optim.SGD(
            model_.model_parameters(),
            args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
        model_.to(device)
        model_._model_optimizer = optimizer
        if args.scheduler == "naive_cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(args.epochs_single), eta_min=args.learning_rate_min
            )
        else:
            assert False, "unsupported scheduler type: %s" % args.scheduler
        for epoch in range(args.epochs_single+1):
            scheduler.step()
            lr = scheduler.get_lr()[0]
            logging.info('epoch %d lr %e', epoch, lr)

            logging.info('Updating %s Model %d Parameters', name, i)
            update_w(train_queue, model_, device)
            utils.save(model_, os.path.join(args.save, '{}_model_{}.pt'.format(name, i)))
        if args.store == 1:
            utils.save(model_, os.path.join(args.save, '{}_model_{}.pt'.format(name, i)))
        else:
            assert 0,'Must Save Model'
        
    return archs

def update_w(valid_queue, model, device):
    objs = utils.AvgrageMeter()
    normal_ent = utils.AvgrageMeter()
    reduce_ent = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    for step, (input, target) in enumerate(valid_queue):
        model.train()
        n = input.size(0)
        input = input.to(device)
        target = target.to(device)
        logits, loss = model.step(input, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('Updating W: Step=%03d Loss=%e Top1=%f Top5=%f Normal_ENT=%f, Reduce_ENT=%f',
                         step, objs.avg, top1.avg, top5.avg, normal_ent.avg, reduce_ent.avg)

if __name__ == '__main__':
    main()
