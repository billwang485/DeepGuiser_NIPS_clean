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
    './predictor/finetune_dataset/Apr 24 06 06 58',]
    # './predictor/finetune_dataset/Apr 24 06 09 24']

from operator import truediv
import os
import shutil
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
import random
from single_model import FinalNetwork as FinalNet
from collections import OrderedDict
'''
This file trains supernet and twin supernet for 50 epochs and save them
You can bypass the first 50 epochs of training by loading pretrained models
'''
localtime = time.asctime( time.localtime(time.time()))
x = re.split(r"[\s,(:)]",localtime)
default_EXP = " ".join(x[1:-1])
parser = argparse.ArgumentParser("NAT")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--lr', type=float, default=0.05, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=int, default=50, help='report frequency')
parser.add_argument('--test_freq', type=int, default=10, help='test frequency')
parser.add_argument('--save_freq', type=int, default=10, help='test frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='number of training epochs')
parser.add_argument('--init_channels', type=int, default=20, help='number of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--save', type=str, default=default_EXP, help='experiment name')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--n_archs', type=int, default=1, help='number of candidate archs')
# parser.add_argument('--num', type=int, default=10000, help=' ')
parser.add_argument('--prefix', type=str, default='.', help='parent save path')
parser.add_argument('--controller_hid', type=int, default=100, help='controller hidden dimension')
parser.add_argument('--entropy_coeff', nargs='+', type=float, default=[0.003, 0.003], help='coefficient for entropy: [normal, reduce]')
parser.add_argument('--gamma', type=float, default=0.99, help='time decay for baseline update')
parser.add_argument('--controller_start_training', type=int, default=0, help='Epoch that the training of controller starts')
parser.add_argument('--scheduler', type=str, default='naive_cosine', help='type of LR scheduler')
parser.add_argument('--lr_min', type=float, default=0.01, help='min learning rate')
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
# parser.add_argument('--pw', type=str, default='LOOSE_END_supernet', help='The path to pretrained weight if there is(dir)')
parser.add_argument('--debug', action='store_true', default=False, help=' ')
parser.add_argument('--surrogate_only', action='store_true', default=False, help=' ')
parser.add_argument('--accu_batch', type=int, default=10, help=' ')
parser.add_argument('--surrogate_num', type=int, default=4, help=' ')
parser.add_argument('--rt', type=str, default='a', help='reward_type')  
parser.add_argument('--train_portion', type=float, default=0.8, help='data portion for training weights') 
args = parser.parse_args()


if args.op_type=='LOOSE_END_PRIMITIVES':
    args.loose_end = True
else:
    args.loose_end = False

if not os.path.exists(args.prefix):
    os.makedirs(args.prefix)


if args.debug:
    args.save = os.path.join(args.prefix, "predictor/finetune_dataset/debug", args.save)
else:
    args.save = os.path.join(args.prefix, "predictor/finetune_dataset", args.save)
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

    train_queue, valid_queue, test_queue = utils.get_train_queue(args)

    flag = 1

    dir_number = 0
    indir_number = 0

    while flag:
        if dir_number >= len(dir):
            logging.info('all training is done')
            flag = 0
            break

        if not os.path.exists(os.path.join(dir[dir_number], str(indir_number))):
            dir_number = dir_number + 1
            indir_number = 0
            continue
        if os.path.exists(os.path.join(dir[dir_number], str(indir_number), 'surrogate_{}'.format(args.surrogate_num - 1))):
            indir_number = indir_number + 1
            continue

        target_path = os.path.join(dir[dir_number], str(indir_number))
        if not os.path.exists(os.path.join(target_path, 'surrogate_model_0.pt')):
            assert os.path.exists(os.path.join(target_path, 'surrogate_model.pt'))
            shutil.copy(os.path.join(target_path, 'surrogate_model.pt'), os.path.join(target_path, 'surrogate_model_0.pt'))
            torch.save(True, os.path.join(target_path, 'surrogate_0'))

        if not os.path.exists(os.path.join(target_path, 'surrogate_arch_0')):
            logging.info('relocate and test surrogate 0')
            surrogate_model = FinalNet( \
                args.init_channels, CIFAR_CLASSES, args.layers, criterion, device, steps=args.num_nodes, controller_hid=args.controller_hid, entropy_coeff=args.entropy_coeff, edge_hid = args.edge_hid, transformer_nfeat = args.transformer_nfeat, transformer_nhid = args.transformer_nhid, transformer_dropout = args.transformer_dropout, transformer_normalize = args.transformer_normalize, loose_end = args.loose_end, op_type=args.op_type\
                )
            utils.load(surrogate_model, os.path.join(target_path, 'surrogate_model_0.pt'))
            surrogate_model.to(device)
            # surrogate_archs = []
            # surrogate_archs.append([surrogate_model.arch_normal, surrogate_model.arch_reduce])
            torch.save((surrogate_model.arch_normal, surrogate_model.arch_reduce), os.path.join(os.path.join(target_path, 'surrogate_arch_0')))
            logging.info('relocate surrogate archs complete')
            target_model = FinalNet( \
                args.init_channels, CIFAR_CLASSES, args.layers, criterion, device, steps=args.num_nodes, controller_hid=args.controller_hid, entropy_coeff=args.entropy_coeff, edge_hid = args.edge_hid, transformer_nfeat = args.transformer_nfeat, transformer_nhid = args.transformer_nhid, transformer_dropout = args.transformer_dropout, transformer_normalize = args.transformer_normalize, loose_end = args.loose_end, op_type=args.op_type\
                )
            utils.load(target_model, os.path.join(target_path, 'target.pt'))
            target_model.to(device)
            target_model_baseline = FinalNet( \
                args.init_channels, CIFAR_CLASSES, args.layers, criterion, device, steps=args.num_nodes, controller_hid=args.controller_hid, entropy_coeff=args.entropy_coeff, edge_hid = args.edge_hid, transformer_nfeat = args.transformer_nfeat, transformer_nhid = args.transformer_nhid, transformer_dropout = args.transformer_dropout, transformer_normalize = args.transformer_normalize, loose_end = args.loose_end, op_type=args.op_type\
                )
            utils.load(target_model_baseline, os.path.join(target_path, 'target_baseline.pt'))
            target_model_baseline.to(device)

            data_point = OrderedDict()
            acc_clean_target, _ = target_model.test_acc_single(valid_queue, logger, args)
            acc_clean_target_baseline, _ = target_model_baseline.test_acc_single(valid_queue, logger, args)
            acc_clean_surrogate, _ = surrogate_model.test_acc_single(valid_queue, logger, args)
            acc_adv_surrogate = utils.AvgrageMeter()
            acc_adv_baseline = utils.AvgrageMeter()
            for step, (input, target) in enumerate(valid_queue):
                if step > args.accu_batch:
                    break
                n = input.size(0)
                input = input.to(device)
                target = target.to(device)
                acc_adv_surrogate.update(target_model.evaluate_transfer(surrogate_model, input, target).item(), n)
                acc_adv_baseline.update(target_model.evaluate_transfer(target_model_baseline, input, target).item(), n)
            data_point['absolute_reward'] = acc_adv_surrogate.avg - acc_adv_baseline.avg
            data_point['relative_reward'] = (acc_adv_surrogate.avg - acc_adv_baseline.avg) / (acc_clean_target / 100 - acc_adv_baseline.avg)
            data_point['acc_clean_target'] = acc_clean_target
            data_point['acc_clean_target_baseline'] = acc_clean_target_baseline
            data_point['acc_clean_surrogate'] = acc_clean_surrogate
            data_point['acc_adv_surrogate'] = acc_adv_surrogate.avg
            data_point['acc_adv_baseline'] = acc_adv_baseline.avg
            data_point['target_arch'] = (target_model.arch_normal, target_model.arch_reduce)
            data_point['surrogate_arch'] = (surrogate_model.arch_normal, surrogate_model.arch_reduce)
            train_info = {}
            train_info['target_dir'] = target_path
            train_info['surrogate_path'] = os.path.join(target_path, 'surrogate_0.pt')
            train_info['surrogate_index'] = 0
            train_data = []
            train_data.append(data_point)
            logging.info("Dir (%d, %d) absolute_reward=%.2f, relative_reward=%.2f, acc_adv_baseline=%.2f, surrogate_acc_adv=%.2f, acc_clean_target=%.2f, acc_clean_target_baseline=%.2f, acc_clean_surrogate=%.2f", dir_number, 0,\
                data_point['absolute_reward'], data_point['relative_reward'], data_point['acc_adv_baseline'], data_point['acc_adv_surrogate'], data_point['acc_clean_target'], data_point['acc_clean_target_baseline'], data_point['acc_clean_surrogate'])
            torch.save(train_data, os.path.join(target_path, 'train_data_{}'.format(indir_number)))
            logging.info('relocate and test complete')

        same = 1

        target_model = FinalNet( \
         args.init_channels, CIFAR_CLASSES, args.layers, criterion, device, steps=args.num_nodes, controller_hid=args.controller_hid, entropy_coeff=args.entropy_coeff, edge_hid = args.edge_hid, transformer_nfeat = args.transformer_nfeat, transformer_nhid = args.transformer_nhid, transformer_dropout = args.transformer_dropout, transformer_normalize = args.transformer_normalize, loose_end = args.loose_end, op_type=args.op_type\
        )

        arch_exist_max = 0
        surrogate_arch_list = []
        for i in range(10000):
            if not os.path.exists(os.path.join(target_path, 'surrogate_{}'.format(i))):
                torch.save(True, os.path.join(target_path, 'surrogate_{}'.format(i)))
                arch_exist_max = i
                break
        for arch_exist in range(arch_exist_max):
            if os.path.exists(os.path.join(target_path, 'surrogate_arch_{}'.format(arch_exist))):
                surrogate_arch_list.append(torch.load(os.path.join(target_path, 'surrogate_arch_{}'.format(arch_exist)), map_location='cpu'))
                # logging.info(surrogate_arch_list)
            else:
                while not os.path.exists(os.path.join(target_path, 'surrogate_arch_{}'.format(arch_exist))):
                    time.sleep(1)
        assert os.path.exists(os.path.join(target_path, 'surrogate_{}'.format(len(surrogate_arch_list) - 1)))

        surrogate_index = len(surrogate_arch_list)

        assert surrogate_index == arch_exist_max

        utils.load(target_model, os.path.join(target_path, 'target.pt'), only_arch=True)

        while same:
            surrogate_normal = target_model.uni_random_transform(target_model.arch_normal)
            surrogate_reduce = target_model.uni_random_transform(target_model.arch_reduce)
            for (arch_normal, arch_reduce) in surrogate_arch_list:
                if surrogate_normal == arch_normal and surrogate_reduce == arch_reduce:
                    same = 1
                    break
                same = 0
        
        surrogate_arch_list.append((surrogate_normal, surrogate_reduce))

        torch.save(surrogate_arch_list[-1], os.path.join(target_path, 'surrogate_arch_{}'.format(surrogate_index)))

        surrogate_model = FinalNet( \
            args.init_channels, CIFAR_CLASSES, args.layers, criterion, device, steps=args.num_nodes, controller_hid=args.controller_hid, entropy_coeff=args.entropy_coeff, edge_hid = args.edge_hid, transformer_nfeat = args.transformer_nfeat, transformer_nhid = args.transformer_nhid, transformer_dropout = args.transformer_dropout, transformer_normalize = args.transformer_normalize, loose_end = args.loose_end, op_type=args.op_type\
            )

        surrogate_model.arch_normal = surrogate_normal
        surrogate_model.arch_reduce = surrogate_reduce

        logging.info('Training surrogate model')

        surrogate_optimizer = torch.optim.SGD(
            surrogate_model.model_parameters(),
            args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )

        surrogate_model.to(device)
        surrogate_model._model_optimizer = surrogate_optimizer

        if args.scheduler == "naive_cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            surrogate_optimizer, float(args.epochs), eta_min=args.lr_min
            )
        else:
            assert False, "unsupported scheduler type: %s" % args.scheduler

        for epoch in range(args.epochs):
            scheduler.step()
            lr = scheduler.get_last_lr()[0]
            logging.info('epoch %d lr %e', epoch, lr)
            top1 = utils.AvgrageMeter()
            top5 = utils.AvgrageMeter()
            objs = utils.AvgrageMeter()
            for step, (input, target) in enumerate(train_queue):
                surrogate_model.train()
                surrogate_model._model_optimizer.zero_grad()
                n = input.size(0)
                input = input.to(device)
                target = target.to(device)
                logits = surrogate_model._inner_forward(input, surrogate_model.arch_normal, surrogate_model.arch_reduce)
                loss = surrogate_model._criterion(logits, target)
                loss.backward()
                surrogate_model._model_optimizer.step()
                prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
                objs.update(loss.item(), n)
                top1.update(prec1.item(), n)
                top5.update(prec5.item(), n)
                if step % args.report_freq == 0:
                    logging.info('Dir=(%d, %d) Surrogate %d: Step=%03d Loss=%e Top1=%f Top5=%f', dir_number, indir_number, surrogate_index, step, objs.avg, top1.avg, top5.avg)
        logging.info('Dir=(%d, %d)  Surrogate %d: Loss=%e Top1=%f Top5=%f', dir_number, indir_number, surrogate_index, objs.avg, top1.avg, top5.avg)

        utils.save(surrogate_model, os.path.join(target_path, 'surrogate_model_{}.pt'.format(surrogate_index)))
        surrogate_acc_clean, _ = surrogate_model.test_acc_single(valid_queue, logger, args)

        logging.info('Training Completed surrogate {}, testing reward'.format(indir_number))

        utils.load(target_model, os.path.join(target_path, 'target.pt'))
        target_model.to(device)
        target_model_baseline = FinalNet( \
         args.init_channels, CIFAR_CLASSES, args.layers, criterion, device, steps=args.num_nodes, controller_hid=args.controller_hid, entropy_coeff=args.entropy_coeff, edge_hid = args.edge_hid, transformer_nfeat = args.transformer_nfeat, transformer_nhid = args.transformer_nhid, transformer_dropout = args.transformer_dropout, transformer_normalize = args.transformer_normalize, loose_end = args.loose_end, op_type=args.op_type\
        )
        utils.load(target_model_baseline, os.path.join(target_path, 'target_baseline.pt'))
        target_model_baseline.to(device)
        utils.load(surrogate_model, os.path.join(target_path, 'surrogate_model_{}.pt'.format(surrogate_index)))
        surrogate_model.to(device)

        target_model.single = True
        target_model_baseline.single = True
        surrogate_model.single = True

        # reward = utils.AvgrageMeter()
        acc_clean_target, _ = target_model.test_acc_single(valid_queue, logger, args)
        acc_adv_surrogate = utils.AvgrageMeter()
        acc_adv_baseline = utils.AvgrageMeter()
        acc_clean_surrogate, _ = surrogate_model.test_acc_single(valid_queue, logger, args)
        acc_clean_target_baseline, _ = target_model_baseline.test_acc_single(valid_queue, logger, args)

        data_point = OrderedDict()

        for step, (input, target) in enumerate(valid_queue):
            if step > args.accu_batch:
                break
            n = input.size(0)
            input = input.to(device)
            target = target.to(device)
            acc_adv_surrogate.update(target_model.evaluate_transfer(surrogate_model, input, target).item(), n)
            acc_adv_baseline.update(target_model.evaluate_transfer(target_model_baseline, input, target).item(), n)
        data_point['absolute_reward'] = acc_adv_surrogate.avg - acc_adv_baseline.avg
        data_point['relative_reward'] = (acc_adv_surrogate.avg - acc_adv_baseline.avg) / (acc_clean_target / 100 - acc_adv_baseline.avg)
        data_point['acc_clean_target'] = acc_clean_target
        data_point['acc_clean_target_baseline'] = acc_clean_target_baseline
        data_point['acc_clean_surrogate'] = acc_clean_surrogate
        data_point['acc_adv_surrogate'] = acc_adv_surrogate.avg
        data_point['acc_adv_baseline'] = acc_adv_baseline.avg
        data_point['target_arch'] = (target_model.arch_normal, target_model.arch_reduce)
        data_point['surrogate_arch'] = (surrogate_model.arch_normal, surrogate_model.arch_reduce)
        train_info = {}
        train_info['target_dir'] = target_path
        train_info['surrogate_path'] = os.path.join(target_path, 'surrogate_{}.pt'.format(surrogate_index))
        train_info['surrogate_index'] = surrogate_index
        train_data = torch.load(os.path.join(target_path, 'train_data_{}'.format(indir_number)), map_location='cpu')
        train_data.append(data_point)
        torch.save(train_data, os.path.join(target_path, 'train_data_{}'.format(indir_number)))
        logging.info("Dir (%d, %d) absolute_reward=%.2f, relative_reward=%.2f, acc_adv_baseline=%.2f, surrogate_acc_adv=%.2f, acc_clean_target=%.2f, acc_clean_target_baseline=%.2f, acc_clean_surrogate=%.2f", dir_number, indir_number,\
            data_point['absolute_reward'], data_point['relative_reward'], data_point['acc_adv_baseline'], data_point['acc_adv_surrogate'], data_point['acc_clean_target'], data_point['acc_clean_target_baseline'], data_point['acc_clean_surrogate'])
        logging.info('testing complete dir = ({}, {}) done'.format(dir_number, indir_number))




if __name__ == '__main__':
    main()