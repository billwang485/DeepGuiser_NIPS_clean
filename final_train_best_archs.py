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
parser.add_argument('--gpu', type=int, default=4, help='gpu device id')
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
parser.add_argument('--op_type', type=str, default='LOOSE_END_PRIMITIVES', help='LOOSE_END_PRIMITIVES | FULLY_CONCAT_PRIMITIVES')
parser.add_argument('--transfer', action='store_true', default=True, help='eval the transferability')
parser.add_argument('--best_archs', type=str, default='LOOSE_END_target_0', help='The path to best archs')
parser.add_argument('--pretrained_weight_target', type=str, default='LOOSE_END_5+50/target_0.pt', help='The path to pretrained weight')
parser.add_argument('--common_save', type=str, default=' ', help=' ')
parser.add_argument('--controller_hid', type=int, default=100, help='controller hidden dimension')
parser.add_argument('--transformer_learning_rate', type=float, default=3e-4, help='learning rate for pruner')
parser.add_argument('--transformer_weight_decay', type=float, default=5e-4, help='learning rate for pruner')
parser.add_argument('--transformer_nfeat', type=int, default=1024, help='feature dimension of each node')
parser.add_argument('--transformer_nhid', type=int, default=100, help='hidden dimension')
parser.add_argument('--transformer_dropout', type=float, default=0, help='dropout rate for transformer')
parser.add_argument('--transformer_normalize', action='store_true', default=False, help='use normalize in GCN')
parser.add_argument('--entropy_coeff', nargs='+', type=float, default=[0.003, 0.003], help='coefficient for entropy: [normal, reduce]')
args = parser.parse_args()

assert args.pretrained_weight_target != ' '

if args.op_type=='LOOSE_END_PRIMITIVES':
    args.loose_end = True
else:
    args.loose_end = False

if not os.path.exists(args.prefix):
    os.makedirs(args.prefix)

args.save = os.path.join(args.prefix, 'final_train', args.save)
args.cutout = False
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py')+glob.glob('*.sh')+glob.glob('*.yml'))
if args.common_save == ' ':
    os.mkdir(os.path.join(args.save, 'common_save'))
    args.common_save = os.path.join(args.save, 'common_save')
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

    target_model=FinalNet(
            args.init_channels, CIFAR_CLASSES, args.layers, criterion, device, steps=args.num_nodes, controller_hid=args.controller_hid, entropy_coeff=args.entropy_coeff, edge_hid = args.edge_hid, transformer_nfeat = args.transformer_nfeat, transformer_nhid = args.transformer_nhid, transformer_dropout = args.transformer_dropout, transformer_normalize = args.transformer_normalize, loose_end = args.loose_end, op_type=args.op_type
        )
    utils.load(target_model, args.pretrained_weight_target)
    target_model.to(device)
    target_acc_clean, _ = target_model.test_acc_single(test_queue, logger, args)
    utils.save(target_model, os.path.join(args.save, 'target.pt'))
    
    if not os.path.exists(os.path.join(args.common_save, 'target.pt')):
        utils.save(target_model, os.path.join(args.common_save, 'target.pt'))
    

    flag = 1
    while flag:

        for i in range(10):
            if os.path.exists(args.best_archs):
                break
            time.sleep(1)

        assert os.path.exists(args.best_archs)

        archs = torch.load(args.best_archs)
        shutil.copy(args.best_archs, os.path.join(args.save, 'best_archs'))
        os.remove(args.best_archs)

        for i, x in enumerate(archs):
            if x[1] == 0:
                num = x[0]
                reward_supernet = x[2]
                target_arch = x[3]
                surrogate_arch = x[4]
                archs[i] = (num, 1, reward_supernet, target_arch, surrogate_arch)
                flag = 1
                break
            flag = 0
        
        torch.save(archs, args.best_archs)

        logging.info('Training surrogate model')

        surrogate_model = FinalNet(
            args.init_channels, CIFAR_CLASSES, args.layers, criterion, device, steps=args.num_nodes, controller_hid=args.controller_hid, entropy_coeff=args.entropy_coeff, edge_hid = args.edge_hid, transformer_nfeat = args.transformer_nfeat, transformer_nhid = args.transformer_nhid, transformer_dropout = args.transformer_dropout, transformer_normalize = args.transformer_normalize, loose_end = args.loose_end, op_type=args.op_type
        )
        surrogate_model.arch_normal = surrogate_arch[0]
        surrogate_model.arch_reduce = surrogate_arch[1]
        surrogate_optimizer = torch.optim.SGD(
            surrogate_model.model_parameters(),
            args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )

        surrogate_model.to(device)
        surrogate_model._model_optimizer = surrogate_optimizer

        if args.scheduler == "naive_cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            surrogate_optimizer, float(args.epochs), eta_min=args.learning_rate_min
            )
        else:
            assert False, "unsupported scheduler type: %s" % args.scheduler

        for epoch in range(args.epochs):
            scheduler.step()
            lr = scheduler.get_lr()[0]
            logging.info('common save to %s', args.common_save)
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
                    logging.info('Surrogate %d: Step=%03d Loss=%e Top1=%f Top5=%f', num, step, objs.avg, top1.avg, top5.avg)
        logging.info('Surrogate %d: Loss=%e Top1=%f Top5=%f', num, objs.avg, top1.avg, top5.avg)

        utils.save(surrogate_model, os.path.join(args.common_save, 'surrogate_model_{}.pt'.format(num)))
        surrogate_acc_clean, _ = surrogate_model.test_acc_single(test_queue, logger, args)


        logging.info('If Target Model baseline is None train it')

        target_model_baseline=FinalNet(
            args.init_channels, CIFAR_CLASSES, args.layers, criterion, device, steps=args.num_nodes, controller_hid=args.controller_hid, entropy_coeff=args.entropy_coeff, edge_hid = args.edge_hid, transformer_nfeat = args.transformer_nfeat, transformer_nhid = args.transformer_nhid, transformer_dropout = args.transformer_dropout, transformer_normalize = args.transformer_normalize, loose_end = args.loose_end, op_type=args.op_type
        )
        if os.path.exists(os.path.join(args.common_save, 'target_baseline.pt')):
            logging.info('pretrained target baseline exists')
            utils.load(target_model_baseline, os.path.join(args.common_save, 'target_baseline.pt'))
            target_model_baseline.to(device)
            target_acc_clean_baseline, _ = target_model_baseline.test_acc_single(test_queue, logger, args)
        else:
            target_baseline_optimizer = torch.optim.SGD(
            target_model_baseline.model_parameters(),
            args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay
            )
            target_model_baseline.to(device)
            target_model_baseline._model_optimizer = target_baseline_optimizer

            if args.scheduler == "naive_cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                target_baseline_optimizer, float(args.epochs), eta_min=args.learning_rate_min
                )
            else:
                assert False, "unsupported scheduler type: %s" % args.scheduler

            target_model_baseline.arch_normal = target_arch[0]
            target_model_baseline.arch_reduce = target_arch[1]  

            for epoch in range(args.epochs):
                scheduler.step()
                lr = scheduler.get_lr()[0]
                logging.info('common save to %s', args.common_save)
                logging.info('epoch %d lr %e', epoch, lr)
                top1 = utils.AvgrageMeter()
                top5 = utils.AvgrageMeter()
                objs = utils.AvgrageMeter()
                for step, (input, target) in enumerate(train_queue):
                    target_model_baseline.train()
                    target_model_baseline._model_optimizer.zero_grad()
                    n = input.size(0)
                    input = input.to(device)
                    target = target.to(device)
                    logits = target_model_baseline._inner_forward(input, target_model_baseline.arch_normal, target_model_baseline.arch_reduce)
                    loss = target_model_baseline._criterion(logits, target)
                    loss.backward()
                    target_model_baseline._model_optimizer.step()
                    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
                    objs.update(loss.item(), n)
                    top1.update(prec1.item(), n)
                    top5.update(prec5.item(), n)
                    if step % args.report_freq == 0:
                        logging.info('Target_Model_Baseline: Step=%03d Loss=%e Top1=%f Top5=%f', step, objs.avg, top1.avg, top5.avg)
                logging.info('Target_Model_Baseline: Loss=%e Top1=%f Top5=%f', objs.avg, top1.avg, top5.avg)

            utils.save(target_model_baseline, os.path.join(args.common_save, 'target_baseline.pt'))
            utils.save(target_model_baseline, os.path.join(args.save, 'target_baseline.pt'))
            target_acc_clean_baseline, _ = target_model_baseline.test_acc_single(test_queue, logger, args)




        logging.info('Training Completed')

        logging.info('Testing transform {}'.format(num))

        surrogate_acc_clean = utils.AvgrageMeter()
        surrogate_acc_adv = utils.AvgrageMeter()
        target_acc_clean_baseline = utils.AvgrageMeter()
        adv_acc_baseline = utils.AvgrageMeter()
        target_acc_clean = utils.AvgrageMeter()
        target_acc_adv = utils.AvgrageMeter()
        reward = utils.AvgrageMeter()

        for step, (input, target) in enumerate(test_queue):
            n = input.size(0)
            input = input.to(device)
            target = target.to(device)
            input_adv, (acc_clean, acc_adv) = surrogate_model.generate_adv_input(input, target, 0.1)
            
            acc_clean, acc_adv = surrogate_model.eval_transfer(input_adv, input, target)
            surrogate_acc_clean.update(acc_clean.item(), n)
            surrogate_acc_adv.update(acc_adv.item(), n)

            input_adv_, (acc_clean, acc_adv) = target_model_baseline.generate_adv_input(input, target, 0.1)
            # logging.info("acc_adv_target_white=%.2f", acc_adv.item())

            (acc_clean, acc_adv_) = target_model.eval_transfer(input_adv_, input, target)
            target_acc_clean_baseline.update(acc_clean.item(), n)
            adv_acc_baseline.update(acc_adv_.item(), n)
            
            (acc_clean, acc_adv) = target_model.eval_transfer(input_adv, input, target)
            target_acc_clean.update(acc_clean.item(), n)
            target_acc_adv.update(acc_adv.item(), n)
            
            reward.update(acc_adv.item() - acc_adv_.item(), n)
            if step % args.report_freq == 0:
                logging.info('common save to %s', args.common_save)
                logging.info('Step=%03d: Surrogate model %d surrogate_acc_clean=%.2f surrogate_acc_adv=%.2f target_acc_clean_baseline=%.2f adv_acc_baseline=%.2f target_acc_clean=%.2f target_acc_adv=%.2f reward=%.2f',\
                    step, num, surrogate_acc_clean.avg, surrogate_acc_adv.avg, target_acc_clean_baseline.avg, adv_acc_baseline.avg, target_acc_clean.avg, target_acc_adv.avg, reward.avg )
        logging.info('Surrogate model %d Final: surrogate_acc_clean=%.2f surrogate_acc_adv=%.2f target_acc_clean_baseline=%.2f adv_acc_baseline=%.2f target_acc_clean=%.2f target_acc_adv=%.2f reward=%.2f',\
                    num, surrogate_acc_clean.avg, surrogate_acc_adv.avg, target_acc_clean_baseline.avg, adv_acc_baseline.avg, target_acc_clean.avg, target_acc_adv.avg, reward.avg )
        logging.info('Orignal reward= %.2f final_train_reward=%.2f final_train_relative_reward=%.2f', reward_supernet, reward.avg, reward.avg / (target_acc_clean.avg - target_acc_clean_baseline.avg))
        save_dict = {}
        save_dict['target_arch'] = target_arch
        save_dict['surrogate_arch'] = surrogate_arch
        save_dict['target_acc_clean'] = target_acc_clean.avg
        save_dict['target_acc_clean_baseline'] = target_acc_clean_baseline.avg
        save_dict['surrogate_acc_clean'] = surrogate_acc_clean.avg
        save_dict['surrogate_acc_adv'] = surrogate_acc_adv.avg
        save_dict['adv_acc_baseline'] = adv_acc_baseline.avg
        save_dict['reward'] = reward.avg
        save_dict['relative_reward'] = reward.avg / (target_acc_clean.avg - target_acc_clean_baseline.avg)
        save_dict['reward_supernet'] = reward_supernet
        torch.save(save_dict, os.path.join(args.common_save, 'surrogate_{}_result'.format(num)))
        torch.save(save_dict, os.path.join(args.save, 'surrogate_{}_result'.format(num)))


if __name__ == '__main__':
    main()






