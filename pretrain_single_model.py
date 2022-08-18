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
from search_model_twin import NASNetwork as Network
import random
from single_model import FinalNetwork as FinalNet
'''
This file trains supernet and twin supernet for 50 epochs and save them
You can bypass the first 50 epochs of training by loading pretrained models
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
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='number of training epochs')
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
parser.add_argument('--name', type=str, default='target', help=' ')
parser.add_argument('--archs', type=str, default=' ', help=' ')
parser.add_argument('--save_archs', type=str, default='LOOSE_END_5+50', help=' ')
args = parser.parse_args()

assert args.archs != ''

if args.op_type=='LOOSE_END_PRIMITIVES':
    args.loose_end = True
else:
    args.loose_end = False

if not os.path.exists(args.prefix):
    os.makedirs(args.prefix)

args.save = os.path.join(args.prefix, "Pretrained_Single_Models", args.save)
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

    flag = 1

    while flag:
        model = FinalNet(
            args.init_channels, CIFAR_CLASSES, args.layers, criterion, device, steps=args.num_nodes, controller_hid=args.controller_hid, entropy_coeff=args.entropy_coeff, edge_hid = args.edge_hid, transformer_nfeat = args.transformer_nfeat, transformer_nhid = args.transformer_nhid, transformer_dropout = args.transformer_dropout, transformer_normalize = args.transformer_normalize, loose_end = args.loose_end, op_type=args.op_type
        )

        for i in range(10):
            if os.path.exists(args.archs):
                break
            time.sleep(1)

        assert os.path.exists(args.archs)

        archs = torch.load(args.archs)
        shutil.copy(args.archs, os.path.join(args.save, 'archs'))
        os.remove(args.archs)

        for i, x in enumerate(archs):
            if x[1] == 0:
                num = x[0]
                arch_normal = x[2]
                arch_reduce = x[3]
                archs[i] = (num, 1, arch_normal, arch_reduce)
                flag = 1
                break
            flag = 0
        
        torch.save(archs, args.archs)

        if flag == 0:
            break

        model.arch_normal = arch_normal
        model.arch_reduce = arch_reduce


        model_optimizer = torch.optim.SGD(
            model.model_parameters(),
            args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )

        if args.scheduler == "naive_cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                model_optimizer, float(args.epochs), eta_min=args.learning_rate_min
            )
        else:
            assert False, "unsupported scheduler type: %s" % args.scheduler

        model.to(device)

        logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
        model._model_optimizer = model_optimizer

        for epoch in range(args.epochs+1):
            scheduler.step()
            lr = scheduler.get_lr()[0]
            logging.info('epoch %d lr %e', epoch, lr)

            logging.info('Updating Shared Parameters of {} {}'.format(args.name, num))
            update_w(train_queue, model, device, num)
           
            utils.save(model, os.path.join(args.save_archs, '{}_{}.pt'.format(args.name, num)))
        # save model
        if args.store == 1:
            utils.save(model, os.path.join(args.save_archs, '{}_{}.pt'.format(args.name, num)))


def update_w(valid_queue, model, device, num):
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
            logging.info('Updating W of %s %d: Step=%03d Loss=%e Top1=%f Top5=%f Normal_ENT=%f, Reduce_ENT=%f',
                         args.name, num, step, objs.avg, top1.avg, top5.avg, normal_ent.avg, reduce_ent.avg)


if __name__ == '__main__':
    main()






