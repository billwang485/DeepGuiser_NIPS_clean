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
from utils import arch_to_genotype, draw_genotype
from PyPDF2 import PdfFileMerger
from torch.utils.tensorboard import SummaryWriter
'''
This files tests the transferbility isotonicity on supernets and trained-from-scratch models
'''

localtime = time.asctime( time.localtime(time.time()))
x = re.split(r"[\s,(:)]",localtime)
default_EXP = " ".join(x[1:-1])
parser = argparse.ArgumentParser("DeepGuiser")
parser.add_argument('--data', type=str, default='/home/eva_share/datasets/ILSVRC2012/', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--lr', type=float, default=0.1, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=int, default=50, help='report frequency')
parser.add_argument('--test_freq', type=int, default=10, help='test frequency')
# parser.add_argument('--gpu', type=int, default=4, help='gpu device id')#
parser.add_argument('--epochs', type=int, default=90, help='number of signle model training epochs')
parser.add_argument('--init_channels', type=int, default=64, help='number of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--save', type=str, default=default_EXP, help='experiment name')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--train_portion', type=float, default=0.8, help='data portion for training weights')
parser.add_argument('--prefix', type=str, default='.', help='parent save path')
parser.add_argument('--gamma', type=float, default=0.99, help='time decay for baseline update')
parser.add_argument('--scheduler', type=str, default='naive_cosine', help='type of LR scheduler')
parser.add_argument('--lr_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--store', type=int, default=1, help='Whether to store the model')
parser.add_argument('--edge_hid', type=int, default=100, help='edge hidden dimension')
parser.add_argument('--num_nodes', type=int, default=4, help='number of intermediate nodes')
parser.add_argument('--op_type', type=str, default='LOOSE_END_PRIMITIVES', help='LOOSE_END_PRIMITIVES | FULLY_CONCAT_PRIMITIVES')#
parser.add_argument('--transfer', action='store_true', default=True, help='eval the transferability')
parser.add_argument('--arch', type=str, default=' ', help='The path to best archs')#
parser.add_argument('--mode', type=str, default=' ', help='The path to best archs')#
parser.add_argument('--name', type=str, default=' ', help='The path to best archs')#
parser.add_argument('--controller_hid', type=int, default=100, help='controller hidden dimension')
parser.add_argument('--transformer_learning_rate', type=float, default=3e-4, help='learning rate for pruner')
parser.add_argument('--transformer_weight_decay', type=float, default=5e-4, help='learning rate for pruner')
parser.add_argument('--transformer_nfeat', type=int, default=1024, help='feature dimension of each node') 
parser.add_argument('--transformer_nhid', type=int, default=100, help='hidden dimension')
parser.add_argument('--cifar_classes', type=int, default=10, help='hidden dimension')
parser.add_argument('--transformer_dropout', type=float, default=0, help='dropout rate for transformer')
parser.add_argument('--transformer_normalize', action='store_true', default=False, help='use normalize in GCN')
parser.add_argument('--entropy_coeff', nargs='+', type=float, default=[0.003, 0.003], help='coefficient for entropy: [normal, reduce]')
parser.add_argument('--gpus', nargs='+', type=int, default=[0, 1], help='coefficient for entropy: [normal, reduce]')
# parser.add_argument('--mode', nargs='+', type=float, default=[0.003, 0.003], help='coefficient for entropy: [normal, reduce]')
args = parser.parse_args()
assert args.name != ' '
assert args.mode != ' '

if args.mode == 'tb':
    args.seed = 123
if args.op_type=='LOOSE_END_PRIMITIVES':
    args.loose_end = True
else:
    args.loose_end = False

if not os.path.exists(args.prefix):
    os.makedirs(args.prefix)

args.save = os.path.join(args.prefix, 'z/imagenet_models', args.save)
args.cutout = False
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py')+glob.glob('*.sh')+glob.glob('*.yml'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logger = logging.getLogger()
CIFAR_CLASSES = 1000

summaryWriter = SummaryWriter(os.path.join(args.save, "runs"))
def main():
    if torch.cuda.is_available():
        tmp = 'cuda: '
        for i in args.gpus:
            tmp = tmp + str(i)
            if i < len(args.gpus) - 1:
                tmp = tmp + ' ' 
        # torch.cuda.set_device(tmp)
        torch.cuda.manual_seed(args.seed)
        cudnn.benchmark = True
        cudnn.enabled = True
        logging.info('GPU device =')
        logging.info(args.gpus)
    else:
        logging.info('no GPU available, use CPU!!')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    logging.info("args = %s" % args)

    torch.backends.cudnn.enabled = False

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    train_transform, valid_transform = utils._data_transforms_imagenet(args)

    # if CIFAR_CLASSES == 10:
    train_data = dset.ImageFolder(root=os.path.join(args.data, 'ILSVRC2012_img_train'), transform=train_transform)
    test_data = dset.ImageFolder(root=os.path.join(args.data, 'ILSVRC2012_img_val'), transform=valid_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    indices_test = list(range(len(test_data)))
    random.shuffle(indices)
    random.shuffle(indices_test)

    test_queue = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size,
        #sampler=torch.utils.data.sampler.SubsetRandomSampler(indices_test),
        shuffle=False,
        pin_memory=True, num_workers=2
    )

    train_queue = torch.utils.data.DataLoader(
        train_data, args.batch_size,
        #sampler=torch.utils.data.sampler.SubsetRandomSampler(indices),
        shuffle= True,
        pin_memory=True, num_workers=2
    )
    

    # im_model=FinalNet(
    #         args.init_channels, CIFAR_CLASSES, args.layers, criterion, device, steps=args.num_nodes, controller_hid=args.controller_hid, entropy_coeff=args.entropy_coeff, edge_hid = args.edge_hid, transformer_nfeat = args.transformer_nfeat, transformer_nhid = args.transformer_nhid, transformer_dropout = args.transformer_dropout, transformer_normalize = args.transformer_normalize, loose_end = args.loose_end, op_type=args.op_type
    #     )
    # im_model.to(device)
    # target_acc_clean, _ = target_model.test_acc_single(test_queue, logger, args)
    # utils.save(target_model, os.path.join(args.save, 'target.pt'))
    
    assert os.path.exists(args.arch)

    archs = torch.load(args.arch, map_location='cpu')
    shutil.copy(args.arch, os.path.join(args.save, 'archs'))

    if archs[1] == 0:
        num = archs[0]
        # reward_supernet = archs[0][2]
        target_arch = archs[3]
        surrogate_arch = archs[4]
    else:
        assert 0

    logging.info('Training target model')

    im_model = FinalNet(
        args.init_channels, CIFAR_CLASSES, args.layers, criterion, device, steps=args.num_nodes, controller_hid=args.controller_hid, entropy_coeff=args.entropy_coeff, edge_hid = args.edge_hid, transformer_nfeat = args.transformer_nfeat, transformer_nhid = args.transformer_nhid, transformer_dropout = args.transformer_dropout, transformer_normalize = args.transformer_normalize, loose_end = args.loose_end, op_type=args.op_type
    )
    if args.mode == 't' or args.mode == 'tb':
        im_model.arch_normal = target_arch[0]
        im_model.arch_reduce = target_arch[1]
    else:
        im_model.arch_normal = surrogate_arch[0]
        im_model.arch_reduce = surrogate_arch[1]
    im_optimzier = torch.optim.SGD(
        im_model.model_parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(im_optimzier, milestones=[30, 60], gamma=0.1)

    # im_model.to(device)
    torch.nn.DataParallel(im_model, device_ids=args.gpus)
    im_model.cuda()
    im_model._model_optimizer = im_optimzier

    # if args.scheduler == "naive_cosine":
    #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     target_optimiim_optimzierzer, float(args.epochs), eta_min=args.learning_rate_min
    #     )
    # else:
    #     assert False, "unsupported scheduler type: %s" % args.scheduler
    lr = args.lr

    for epoch in range(args.epochs):
        
        logging.info('epoch %d lr %e', epoch, lr)
        top1 = utils.AvgrageMeter()
        top5 = utils.AvgrageMeter()
        objs = utils.AvgrageMeter()
        for step, (input, target) in enumerate(train_queue):
            im_model.train()
            im_model._model_optimizer.zero_grad()
            n = input.size(0)
            input = input.cuda().contiguous()
            target = target.cuda().contiguous()
            logits = im_model._inner_forward(input, im_model.arch_normal, im_model.arch_reduce)
            loss = im_model._criterion(logits, target)

            loss.backward()
            im_model._model_optimizer.step()
            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)
            summaryWriter.add_scalar('loss', objs.avg, step + len(train_queue) * epoch)
            summaryWriter.add_scalar('acc', top1.avg, step + len(train_queue) * epoch)
            if step % args.report_freq == 0:
                logging.info('target_model: Step=%03d Loss=%e Top1=%f Top5=%f', step, objs.avg, top1.avg, top5.avg)
        scheduler.step()
        lr = scheduler.get_lr()[0]
    logging.info('target_model: Loss=%e Top1=%f Top5=%f', objs.avg, top1.avg, top5.avg)

    if args.mode == 't':
        tmp = 'target'
    elif args.mode == 'tb':
        tmp = 'target_baseline'
    else:
        tmp = 'surrogate_model'

    utils.save(im_model, os.path.join(args.save, '{}_{}.pt'.format(args.name, tmp)))
    target_acc_clean, _ = im_model.test_acc_single(test_queue, logger, args)

if __name__ == '__main__':
    main()






