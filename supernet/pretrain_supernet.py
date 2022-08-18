import os
import sys
sys.path.append(os.path.join(os.getcwd(), '..'))
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
from basic_parts.basic_integrated_model import NASNetwork as Network


'''
This file trains supernet and twin supernet for 50 epochs and save them
You can bypass the first 50 epochs of training by loading pretrained models
'''

parser = argparse.ArgumentParser("DeepGuiser")
parser.add_argument('--data', type=str, default=os.path.join(os.getcwd(), '..','../data'), help='location of the data corpus')
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
args = parser.parse_args()

if args.op_type=='LOOSE_END_PRIMITIVES':
    args.loose_end = True
else:
    args.loose_end = False
args.cutout = False

utils.preprocess_exp_dir(args)
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py')+glob.glob('*.sh')+glob.glob('*.yml'))

utils.initialize_logger(args)

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
    
    model = Network(
        args.init_channels, CIFAR_CLASSES, args.layers, criterion, device, steps=args.num_nodes, controller_hid=args.controller_hid, entropy_coeff=args.entropy_coeff, edge_hid = args.edge_hid, transformer_nfeat = args.transformer_nfeat, transformer_nhid = args.transformer_nhid, transformer_dropout = args.transformer_dropout, transformer_normalize = args.transformer_normalize, loose_end = args.loose_end, op_type=args.op_type
    )
    model_optimizer = torch.optim.SGD(
        model.model_parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    train_queue, test_queue, _ = utils.get_cifar_data_queue(args)

    if args.scheduler == "naive_cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            model_optimizer, float(args.epochs), eta_min=args.learning_rate_min
        )
    else:
        assert False, "unsupported scheduler type: %s" % args.scheduler

    model.to(device)

    normal_list = []
    reduce_list = []

    for i in range(args.test_archs):
        normal_list.append(model.arch_normal_master.forward())
        reduce_list.append(model.arch_reduce_master.forward())

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
    model._model_optimizer = model_optimizer

    for epoch in range(args.epochs+1):
        scheduler.step()
        lr = scheduler.get_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)

        logging.info('Updating Shared Parameters')
        update_w(train_queue, model, device, epoch)
        if (epoch) % args.test_freq == 0:
            valid_acc = utils.AvgrageMeter()
            for i in range(args.test_archs):
                valid_acc.update(model._test_acc(test_queue, normal_list[i], reduce_list[i]), 1)
            summaryWriter.add_scalar('valid_acc', valid_acc.avg, epoch)
        
        if epoch % 50 == 0:
            utils.save(model, os.path.join(args.save, 'supernet_{}.pt'.format(epoch)))

    # save model
    if args.store == 1:
        utils.save(model, os.path.join(args.save, 'supernet.pt'))

def update_w(valid_queue, model, device, epoch):
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
            summaryWriter.add_scalar('train_loss', objs.avg, step + epoch * len(valid_queue))
            summaryWriter.add_scalar('train_acc', top1.avg, step + epoch * len(valid_queue))


if __name__ == '__main__':
    main()






