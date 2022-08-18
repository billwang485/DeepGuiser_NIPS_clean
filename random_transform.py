import os
import sys
import glob
import numpy as np
import torch
from genotypes import ResBlock
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
from nat_learner_twin import Transformer
import random
import shutil
from distillation import Linf_PGD
from single_model import FinalNetwork as FinalNet

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
parser.add_argument('--num', type=int, default=500, help=' ')
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
parser.add_argument('--pw', type=str, default='pretrain_models/LOOSE', help='The path to pretrained weight if there is')
# parser.add_argument('--pw_twin', type=str, default=' ', help='The path to pretrained weight if there is')
parser.add_argument('--num_surrogate', type=int, default=500, help=' ')
parser.add_argument('--accu_batch', type=int, default=10, help=' ')
parser.add_argument('--target_model', type=str, default=' ', help=' ')
parser.add_argument('--test_arch', action='store_true', default=True, help=' ')
args = parser.parse_args()

if args.op_type=='LOOSE_END_PRIMITIVES':
    args.loose_end = True
else:
    args.loose_end = False

if not os.path.exists(args.prefix):
    os.makedirs(args.prefix)

args.save = os.path.join(args.prefix, "random_transform", args.save)
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

    # target_model = FinalNet(
    #     args.init_channels, CIFAR_CLASSES, args.layers, criterion, device, steps=args.num_nodes, controller_hid=args.controller_hid, entropy_coeff=args.entropy_coeff, edge_hid = args.edge_hid, transformer_nfeat = args.transformer_nfeat, transformer_nhid = args.transformer_nhid, transformer_dropout = args.transformer_dropout, transformer_normalize = args.transformer_normalize, loose_end = args.loose_end, op_type=args.op_type
    # )

    train_queue, valid_queue, _ = utils.get_train_queue(args)

    model.to(device)
    model_twin.to(device)

    utils.load(model, os.path.join(args.pw, 'model.pt'))
    utils.load(model_twin, os.path.join(args.pw, 'model_twin.pt'))
    
    
    utils.save(model, os.path.join(args.save, 'model.pt'))
    utils.save(model, os.path.join(args.save, 'model_twin.pt'))

    # utils.load(target_model, args.target_model)

    # if args.test_arch:
    #     # target_model.to(device)
    #     logging.info('Testing Target Model')
        
    #     acc_clean = utils.AvgrageMeter()
    #     for step, (input, target) in enumerate(valid_queue):
    #         n = input.size(0)
    #         input = input.to(device)
    #         target = target.to(device)

    #         target_model.eval()

    #         logits = target_model._inner_forward(input, target_model.arch_normal, target_model.arch_reduce)
    #         acc_clean_ = utils.accuracy(logits, target, topk=(1, 5))[0] / 100.0

    #         acc_clean.update(acc_clean_, n)

    #         if step % args.report_freq == 0:
    #             # logging.info('save to {}'.format(args.save))
    #             logging.info('Testing target model: Step=%03d Top1=%e ',
    #                     step, acc_clean.avg)
    #     logging.info('Testing target model: Top1=%e ',
    #                     acc_clean.avg)

    # target_normal = target_model.arch_normal
    # target_reduce = target_model.arch_reduce

    # utils.save(target_model, os.path.join(args.save, 'target_model.pt'))
    arch_normal, arch_reduce = utils.genotype_to_arch(ResBlock, model.op_type)

    eps = 0.031
    tmp = 0

    archs = []

    for num in range(args.num):
        logging.info('Testing Transform {}'.format(num))
        logging.info('save to {}'.format(args.save))
        surrogate_normal = model.uni_random_transform(arch_normal)
        surrogate_reduce = model.uni_random_transform(arch_reduce)
        acc_clean = utils.AvgrageMeter()
        optimized_acc_adv = utils.AvgrageMeter()
        acc_adv = utils.AvgrageMeter()
        for step, (input, target) in enumerate(valid_queue):
            if step >= args.accu_batch:
                break
            n = input.size(0)
            input = input.to(device)
            target = target.to(device)
            input_adv = Linf_PGD(model_twin, surrogate_normal, surrogate_reduce, input, target, eps, alpha=eps/10, steps=10)
            input_adv_ = Linf_PGD(model_twin, arch_normal, arch_reduce, input, target, eps, alpha=eps/10, steps=10)

            logits = model._inner_forward(input, arch_normal, arch_reduce)
            acc_clean_ = utils.accuracy(logits, target, topk=(1, 5))[0] / 100.0

            logits = model._inner_forward(input_adv, arch_normal, arch_reduce)
            acc_adv_ = utils.accuracy(logits, target, topk=(1, 5))[0] / 100.0


            logits = model._inner_forward(input_adv_, arch_normal, arch_reduce)
            optimized_acc_adv_ = utils.accuracy(logits, target, topk=(1, 5))[0] / 100.0

            acc_clean.update(acc_clean_, n)
            acc_adv.update(acc_adv_, n)
            optimized_acc_adv.update(optimized_acc_adv_, n)


            if step % args.report_freq == 0:
                # logging.info('save to {}'.format(args.save))
                logging.info('Transform %d: Step=%03d acc_clean=%e acc_adv=%f self_attack_acc_adv=%f reward=%f',
                         num, step, acc_clean.avg, acc_adv.avg, optimized_acc_adv.avg, acc_adv.avg - optimized_acc_adv.avg)
        logging.info('Transform %d: acc_clean=%e acc_adv=%f self_attack_acc_adv=%f reward=%f',
                         num, acc_clean.avg, acc_adv.avg, optimized_acc_adv.avg, acc_adv.avg - optimized_acc_adv.avg)
        update_arch(archs, arch_normal, arch_reduce, surrogate_normal, surrogate_reduce, acc_adv.avg - optimized_acc_adv.avg, args)
            
        if len(archs) > tmp * 50:
            tmp = tmp + 1
            shutil.copy(os.path.join(args.save, 'archs'), os.path.join(args.save, 'archs_{}'.format(len(archs))))
            utils.save(model, os.path.join(args.save, 'model.pt'))
            utils.save(model, os.path.join(args.save, 'model_twin.pt'))

    # save model
    if args.store == 1:
        utils.save(model, os.path.join(args.save, 'model.pt'))
        utils.save(model, os.path.join(args.save, 'model_twin.pt'))
        

def update_arch(arch_list, arch_normal, arch_reduce, optimized_normal, optimized_reduce, reward, args):
    if len(arch_list) < args.num:
        arch_list.append((reward, (arch_normal, arch_reduce), (optimized_normal, optimized_reduce)))
        arch_list.sort(reverse=True, key=lambda x: x[0])
    else:
        arch_list.sort(reverse=True, key=lambda x: x[0])
        if reward > arch_list[-1][0]:
            arch_list[-1] = ((reward, (arch_normal, arch_reduce), (optimized_normal, optimized_reduce)))
    arch_list.sort(reverse=True, key=lambda x: x[0])
    torch.save(arch_list, os.path.join(args.save, 'archs'))

if __name__ == '__main__':
    main()






