dirs = ['z/transformer_random_test/May 18 07 41 07',
    'z/transformer_random_test/May 18 07 41 17',
    'z/transformer_random_test/May 18 07 41 52',
    'z/transformer_random_test/May 18 07 42 32',
    'z/transformer_random_test/May 18 07 44 18',
    'z/transformer_random_test/May 18 07 44 49',
    'z/transformer_random_test/May 18 07 45 50',
    'z/transformer_random_test/May 18 07 52 32',
    'z/transformer_random_test/May 18 07 53 02',
    'z/transformer_random_test/May 18 07 54 10',
    'z/transformer_random_test/May 18 07 54 58',
    'z/transformer_random_test/May 18 07 55 26',
    'z/transformer_random_test/May 18 07 55 54',
    'z/transformer_random_test/May 18 07 56 22',
    'z/transformer_random_test/May 18 07 56 43',
    'z/transformer_random_test/May 18 07 57 28',
    'z/transformer_random_test/May 18 08 14 24',
    'z/transformer_random_test/May 18 08 15 15',
    'z/transformer_random_test/May 18 08 15 34',
    'z/transformer_random_test/May 18 08 16 15',
    ]

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
'''
This files tests the transferbility isotonicity on supernets and trained-from-scratch models
'''

localtime = time.asctime( time.localtime(time.time()))
x = re.split(r"[\s,(:)]",localtime)
default_EXP = " ".join(x[1:-1])
parser = argparse.ArgumentParser("DeepGuiser")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=int, default=50, help='report frequency')
parser.add_argument('--test_freq', type=int, default=10, help='test frequency')
parser.add_argument('--gpu', type=int, default=4, help='gpu device id')#
parser.add_argument('--epochs', type=int, default=100, help='number of signle model training epochs')
parser.add_argument('--init_channels', type=int, default=20, help='number of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--save', type=str, default=default_EXP, help='experiment name')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--train_portion', type=float, default=0.8, help='data portion for training weights')
parser.add_argument('--prefix', type=str, default='.', help='parent save path')
parser.add_argument('--gamma', type=float, default=0.99, help='time decay for baseline update')
parser.add_argument('--scheduler', type=str, default='naive_cosine', help='type of LR scheduler')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--store', type=int, default=1, help='Whether to store the model')
parser.add_argument('--edge_hid', type=int, default=100, help='edge hidden dimension')
parser.add_argument('--index', type=int, default=100, help='edge hidden dimension')
parser.add_argument('--num_nodes', type=int, default=4, help='number of intermediate nodes')
parser.add_argument('--op_type', type=str, default='LOOSE_END_PRIMITIVES', help='LOOSE_END_PRIMITIVES | FULLY_CONCAT_PRIMITIVES')#
parser.add_argument('--transfer', action='store_true', default=True, help='eval the transferability')
parser.add_argument('--arch', type=str, default=' ', help='The path to best archs')#
# parser.add_argument('--no', type=int, default=0, help='The path to best archs')#
parser.add_argument('--controller_hid', type=int, default=100, help='controller hidden dimension')
parser.add_argument('--transformer_learning_rate', type=float, default=3e-4, help='learning rate for pruner')
parser.add_argument('--transformer_weight_decay', type=float, default=5e-4, help='learning rate for pruner')
parser.add_argument('--transformer_nfeat', type=int, default=1024, help='feature dimension of each node') 
parser.add_argument('--cifar_classes', type=int, default=10, help='feature dimension of each node') 
parser.add_argument('--transformer_nhid', type=int, default=100, help='hidden dimension')
parser.add_argument('--transformer_dropout', type=float, default=0, help='dropout rate for transformer')
parser.add_argument('--transformer_normalize', action='store_true', default=False, help='use normalize in GCN')
parser.add_argument('--entropy_coeff', nargs='+', type=float, default=[0.003, 0.003], help='coefficient for entropy: [normal, reduce]')
args = parser.parse_args()
# assert args.name == 'ResBlock' or args.name == 'VGG' or args.name == 'Mobilenetv2'
# args.no = int(args.arch.split('/')[-1].split('_')[-1])
if args.op_type=='LOOSE_END_PRIMITIVES':
    args.loose_end = True
else:
    args.loose_end = False

if not os.path.exists(args.prefix):
    os.makedirs(args.prefix)

args.save = os.path.join(args.prefix, 'z/0519', args.save)
args.cutout = False
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py')+glob.glob('*.sh')+glob.glob('*.yml'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logger = logging.getLogger()
CIFAR_CLASSES = args.cifar_classes


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

    # train_queue, valid_queue, test_queue = utils.get_train_queue(args)
    train_queue, test_queue = utils.get_final_train_data(args, CIFAR_CLASSES)

    
    # target_acc_clean, _ = target_model.test_acc_single(test_queue, logger, args)
    # utils.save(target_model, os.path.join(args.save, 'target.pt'))
    
    # assert os.path.exists(args.arch)

    archs = torch.load(os.path.join('derive/gates/May 19 04 41 22', 'train_info_{}').format(args.index), map_location='cpu')
    # shutil.copy(args.arch, os.path.join(args.save, 'archs'))

    if archs[1] == 0:
        num = archs[0]
        # reward_supernet = archs[0][2]
        target_arch = archs[3]
        surrogate_arch = archs[4]
    else:
        assert 0

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
                logging.info('Surrogate: Step=%03d Loss=%e Top1=%f Top5=%f', step, objs.avg, top1.avg, top5.avg)
    logging.info('Surrogate: Loss=%e Top1=%f Top5=%f', objs.avg, top1.avg, top5.avg)

    utils.save(surrogate_model, os.path.join(args.save, 'surrogate_model.pt'))
    surrogate_acc_clean, _ = surrogate_model.test_acc_single(test_queue, logger, args)


    logging.info('Training target_baseline')

    target_model_baseline=FinalNet(
        args.init_channels, CIFAR_CLASSES, args.layers, criterion, device, steps=args.num_nodes, controller_hid=args.controller_hid, entropy_coeff=args.entropy_coeff, edge_hid = args.edge_hid, transformer_nfeat = args.transformer_nfeat, transformer_nhid = args.transformer_nhid, transformer_dropout = args.transformer_dropout, transformer_normalize = args.transformer_normalize, loose_end = args.loose_end, op_type=args.op_type
    )
    # if args.name == 'ResBlock':
    #     if CIFAR_CLASSES == 10:
    #         utils.load(target_model_baseline, 'z/test_one_transform_arch/May 10 05 44 13/target_baseline.pt')
    #     else:
    #         utils.load(target_model_baseline, 'z/test_one_transform_arch/May 10 18 02 39/target_baseline.pt')
    # elif args.name == 'VGG':
    #     if CIFAR_CLASSES == 10:
    #         utils.load(target_model_baseline, './z/two_set_one_test/May 11 12 38 51/target_baseline_cifar_10.pt')
    #     else:
    #         utils.load(target_model_baseline, './z/two_set_one_test/May 11 12 38 51/target_baseline_cifar_100.pt')
    # elif args.name == 'Mobilenetv2':
    #     if CIFAR_CLASSES == 10:
    #         utils.load(target_model_baseline, './z/two_set_one_test/May 11 12 43 34/target_baseline_cifar_10.pt')
    #     else:
    #         utils.load(target_model_baseline, './z/two_set_one_test/May 11 12 43 34/target_baseline_cifar_100.pt')
    utils.load(target_model_baseline, os.path.join(dirs[args.index], 'target_baseline.pt'))

    target_model_baseline.to(device)

    target_model=FinalNet(
            args.init_channels, CIFAR_CLASSES, args.layers, criterion, device, steps=args.num_nodes, controller_hid=args.controller_hid, entropy_coeff=args.entropy_coeff, edge_hid = args.edge_hid, transformer_nfeat = args.transformer_nfeat, transformer_nhid = args.transformer_nhid, transformer_dropout = args.transformer_dropout, transformer_normalize = args.transformer_normalize, loose_end = args.loose_end, op_type=args.op_type
        )
    
    # if args.name == 'ResBlock':
    #     if CIFAR_CLASSES == 10:
    #         utils.load(target_model, 'z/test_one_transform_arch/May 10 05 44 13/target.pt')
    #     else:
    #         utils.load(target_model, 'z/test_one_transform_arch/May 10 18 02 39/target.pt')
    # elif args.name == 'VGG':
    #     if CIFAR_CLASSES == 10:
    #         utils.load(target_model, './z/two_set_one_test/May 11 12 38 51/target_CIFAR_10.pt')
    #     else:
    #         utils.load(target_model, './z/two_set_one_test/May 11 12 38 51/target_CIFAR_100.pt')
    # elif args.name == 'Mobilenetv2':
    #     if CIFAR_CLASSES == 10:
    #         utils.load(target_model, './z/two_set_one_test/May 11 12 43 34/target_CIFAR_10.pt')
    #     else:
    #         utils.load(target_model, './z/two_set_one_test/May 11 12 43 34/target_CIFAR_100.pt')
    utils.load(target_model, os.path.join(dirs[args.index], 'target.pt'))
    # utils.laod(target_model, os.path.join(dirs[args.no], target.pt))
        
    target_model.to(device)

    logging.info('Training Completed')

    logging.info('Testing transform')

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
        target_model.eval()
        target_model_baseline.eval()
        surrogate_model.eval()
        input_adv, (acc_clean, acc_adv) = surrogate_model.generate_adv_input(input, target, 0.031)
        acc_clean, acc_adv = surrogate_model.eval_transfer(input_adv, input, target)
        surrogate_acc_clean.update(acc_clean.item(), n)
        surrogate_acc_adv.update(acc_adv.item(), n)
        input_adv_, (acc_clean, acc_adv) = target_model_baseline.generate_adv_input(input, target, 0.031)
        # logging.info("acc_adv_target_white=%.2f", acc_adv.item())
        (acc_clean, acc_adv_) = target_model.eval_transfer(input_adv_, input, target)
        target_acc_clean_baseline.update(acc_clean.item(), n)
        adv_acc_baseline.update(acc_adv_.item(), n)
        (acc_clean, acc_adv) = target_model.eval_transfer(input_adv, input, target)
        target_acc_clean.update(acc_clean.item(), n)
        target_acc_adv.update(acc_adv.item(), n)
        reward.update(acc_adv.item() - acc_adv_.item(), n)
        if step % args.report_freq == 0:
            logging.info('Step=%03d: Surrogate model %d surrogate_acc_clean=%.2f surrogate_acc_adv=%.2f target_acc_clean_baseline=%.2f adv_acc_baseline=%.2f target_acc_clean=%.2f target_acc_adv=%.2f reward=%.2f',\
                step, num, surrogate_acc_clean.avg, surrogate_acc_adv.avg, target_acc_clean_baseline.avg, adv_acc_baseline.avg, target_acc_clean.avg, target_acc_adv.avg, reward.avg )
    logging.info('Surrogate model %d Final: surrogate_acc_clean=%.2f surrogate_acc_adv=%.2f target_acc_clean_baseline=%.2f adv_acc_baseline=%.2f target_acc_clean=%.2f target_acc_adv=%.2f reward=%.2f',\
                num, surrogate_acc_clean.avg, surrogate_acc_adv.avg, target_acc_clean_baseline.avg, adv_acc_baseline.avg, target_acc_clean.avg, target_acc_adv.avg, reward.avg )
    logging.info(' final_train_reward=%.2f', reward.avg)
    save_dict = {}
    save_dict['target_arch'] = target_arch
    save_dict['surrogate_arch'] = surrogate_arch
    save_dict['target_acc_clean'] = target_acc_clean.avg
    save_dict['target_acc_clean_baseline'] = target_acc_clean_baseline.avg
    save_dict['surrogate_acc_clean'] = surrogate_acc_clean.avg
    save_dict['surrogate_acc_adv'] = surrogate_acc_adv.avg
    save_dict['adv_acc_baseline'] = adv_acc_baseline.avg
    save_dict['reward'] = reward.avg
    torch.save(save_dict, os.path.join(args.save, 'surrogate_gates_result'))

    genotype = arch_to_genotype(target_arch[0], target_arch[1], target_model._steps, target_model.op_type, [5], [5])
    transformed_genotype = arch_to_genotype(surrogate_arch[0], surrogate_arch[1], target_model._steps, target_model.op_type, [5], [5])

    draw_genotype(genotype.normal, target_model._steps, os.path.join(args.save, "normal_target"), genotype.normal_concat)
    draw_genotype(genotype.reduce, target_model._steps, os.path.join(args.save, "reduce_target"), genotype.reduce_concat)
    draw_genotype(transformed_genotype.normal, target_model._steps, os.path.join(args.save, "disguised_normal"), transformed_genotype.normal_concat)
    draw_genotype(transformed_genotype.reduce, target_model._steps, os.path.join(args.save, "disguised_reduce"), transformed_genotype.reduce_concat)
    file_merger = PdfFileMerger()

    file_merger.append(os.path.join(args.save, "normal_target.pdf"))
    file_merger.append(os.path.join(args.save, "reduce_target.pdf"))

    file_merger.write(os.path.join(args.save, "target.pdf"))

    file_merger = PdfFileMerger()

    file_merger.append(os.path.join(args.save, "disguised_normal.pdf"))
    file_merger.append(os.path.join(args.save, "disguised_reduce.pdf"))

    file_merger.write(os.path.join(args.save, "disguised_target.pdf"))

    os.remove(os.path.join(args.save, "normal_target.pdf"))
    os.remove(os.path.join(args.save, "normal_target"))
    os.remove(os.path.join(args.save, "reduce_target"))
    os.remove(os.path.join(args.save, "reduce_target.pdf"))
    os.remove(os.path.join(args.save, "disguised_normal.pdf"))
    os.remove(os.path.join(args.save, "disguised_normal"))
    os.remove(os.path.join(args.save, "disguised_reduce.pdf"))
    os.remove(os.path.join(args.save, "disguised_reduce"))


if __name__ == '__main__':
    main()