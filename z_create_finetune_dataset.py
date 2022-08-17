import os
from shutil import copy
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
name = 'train_data_constrain_'
name1 = 'z/high_data_constrain'
count = 0
for i, path in enumerate(dir):
    if os.path.exists(os.path.join(path, name)):
        count = count + 1
        copy(os.path.join(path, name), os.path.join(name1, str(i)))

data = []

for i in range(count):
    tmp = torch.load(os.path.join(name1, str(i)), map_location= 'cpu')
    data.extend(tmp)

torch.save(data, os.path.join(name1, name))


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
tmp = 'predictor/test_finetune_on_supernet'
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

    model = Network( \
        args.init_channels, CIFAR_CLASSES, args.layers, criterion, device, steps=args.num_nodes, controller_hid=args.controller_hid, entropy_coeff=args.entropy_coeff, edge_hid = args.edge_hid, transformer_nfeat = args.transformer_nfeat, transformer_nhid = args.transformer_nhid, transformer_dropout = args.transformer_dropout, transformer_normalize = args.transformer_normalize, loose_end = args.loose_end, op_type=args.op_type\
    )

    model_twin = Network( \
        args.init_channels, CIFAR_CLASSES, args.layers, criterion, device, steps=args.num_nodes, controller_hid=args.controller_hid, entropy_coeff=args.entropy_coeff, edge_hid = args.edge_hid, transformer_nfeat = args.transformer_nfeat, transformer_nhid = args.transformer_nhid, transformer_dropout = args.transformer_dropout, transformer_normalize = args.transformer_normalize, loose_end = args.loose_end, op_type=args.op_type\
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
        shuffle = False,
        pin_memory=True, num_workers=2
    )
    # train_queue, test_queue, _ = utils.get_train_queue(args)

    utils.load(model, os.path.join(args.pw, 'model.pt'))
    model.re_initialize_arch_transformer()
    utils.load(model_twin, os.path.join(args.pw, 'model_twin.pt'))
    model_twin.re_initialize_arch_transformer()
    transformer = Transformer(model, model_twin, args)
    # model._initialize_predictor("WarmUp")

    model.to(device)
    model_twin.to(device)

    assert (args.rt == 'a' or args.rt == 'r')

    if args.constrain:
        _name = 'train_data_constrain_'
    else:
        _name = 'train_data_free_'

    model.reward_type = 'absolute' if args.rt == 'a' else 'relative'

    train_dataset = []

    testlist = torch.load(os.path.join(name1, name), map_location='cpu')

    for index, data_point in enumerate(testlist):
        
        logging.info('number %d', index)

        target_normal, target_reduce = data_point['target_arch']


        surrogate_normal, surrogate_reduce = data_point['surrogate_arch']

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
            # logging.info("number: %d step: %d", index, step)
            reward.update(reward_, n)
            acc_clean.update(acc_clean_, n)
            acc_adv.update(acc_adv_, n)
            optimized_acc_adv.update(optimized_acc_adv_, n)
            optimized_acc_clean.update(optimized_acc_clean_, n)
        logging.info("absolute_reward=%.2f, relative_reward=%.2f, target_acc_clean=%.2f, surrogate_acc_clean=%.2f, acc_adv=%.2f, surrogate_acc_adv=%.2f",\
            reward.avg, (optimized_acc_adv.avg - acc_adv.avg) / (acc_clean.avg - acc_adv.avg), acc_clean.avg, optimized_acc_clean.avg, acc_adv.avg, optimized_acc_adv.avg)
        # data_point = OrderedDict()
        
        # data_point["target_arch"] = (target_normal, target_reduce)
        # data_point["surrogate_arch"] = (surrogate_normal, surrogate_reduce)
        data_point["absolute_reward_supernet"] = reward.avg
        data_point["relative_reward_supernet"] = (optimized_acc_adv.avg - acc_adv.avg) / (acc_clean.avg - acc_adv.avg)
        data_point["target_acc_clean_supernet"] = acc_clean.avg
        data_point["surrogate_acc_clean_supernet"] = optimized_acc_clean.avg
        data_point["acc_adv_supernet"] = acc_adv.avg
        data_point["surrogate_acc_adv_supernet"] = optimized_acc_adv.avg
        # data_point["constrain"] = args.constrain
        data_point["supernet_test_info"] = {"index": index, "info": args.save}

        assert index == len(train_dataset)

        train_dataset.append(data_point)

        

        if len(data_point) % args.save_freq == 0:
            torch.save(train_dataset, os.path.join(args.save, (_name + str(len(data_point)))))
        
        torch.save(train_dataset, os.path.join(args.save, _name))
    

if __name__ == '__main__':
    main()