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
from distillation import Linf_PGD
from matplotlib import pyplot as plt
from genotypes import ResBlock,VGG
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
parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
parser.add_argument('--init_channels', type=int, default=20, help='number of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--save', type=str, default=default_EXP, help='experiment name')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--train_portion', type=float, default=0.4, help='data portion for training weights')
parser.add_argument('--n_archs', type=int, default=10, help='number of candidate archs')
parser.add_argument('--prefix', type=str, default='.', help='parent save path')
parser.add_argument('--controller_hid', type=int, default=100, help='controller hidden dimension')
parser.add_argument('--entropy_coeff', nargs='+', type=float, default=[0.000, 0.000], help='coefficient for entropy: [normal, reduce]')
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
parser.add_argument('--pw', type=str, default=' ', help='The path to pretrained weight if there is(dir)')
parser.add_argument('--test_num', type=int, default='100', help='')
parser.add_argument('--debug', action='store_true', default=False, help='use normalize in GCN')
args = parser.parse_args()

assert args.pw != ' '
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
utils.parse_args_(args)

if args.op_type=='LOOSE_END_PRIMITIVES':
    args.loose_end = True
else:
    args.loose_end = False

if not os.path.exists(args.prefix):
    os.makedirs(args.prefix)

if args.debug:
    args.save = os.path.join(args.prefix,'z/test_variance/debug', args.save)
else:
    args.save = os.path.join(args.prefix,'z/test_variance', args.save)
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

    network_args = 'Network( \
        args.init_channels, CIFAR_CLASSES, args.layers, criterion, device, steps=args.num_nodes, controller_hid=args.controller_hid, entropy_coeff=args.entropy_coeff, edge_hid = args.edge_hid, transformer_nfeat = args.transformer_nfeat, transformer_nhid = args.transformer_nhid, transformer_dropout = args.transformer_dropout, transformer_normalize = args.transformer_normalize, loose_end = args.loose_end, op_type=args.op_type\
    )'

    model = Network( \
        args.init_channels, CIFAR_CLASSES, args.layers, criterion, device, steps=args.num_nodes, controller_hid=args.controller_hid, entropy_coeff=args.entropy_coeff, edge_hid = args.edge_hid, transformer_nfeat = args.transformer_nfeat, transformer_nhid = args.transformer_nhid, transformer_dropout = args.transformer_dropout, transformer_normalize = args.transformer_normalize, loose_end = args.loose_end, op_type=args.op_type\
    )

    model_twin = Network( \
        args.init_channels, CIFAR_CLASSES, args.layers, criterion, device, steps=args.num_nodes, controller_hid=args.controller_hid, entropy_coeff=args.entropy_coeff, edge_hid = args.edge_hid, transformer_nfeat = args.transformer_nfeat, transformer_nhid = args.transformer_nhid, transformer_dropout = args.transformer_dropout, transformer_normalize = args.transformer_normalize, loose_end = args.loose_end, op_type=args.op_type\
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
        shuffle = False,
        pin_memory=True, num_workers=2
    )

    if args.scheduler == "naive_cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            model_optimizer, float(args.epochs), eta_min=args.learning_rate_min
        )
    else:
        assert False, "unsupported schudeler type: %s" % args.scheduler

    
    # logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
    model._model_optimizer = model_optimizer
    model_twin._model_optimizer = model_twin_optimizer

    transformer = Transformer(model, model_twin, args)

    utils.load(model, os.path.join(args.pw, 'model.pt'))
    # model.re_initialize_arch_master()
    model.re_initialize_arch_transformer()
    utils.load(model_twin, os.path.join(args.pw, 'model_twin.pt'))
    # model_twin.re_initialize_arch_master()
    model_twin.re_initialize_arch_transformer()

    model.to(device)
    model_twin.to(device)

    # transformer.model = model_twin

    arch_normal, arch_reduce = utils.genotype_to_arch(ResBlock)

    eps = 0.031

    acc_adv_list = []

    batch_num = 5

    for i in range(args.test_num):
        acc_clean_  = utils.AvgrageMeter()
        acc_adv_  = utils.AvgrageMeter()
        for step, (input, target) in enumerate(test_queue):
            n = input.size(0)
            if step >= batch_num:
                break
            input = input.to(device)
            target = target.to(device)
            input_adv = Linf_PGD(model_twin, arch_normal, arch_reduce, input, target, eps, alpha= eps / 10, steps = 10, rand_start=True)
            logits_clean = model._inner_forward(input, arch_normal, arch_reduce)
            logits_adv = model._inner_forward(input_adv, arch_normal, arch_reduce)
            acc_clean_.update(utils.accuracy(logits_clean, target)[0].item(), n)
            acc_adv_.update(utils.accuracy(logits_adv, target)[0].item(), n)
        logging.info('testing resblock time %d:acc_adv = %.2f', i, acc_adv_.avg)
        acc_adv_list.append(acc_adv_.avg)

    acc_adv_list = np.array(acc_adv_list)
    
    variance = np.var(acc_adv_list)
    mean = np.mean(acc_adv_list)
    min_acc = np.min(acc_adv_list)
    max_acc = np.max(acc_adv_list)
    total_step = 10
    step = (max_acc - min_acc) / total_step
    freqs = [0] * total_step
    fig, ax = plt.subplots()  

    plt.hist(acc_adv_list, bins=total_step, density=True)
    plt.text(0.5, 1.1,'var = {} mean = {} min={} max={}'.format(variance, mean, min_acc, max_acc),
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
    ax.set_ylabel('density')
    ax.set_xlabel('adversarial_accuracy')

    plt.savefig(os.path.join(args.save, 'VGG{}.png'.format(batch_num)))

    # save model
    if args.store == 1:
        utils.save(model, os.path.join(args.save, 'models.pt'))

if __name__ == '__main__':
    main()

