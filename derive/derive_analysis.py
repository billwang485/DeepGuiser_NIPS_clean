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

from search_model_gates import NASNetwork as Network
import random
import shutil
import genotypes
import time
import re
localtime = time.asctime( time.localtime(time.time()))
x = re.split(r"[\s,(:)]",localtime)
default_EXP = " ".join(x[1:-1])
parser = argparse.ArgumentParser("NAT")
parser = argparse.ArgumentParser("CompactNAS")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
# parser.add_argument('--train_portion', type=float, default=0.4, help='data portion for training weights')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--init_channels', type=int, default=20, help='number of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
# parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--save', type=str, default=default_EXP, help='experiment name')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--gamma', type=float, default=0.99, help='time decay for baseline update')
parser.add_argument('--n_archs', type=int, default=10, help='number of candidate archs')
parser.add_argument('--prefix', type=str, default='.', help='parent save path')
parser.add_argument('--controller_hid', type=int, default=100, help='temperature for lstm')
parser.add_argument('--entropy_coeff', nargs='+', type=float, default=[0.003, 0.003], help='coefficient for entropy: [normal, reduce]')
parser.add_argument('--transformer_learning_rate', type=float, default=3e-4, help='learning rate for transformer')
parser.add_argument('--transformer_weight_decay', type=float, default=5e-4, help='learning rate for transformer')
parser.add_argument('--edge_hid', type=int, default=100, help='edge hidden dimension')
parser.add_argument('--transformer_nfeat', type=int, default=1024, help='feature dimension of each node')
parser.add_argument('--transformer_nhid', type=int, default=100, help='hidden dimension')
parser.add_argument('--transformer_dropout', type=float, default=0, help='dropout rate for transformer')
parser.add_argument('--transformer_normalize', action='store_true', default=False, help='use normalize in GCN')
parser.add_argument('--arch', type=str, default='ResBlock', help='which architecture to use')
parser.add_argument('--num_nodes', type=int, default=4, help='number of intermediate nodes')
parser.add_argument('--op_type', type=str, default='L', help='LOOSE_END_PRIMITIVES | FULLY_CONCAT_PRIMITIVES')
parser.add_argument('--pwt', type=str, default='pretrain_models/LOOSE/model_twin.pt', help=' ')
parser.add_argument('--pw', type=str, default=' ', help=' ')
parser.add_argument('--accu_batch', type=int, default=10, help=' ')
parser.add_argument('--num', type=int, default=100, help=' ')
parser.add_argument('--debug', action='store_true', default=False, help=' ')
args = parser.parse_args()

assert args.pw != ' '
if args.op_type == 'L':
    args.op_type = 'LOOSE_END_PRIMITIVES'
elif args.op_type == 'B':
    args.op_type = 'BOTTLENECK_PRIMITIVES'
else:
    assert 0
if args.op_type=='LOOSE_END_PRIMITIVES':
    args.loose_end = True
else:
    args.loose_end = False

if not os.path.exists(args.prefix):
    os.makedirs(args.prefix)
tmp = 'derive/gates/analysis'
if args.debug:
    tmp = os.path.join(tmp, 'debug')
args.save = os.path.join(args.prefix, tmp, args.save)

utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py')+glob.glob('*.sh')+glob.glob('*.yml'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logger = logging.getLogger()

CIFAR_CLASSES = 10

sample_arch_count = []
target_arch_list = []
surrogate_arch_list = []
for i in range(9):
    sample_arch_count.append([0] * 9)

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

    genotype = eval("genotypes.%s" % args.arch)
    # arch_normal, arch_reduce = utils.genotype_to_arch(genotype, args.op_type)

    model = Network( \
        args.init_channels, CIFAR_CLASSES, args.layers, criterion, device, steps=args.num_nodes, controller_hid=args.controller_hid, entropy_coeff=args.entropy_coeff, edge_hid = args.edge_hid, transformer_nfeat = args.transformer_nfeat, transformer_nhid = args.transformer_nhid, transformer_dropout = args.transformer_dropout, transformer_normalize = args.transformer_normalize, loose_end = args.loose_end, op_type=args.op_type\
    )

    model_twin = Network( \
        args.init_channels, CIFAR_CLASSES, args.layers, criterion, device, steps=args.num_nodes, controller_hid=args.controller_hid, entropy_coeff=args.entropy_coeff, edge_hid = args.edge_hid, transformer_nfeat = args.transformer_nfeat, transformer_nhid = args.transformer_nhid, transformer_dropout = args.transformer_dropout, transformer_normalize = args.transformer_normalize, loose_end = args.loose_end, op_type=args.op_type\
    )

    _ , derive_queue, _ = utils.get_train_queue(args)
    # model.re_initialize_arch_master()
    model.re_initialize_arch_transformer()
    utils.load(model, args.pw) 
    utils.load(model_twin, args.pwt)
    model.accu_batch = args.accu_batch

    model.to(device)
    model_twin.to(device)

    result = []

    for i in range(args.num):
        target_normal = model.arch_normal_master_demo.forward()
        target_reduce = model.arch_normal_master_demo.forward()
        # target_arch_list.append([target_normal, target_reduce])
        # surrogate_arch_list.append([surrogate_normal, target_reduce])

    # derive optimized arch
        result_ = model.derive_optimized_arch(model_twin, derive_queue, target_normal, target_reduce, 10, logger, args.save, "derive_{}".format(i), normal_concat=genotype.normal_concat, reduce_concat=genotype.reduce_concat)
        result.append(result_)
        target_arch_list.append(result_['target_arch'])
        surrogate_arch_list.append(result_['surrogate_arch'])
        # for i, (op, _, _), (op1, _, _) in enumerate(zip(result_['target_arch'], result_['surrogate_arch'])):
        #     sample_arch_count[op][op1] = sample_arch_count[op][op1] + 1
        for i, ((op, _, _), (op1, _, _)) in enumerate(zip(target_arch_list[-1][0], surrogate_arch_list[-1][0])):
            sample_arch_count[op][op1] = sample_arch_count[op][op1] + 1
        for i, ((op, _, _), (op1, _, _)) in enumerate(zip(target_arch_list[-1][1], surrogate_arch_list[-1][1])):
            sample_arch_count[op][op1] = sample_arch_count[op][op1] + 1
        shutil.copy(os.path.join(args.save, 'target.pdf'), os.path.join(args.save, 'target_{}.pdf'.format(i)))
        shutil.copy(os.path.join(args.save, 'disguised_target.pdf'), os.path.join(args.save, 'disguised_target_{}.pdf'.format(i)))
        train_info = (0, 0, result_["absolute_supernet_reward"], result_["target_arch"], result_["surrogate_arch"])
        # torch.save(result, os.path.join(args.save, 'result_dict'))
        torch.save(train_info, os.path.join(args.save, 'train_info_{}'.format(i)))

    torch.save(result, os.path.join(args.save, 'result_dict'))
    torch.save(target_arch_list, os.path.join(args.save, 'target_arch_list'))
    torch.save(surrogate_arch_list, os.path.join(args.save, 'surrogate_arch_list'))
    torch.save(sample_arch_count, os.path.join(args.save, 'sample_arch_count'))
    

if __name__ == '__main__':
    main()
