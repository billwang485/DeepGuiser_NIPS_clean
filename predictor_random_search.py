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
from utils import draw_genotype, arch_to_genotype
from search_model_predictor import NASNetwork as Network
import random
from PyPDF2 import PdfFileMerger

import genotypes
import time
import re
localtime = time.asctime( time.localtime(time.time()))
x = re.split(r"[\s,(:)]",localtime)
default_EXP = " ".join(x[1:-1])
parser = argparse.ArgumentParser("DeepGuiser")
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
parser.add_argument('--pw', type=str, default=' ', help=' ')
parser.add_argument('--pwp', type=str, default=' ', help=' ')
parser.add_argument('--accu_batch', type=int, default=10, help=' ')
parser.add_argument('--rt', type=str, default='a', help=' ')
parser.add_argument('--debug', action='store_true', default=False, help=' ')
args = parser.parse_args()

# assert args.pw != ' '
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
tmp = 'predictor_random_search'
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
    arch_normal, arch_reduce = utils.genotype_to_arch(genotype, args.op_type)

    model = Network( \
        args.init_channels, CIFAR_CLASSES, args.layers, criterion, device, steps=args.num_nodes, controller_hid=args.controller_hid, entropy_coeff=args.entropy_coeff, edge_hid = args.edge_hid, transformer_nfeat = args.transformer_nfeat, transformer_nhid = args.transformer_nhid, transformer_dropout = args.transformer_dropout, transformer_normalize = args.transformer_normalize, loose_end = args.loose_end, op_type=args.op_type\
    )

    # model.re_initialize_arch_master()
    model.re_initialize_arch_transformer()
    model._initialize_predictor(args, 'WarmUp')
    # utils.load(model.arch_transformer, args.pw) 
    # model.arch_transformer.load_state_dict(torch.load(args.pw, map_location='cpu'))
    model.predictor.load_state_dict(torch.load(args.pwp, map_location='cpu'))
    model.predictor.train()

    model.to(device)

    model.predictor.eval()

    # derive optimized arch
    # result = model.derive_optimized_arch(arch_normal, arch_reduce, args.n_archs, logger, args.save, "derive", normal_concat=genotype.normal_concat, reduce_concat=genotype.reduce_concat)

    best_reward = -np.inf
    # best_optimized_acc_clean = -np.inf
    best_arch_logP = None
    best_arch_ent = None
    best_optimized_arch_normal = None
    best_optimized_arch_reduce = None

    data_trace = []
    # score = self.predictor.forward([arch_normal_, arch_reduce_], [optimized_normal_, optimized_reduce_])

    # acc_clean_baseline, acc_adv_baseline = self._test_transfer(model_twin, test_queue, arch_normal, arch_reduce, arch_normal, arch_reduce)
    logger.info("Sampling candidate architectures ...")
    mean = utils.AvgrageMeter()
    for i in range(args.n_archs):
        # arch_normal = model.arch_normal_master_demo.forward()
        # arch_reduce = model.arch_reduce_master_demo.forward()
        # optimized_normal, optimized_reduce, optimized_logP, optimized_entropy, probs_normal, probs_reduce = model.arch_transformer.forward(arch_normal, arch_reduce)
        optimized_normal = model.uni_random_transform(arch_normal)
        optimized_reduce = model.uni_random_transform(arch_reduce)
        arch_normal_ = []
        arch_reduce_ = []
        optimized_normal_ = []
        optimized_reduce_ = []
        for i in range(len(arch_normal)):
            arch_normal_.append(([arch_normal[i][0]], [arch_normal[i][1]], [arch_normal[i][2]]))
            arch_reduce_.append(([arch_reduce[i][0]], [arch_reduce[i][1]], [arch_reduce[i][2]]))
            optimized_normal_.append(([optimized_normal[i][0]], [optimized_normal[i][1]], [optimized_normal[i][2]]))
            optimized_reduce_.append(([optimized_reduce[i][0]], [optimized_reduce[i][1]], [optimized_reduce[i][2]]))
        score0 = model.predictor.forward([arch_normal_, arch_reduce_], [optimized_normal_, optimized_reduce_]).item()
        score1 = model.predictor.forward([arch_normal_, arch_reduce_], [arch_normal_, arch_reduce_]).item()
        data_trace.append([arch_normal, arch_reduce, optimized_normal, optimized_reduce, score0, score1])
        mean.update(score0 - score1, 1)
        if (score0 - score1) > best_reward:
            best_reward = score0 - score1
            # best_optimized_acc_clean = optimized_acc_clean
            best_optimized_arch_normal = optimized_normal
            best_optimized_arch_reduce = optimized_reduce
            # best_arch_logP = optimized_logP
            # best_arch_ent = optimized_entropy
    # logger.info("Target: acc_clean = %.2f acc_adv = %.2f", acc_clean_baseline, acc_adv_baseline )
    # logger.info("Best surrogate: acc_clean = %.2f acc_adv = %.2f", best_optimized_acc_clean, best_optimized_acc_adv )
    logger.info("Absolute reward = %.2f", best_reward)

    genotype = arch_to_genotype(arch_normal, arch_reduce, model._steps, model.op_type, genotype.normal_concat, genotype.reduce_concat)
    transformed_genotype = arch_to_genotype(best_optimized_arch_normal, best_optimized_arch_reduce, model._steps, model.op_type, genotype.normal_concat, genotype.reduce_concat)
    draw_genotype(genotype.normal, model._steps, os.path.join(args.save, "normal"), genotype.normal_concat)
    draw_genotype(genotype.reduce, model._steps, os.path.join(args.save, "reduce"), genotype.reduce_concat)
    draw_genotype(transformed_genotype.normal, model._steps, os.path.join(args.save, "disguised_normal"), transformed_genotype.normal_concat)
    draw_genotype(transformed_genotype.reduce, model._steps, os.path.join(args.save, "disguised_reduce"), transformed_genotype.reduce_concat)
    file_merger = PdfFileMerger()

    from copy import deepcopy

    file_merger.append(os.path.join(args.save, "normal.pdf"))
    file_merger.append(os.path.join(args.save, "reduce.pdf"))

    file_merger.write(os.path.join(args.save, "target.pdf"))

    file_merger = PdfFileMerger()

    file_merger.append(os.path.join(args.save, "disguised_normal.pdf"))
    file_merger.append(os.path.join(args.save, "disguised_reduce.pdf"))

    file_merger.write(os.path.join(args.save, "disguised_target.pdf"))

    logger.info('genotype = %s', genotype)
    logger.info('optimized_genotype = %s', transformed_genotype)
    result = {}
    result["target_arch"] = (deepcopy(arch_normal), deepcopy(arch_reduce))
    result["surrogate_arch"] = (deepcopy(best_optimized_arch_normal), deepcopy(best_optimized_arch_reduce))
    result["absolute_predictor_reward"] = best_reward
    result["best_arch_logP"] = best_arch_logP
    result["best_arch_ent"] = best_arch_ent

    torch.save(result, os.path.join(args.save, 'result_dict'))
    train_info = (0, 0, result["absolute_predictor_reward"], result["target_arch"], result["surrogate_arch"])
    torch.save(result, os.path.join(args.save, 'result_dict'))
    torch.save(train_info, os.path.join(args.save, 'train_info'))
    torch.save(data_trace, os.path.join(args.save, 'data_trace'))
    logging.info(mean.avg)

if __name__ == '__main__':
    main()
