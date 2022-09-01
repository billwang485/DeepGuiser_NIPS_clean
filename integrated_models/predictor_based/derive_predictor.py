import os
import sys
STEM_WORK_DIR = os.path.join(os.getcwd(), "..", "..")
sys.path.append(STEM_WORK_DIR)
import glob
import time
import shutil
import re
import random
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
import utils
from integrated_models.predictor_based.predictor_based_disguiser import PredictorBasedDisguiser

import genotypes
parser = argparse.ArgumentParser("DeepGuiser")
parser.add_argument('--save_freq', type=int, default=1000, help='save frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--iteration', "-its", type=int, default=10000, help='number of training iteration')
parser.add_argument('--save', type=str, default=utils.localtime_as_dirname(), help='experiment name')
parser.add_argument('--arch', type=str, default="ResBlock", help='target arch to transform')
parser.add_argument('--pretrained_weight', "-pw", type=str, default="ResBlock", help='target arch to transform')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--prefix', type=str, default='.', help='parent save path')
parser.add_argument('--entropy_coeff', nargs='+', type=float, default=[0.005, 0.000], help='coefficient for entropy: [normal, reduce]')
parser.add_argument('--gamma', type=float, default=0.99, help='time decay for baseline update')
parser.add_argument('--controller_start_training', type=int, default=0, help='Epoch that the training of controller starts')
parser.add_argument('--num_nodes', type=int, default=4, help='number of intermediate nodes')
parser.add_argument('--op_type', type=str, default='LOOSE_END_PRIMITIVES', help='LOOSE_END_PRIMITIVES | FULLY_CONCAT_PRIMITIVES')
parser.add_argument('--predictor_config', type=str, default='configs/predictor_config.yaml', help='predictor config file')
parser.add_argument('--transformer_config', type=str, default='configs/transformer_config.yaml', help='transformer config file')
parser.add_argument('--search_space_config', type=str, default='configs/search_space_config.yaml', help='search space config file')
parser.add_argument('--strategy_config', type=str, default='configs/strategy_config.yaml', help='search strategy config file (how to compute reward, how to update transformer parameters)')
parser.add_argument('--optimizer_config', type=str, default='configs/optimizer_config.yaml', help='optimizer config file')
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

utils.preprocess_exp_dir(args, "derive")

utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py')+glob.glob('*.sh')+glob.glob('*.yml'))

logger = utils.initialize_logger(args)

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

    predictor_config = utils.load_yaml(args.predictor_config)
    shutil.copy(args.predictor_config, os.path.join(args.save, "predictor_config.yaml"))
    search_space_config = utils.load_yaml(args.search_space_config)
    shutil.copy(args.search_space_config, os.path.join(args.save, "search_space_config.yaml"))
    transformer_config = utils.load_yaml(args.transformer_config)
    shutil.copy(args.transformer_config, os.path.join(args.save, "transformer_config.yaml"))
    strategy_config = utils.load_yaml(args.strategy_config)
    shutil.copy(args.strategy_config, os.path.join(args.save, "strategy_config.yaml"))

    model = PredictorBasedDisguiser(
        device, predictor_config, search_space_config, transformer_config, strategy_config, args.save
    )

    utils.load_predictor_based_disguiser(model.arch_transformer, args.pretrained_weight) 

    model.to(device)

    model.set_thre(args)

    genotype = eval("genotypes.%s" % args.arch)

    arch_normal, arch_reduce = utils.genotype_to_arch(genotype, args.op_type)

    for i in range(1):

        result = model.derive_optimized_arch(arch_normal, arch_reduce, args.n_archs, logger, args.save, "derive", normal_concat=genotype.normal_concat, reduce_concat=genotype.reduce_concat)

        # torch.save(result, os.path.join(args.save, 'result_dict'))
        train_info = (0, 0, result["absolute_predictor_reward"], result["target_arch"], result["surrogate_arch"])
        torch.save(result, os.path.join(args.save, 'result_dict_{}'.format(i)))
        torch.save(train_info, os.path.join(args.save, 'train_info_{}'.format(i)))

        shutil.copy(os.path.join(args.save, 'target.pdf'), os.path.join(args.save, 'target_{}.pdf'.format(i)))
        shutil.copy(os.path.join(args.save, 'disguised_target.pdf'), os.path.join(args.save, 'disguised_target_{}.pdf'.format(i)))

if __name__ == '__main__':
    main()
