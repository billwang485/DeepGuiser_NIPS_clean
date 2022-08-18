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
import genotypes
from search_model_predictor import NASNetwork as Network
from nat_learner_predictor import Transformer
import random
from torch.utils.tensorboard import SummaryWriter
localtime = time.asctime( time.localtime(time.time()))
x = re.split(r"[\s,(:)]",localtime)
default_EXP = " ".join(x[1:-1])
parser = argparse.ArgumentParser("DeepGuiser")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--lr', type=float, default=3e-4, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--save_freq', type=int, default=100, help='save frequency')
# parser.add_argument('--test_freq', type=int, default=10, help='test frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--its', type=int, default=1000, help='number of training iteration')
parser.add_argument('--init_channels', type=int, default=20, help='number of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--save', type=str, default=default_EXP, help='experiment name')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
# parser.add_argument('--train_portion', type=float, default=0.4, help='data portion for training weights')
# parser.add_argument('--n_archs', type=int, default=10, help='number of candidate archs')
parser.add_argument('--prefix', type=str, default='.', help='parent save path')
parser.add_argument('--controller_hid', type=int, default=100, help='controller hidden dimension')
parser.add_argument('--entropy_coeff', nargs='+', type=float, default=[0.003, 0.000], help='coefficient for entropy: [normal, reduce]')
parser.add_argument('--gamma', type=float, default=0.99, help='time decay for baseline update')
parser.add_argument('--controller_start_training', type=int, default=0, help='Epoch that the training of controller starts')
parser.add_argument('--scheduler', type=str, default='naive_cosine', help='type of LR scheduler')
parser.add_argument('--lr_min', type=float, default=0.01, help='min learning rate')
parser.add_argument('--store', type=int, default=1, help='Whether to store the model')
parser.add_argument('--edge_hid', type=int, default=100, help='edge hidden dimension')
# parser.add_argument('--transformer_learning_rate', type=float, default=3e-4, help='learning rate for pruner')
parser.add_argument('--transformer_weight_decay', type=float, default=5e-4, help='learning rate for pruner')
parser.add_argument('--transformer_nfeat', type=int, default=1024, help='feature dimension of each node')
parser.add_argument('--transformer_nhid', type=int, default=100, help='hidden dimension')
parser.add_argument('--thre', type=int, default=8, help='hidden dimension')
parser.add_argument('--transformer_dropout', type=float, default=0, help='dropout rate for transformer')
parser.add_argument('--transformer_normalize', action='store_true', default=False, help='use normalize in GCN')
parser.add_argument('--num_nodes', type=int, default=4, help='number of intermediate nodes')
parser.add_argument('--n_archs', type=int, default=10, help='number of intermediate nodes')
parser.add_argument('--op_type', type=str, default='LOOSE_END_PRIMITIVES', help='LOOSE_END_PRIMITIVES | FULLY_CONCAT_PRIMITIVES')
parser.add_argument('--pwp', type=str, default='predictor/finetune/May 14 10 52 44/predictor_state_dict_30.pt', help='The path to pretrained weight if there is(dir)')
parser.add_argument('--rt', type=str, default='a', help='reward_type')
parser.add_argument('--arch', type=str, default='ResBlock', help='reward_type')
parser.add_argument('--debug', action='store_true', default=False, help=' ')
parser.add_argument('--only_mlp', action='store_true', default=False, help=' ')
parser.add_argument('--pt_embedder', action='store_true', default=True, help=' ')
parser.add_argument('--vpi', action='store_true', default=True, help=' ')
parser.add_argument('--opdiv', action='store_true', default=False, help=' ')
parser.add_argument('--flps', action='store_true', default=False, help=' ')
parser.add_argument('--nlit', action='store_true', default=True, help=' ')
parser.add_argument('--imitation', action='store_true', default=False, help=' ')
parser.add_argument('--ss', type=str, default='null', help='reward_type')
args = parser.parse_args()

utils.parse_args_(args)

if args.op_type=='LOOSE_END_PRIMITIVES':
    args.loose_end = True
else:
    args.loose_end = False

if not os.path.exists(args.prefix):
    os.makedirs(args.prefix)
tmp = 'train_search/train_search_derive'
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

    model.re_initialize_arch_transformer()
    model.imitation = args.imitation
    model.single = False
    model.vpi = args.vpi
    model.ss = args.ss
    model._initialize_predictor(args, 'WarmUp')
    if args.pwp != ' ':
        model.predictor.load_state_dict(torch.load(args.pwp, map_location = 'cpu'))

    if args.pt_embedder:
        model.arch_transformer.arch_embedder.load_state_dict(model.predictor.arch_embedder.state_dict())

    transformer = Transformer(model, args)

    model.to(device)

    assert (args.rt == 'a' or args.rt == 'r')

    model.reward_type = 'absolute' if args.rt == 'a' else 'relative'

    # print(model.arch_transformer.arch_embedder.gcns)
    model.flops_limit = args.flps
    model.op_diversity = args.opdiv
    model.num_limit = args.nlit

    model.set_thre(args)
    model.use_arch= True
    genotype = eval("genotypes.%s" % args.arch)
    arch_normal, arch_reduce = utils.genotype_to_arch(genotype, args.op_type)
    model.set_arch(arch_normal, arch_reduce)
    convergence = []


    for iteration in range(1000):

        
        
        reward, ent, loss, flops_limit, op_div, nlit = transformer.step()
        convergence.append(reward)
        # print(convergence)
        summaryWriter.add_scalar('reward', reward, iteration)
        summaryWriter.add_scalar('loss', loss, iteration)
        summaryWriter.add_scalar('flops_limit', flops_limit, iteration)
        summaryWriter.add_scalar('op_div', op_div, iteration)
        summaryWriter.add_scalar('nlit', nlit, iteration)
        if iteration % 100 == 0:
            logging.info('iteration %d lr %e', iteration, transformer.current_lr)
            logging.info('Updating Theta iteration=%d reward=%.2f, ent = %.2f,loss=%.3f', iteration, reward, ent, loss)
        if iteration % args.save_freq == 0:
            utils.save(model.arch_transformer, os.path.join(args.save, 'model_{}.pt'.format(iteration)))
            torch.save(model.best_pair, os.path.join(args.save, "best_transform_pair"))
        # transformer.scheduler.step()
        if len(convergence) > 100:
            flag = 1
            for i in range(1, 11):
                if convergence[-i] != convergence[-i + 1]:
                    flag = 0
            if flag:
                break   


    if args.store == 1:
        utils.save(model.arch_transformer, os.path.join(args.save, 'model_final.pt'))
        torch.save(model.best_pair, os.path.join(args.save, "best_transform_pair"))
    
    
    
    result = model.derive_optimized_arch(arch_normal, arch_reduce, args.n_archs, logger, args.save, "derive", normal_concat=genotype.normal_concat, reduce_concat=genotype.reduce_concat)

    # torch.save(result, os.path.join(args.save, 'result_dict'))
    train_info = (0, 0, result["absolute_predictor_reward"], result["target_arch"], result["surrogate_arch"])
    torch.save(result, os.path.join(args.save, 'result_dict_{}'.format(args.arch)))
    torch.save(train_info, os.path.join(args.save, 'train_info_{}'.format(args.arch)))


if __name__ == '__main__':
    main()