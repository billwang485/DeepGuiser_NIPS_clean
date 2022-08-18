dirp = ['z/transformer_random_test/May 18 07 41 07',
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
import torch.backends.cudnn as cudnn
import time
import re
import torchvision.datasets as dset
from search_model_twin import NASNetwork as Network
from single_model import FinalNetwork as FinalNet
import random
from copy import deepcopy
from scipy.stats import kendalltau
import shutil
from utils import arch_to_genotype, draw_genotype
from PyPDF2 import PdfFileMerger
from torch import tensor
localtime = time.asctime( time.localtime(time.time()))
x = re.split(r"[\s,(:)]",localtime)
default_EXP = " ".join(x[1:-1])
parser = argparse.ArgumentParser("DeepGuiser")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.05, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=int, default=10, help='report frequency')
parser.add_argument('--test_freq', type=int, default=10, help='test frequency')
parser.add_argument('--gpu', type=int, default=5, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='number of signle model training epochs')
parser.add_argument('--init_channels', type=int, default=20, help='number of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--save', type=str, default=default_EXP, help='experiment name')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--cc', type=int, default=10, help='random seed')
parser.add_argument('--train_portion', type=float, default=0.8, help='data portion for training weights')
parser.add_argument('--prefix', type=str, default='.', help='parent save path')
parser.add_argument('--gamma', type=float, default=0.99, help='time decay for baseline update')
parser.add_argument('--scheduler', type=str, default='naive_cosine', help='type of LR scheduler')
parser.add_argument('--learning_rate_min', type=float, default=0.01, help='min learning rate')
parser.add_argument('--store', type=int, default=1, help='Whether to store the model')
parser.add_argument('--edge_hid', type=int, default=100, help='edge hidden dimension')
parser.add_argument('--num_nodes', type=int, default=4, help='number of intermediate nodes')
parser.add_argument('--op_type', type=str, default='LOOSE_END_PRIMITIVES', help='LOOSE_END_PRIMITIVES | FULLY_CONCAT_PRIMITIVES')
parser.add_argument('--transfer', action='store_true', default=True, help='eval the transferability')
# parser.add_argument('--pwt', type=str, default=' ', help='The path to pretrained weight')
# parser.add_argument('--pwtb', type=str, default=' ', help='The path to pretrained weight')
parser.add_argument('--arch', type=str, default=' ', help='The path to pretrained weight')
parser.add_argument('--controller_hid', type=int, default=100, help='controller hidden dimension')
parser.add_argument('--accu_batch', type=int, default=10, help='controller hidden dimension')
parser.add_argument('--transformer_learning_rate', type=float, default=3e-4, help='learning rate for pruner')
parser.add_argument('--transformer_weight_decay', type=float, default=5e-4, help='learning rate for pruner')
parser.add_argument('--transformer_nfeat', type=int, default=1024, help='feature dimension of each node')
parser.add_argument('--transformer_nhid', type=int, default=100, help='hidden dimension')
parser.add_argument('--transformer_dropout', type=float, default=0, help='dropout rate for transformer')
parser.add_argument('--transformer_normalize', action='store_true', default=False, help='use normalize in GCN')
parser.add_argument('--entropy_coeff', nargs='+', type=float, default=[0.003, 0.003], help='coefficient for entropy: [normal, reduce]')
args = parser.parse_args()
args.batch_size = 1
assert args.batch_size == 1

# args.pwt = 'z/test_one_transform_model/May 10 14 21 51/target.pt'
# args.pws = 'z/test_one_transform_model/May 10 14 21 51/surrogate_model.pt'

if args.op_type=='LOOSE_END_PRIMITIVES':
    args.loose_end = True
else:
    args.loose_end = False

if not os.path.exists(args.prefix):
    os.makedirs(args.prefix)

args.save = os.path.join(args.prefix, 'z/visualization_20_random', args.save)
args.cutout = False
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py')+glob.glob('*.sh')+glob.glob('*.yml'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logger = logging.getLogger()
CIFAR_CLASSES = 10

dirs = []
for mypath in os.listdir('z/0519'):
    dirs.append(os.path.join('z/0519', mypath))
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

    train_transform, valid_transform = utils._data_transforms_cifar10(args)

    # train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    if CIFAR_CLASSES == 10:
        test_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)
    elif CIFAR_CLASSES == 100:
        test_data = dset.CIFAR100(root=args.data, train=False, download=True, transform=valid_transform)
    else:
        train_transform, valid_transform = utils._data_transforms_cifar10(args)
        test_data = dset.ImageFolder(root=os.path.join(args.data, 'tiny-imagenet-200', 'val'), transform=valid_transform)


    # num_train = len(train_data)
    # indices = list(range(num_train))
    indices_test = list(range(len(test_data)))
    # random.shuffle(indices)
    random.shuffle(indices_test)

    test_queue = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SequentialSampler(indices_test),
        pin_memory=True, num_workers=2
    )
    result0 = []
    flops0 = []
    param0 = []

    result1 = []
    flops1 = []
    param1 = []
    result2 = []
    flops2 = []
    param2 = []
    result3 = []
    flops3 = []
    param3 = []

    for i in range(0, 20):

        os.mkdir(os.path.join(args.save, str(i)))


        tmp = torch.load(os.path.join(dirp[i], 'surrogate_predictor_result'), map_location= 'cpu')
        tmp1 = torch.load(os.path.join(dirs[i], 'surrogate_gates_result'), map_location= 'cpu')
        tmp2 = torch.load(os.path.join(dirp[i], 'random_surrogate_0_result'), map_location= 'cpu')


        target_model=FinalNet(
                args.init_channels, CIFAR_CLASSES, args.layers, criterion, device, steps=args.num_nodes, controller_hid=args.controller_hid, entropy_coeff=args.entropy_coeff, edge_hid = args.edge_hid, transformer_nfeat = args.transformer_nfeat, transformer_nhid = args.transformer_nhid, transformer_dropout = args.transformer_dropout, transformer_normalize = args.transformer_normalize, loose_end = args.loose_end, op_type=args.op_type
            )
        # utils.load(target_model, args.pwt)
        target_model.arch_normal = tmp['target_arch'][0]
        target_model.arch_reduce = tmp['target_arch'][1]
        
        target_model.single = True
        # utils.save(target_model, os.path.join(args.save, 'target.pt')) 

        surrogate_model_predictor=FinalNet(
                args.init_channels, CIFAR_CLASSES, args.layers, criterion, device, steps=args.num_nodes, controller_hid=args.controller_hid, entropy_coeff=args.entropy_coeff, edge_hid = args.edge_hid, transformer_nfeat = args.transformer_nfeat, transformer_nhid = args.transformer_nhid, transformer_dropout = args.transformer_dropout, transformer_normalize = args.transformer_normalize, loose_end = args.loose_end, op_type=args.op_type
            )

        # utils.load(surrogate_model, args.pws)
        surrogate_model_predictor.arch_normal = tmp['surrogate_arch'][0]
        surrogate_model_predictor.arch_reduce = tmp['surrogate_arch'][1]
        
        surrogate_model_predictor.single = True

        surrogate_model_supernet=FinalNet(
                args.init_channels, CIFAR_CLASSES, args.layers, criterion, device, steps=args.num_nodes, controller_hid=args.controller_hid, entropy_coeff=args.entropy_coeff, edge_hid = args.edge_hid, transformer_nfeat = args.transformer_nfeat, transformer_nhid = args.transformer_nhid, transformer_dropout = args.transformer_dropout, transformer_normalize = args.transformer_normalize, loose_end = args.loose_end, op_type=args.op_type
            )

        # utils.load(surrogate_model, args.pws)
        surrogate_model_supernet.arch_normal = tmp1['surrogate_arch'][0]
        surrogate_model_supernet.arch_reduce = tmp1['surrogate_arch'][1]
        
        surrogate_model_supernet.single = True

        surrogate_model_random=FinalNet(
                args.init_channels, CIFAR_CLASSES, args.layers, criterion, device, steps=args.num_nodes, controller_hid=args.controller_hid, entropy_coeff=args.entropy_coeff, edge_hid = args.edge_hid, transformer_nfeat = args.transformer_nfeat, transformer_nhid = args.transformer_nhid, transformer_dropout = args.transformer_dropout, transformer_normalize = args.transformer_normalize, loose_end = args.loose_end, op_type=args.op_type
            )
        
        surrogate_model_random.arch_normal = tmp2['surrogate_arch'][0]
        surrogate_model_random.arch_reduce = tmp2['surrogate_arch'][1]
        
        surrogate_model_random.single = True

        # utils.save(surrogate_model_predictor, os.path.join(args.save, 'surrogate_model.pt')) 
        target_model.eval()
        surrogate_model_predictor.eval()
        surrogate_model_supernet.eval()
        surrogate_model_random.eval()


        genotype = arch_to_genotype(target_model.arch_normal, target_model.arch_reduce, 4, 'LOOSE_END_PRIMITIVES', [5], [5])
        predictor_genotype = arch_to_genotype(surrogate_model_predictor.arch_normal, surrogate_model_predictor.arch_reduce, 4, 'LOOSE_END_PRIMITIVES', [5], [5])
        supernet_genotype = arch_to_genotype(surrogate_model_supernet.arch_normal, surrogate_model_supernet.arch_reduce, 4, 'LOOSE_END_PRIMITIVES', [5], [5])
        random_genotype = arch_to_genotype(surrogate_model_random.arch_normal, surrogate_model_random.arch_reduce, 4, 'LOOSE_END_PRIMITIVES', [5], [5])
       
        draw_genotype(genotype.normal, 4, os.path.join(args.save, str(i), "normal_target"), genotype.normal_concat)
        draw_genotype(genotype.reduce, 4, os.path.join(args.save, str(i), "reduce_target"), genotype.reduce_concat)
        file_merger = PdfFileMerger()

        file_merger.append(os.path.join(args.save, str(i), "normal_target.pdf"))
        file_merger.append(os.path.join(args.save, str(i), "reduce_target.pdf"))

        file_merger.write(os.path.join(args.save, str(i), "target.pdf"))

        os.remove( os.path.join(args.save, str(i), "normal_target"))
        os.remove( os.path.join(args.save, str(i), "reduce_target"))

        draw_genotype(predictor_genotype.normal, 4, os.path.join(args.save, str(i), "normal_predictor"), predictor_genotype.normal_concat)
        draw_genotype(predictor_genotype.reduce, 4, os.path.join(args.save, str(i), "reduce_predictor"), predictor_genotype.reduce_concat)
        file_merger = PdfFileMerger()

        file_merger.append(os.path.join(args.save, str(i), "normal_predictor.pdf"))
        file_merger.append(os.path.join(args.save, str(i), "reduce_predictor.pdf"))

        file_merger.write(os.path.join(args.save, str(i), "disguised_predictor.pdf"))

        os.remove( os.path.join(args.save, str(i), "normal_predictor"))
        os.remove( os.path.join(args.save, str(i), "reduce_predictor"))

        draw_genotype(supernet_genotype.normal, 4, os.path.join(args.save, str(i), "normal_supernet"), supernet_genotype.normal_concat)
        draw_genotype(supernet_genotype.reduce, 4, os.path.join(args.save, str(i), "reduce_supernet"), supernet_genotype.reduce_concat)
        file_merger = PdfFileMerger()

        file_merger.append(os.path.join(args.save, str(i), "normal_supernet.pdf"))
        file_merger.append(os.path.join(args.save, str(i), "reduce_supernet.pdf"))

        file_merger.write(os.path.join(args.save, str(i), "disguised_supernet.pdf"))

        os.remove( os.path.join(args.save, str(i), "normal_supernet"))
        os.remove( os.path.join(args.save, str(i), "reduce_supernet"))

        draw_genotype(random_genotype.normal, 4, os.path.join(args.save, str(i), "normal_random"), random_genotype.normal_concat)
        draw_genotype(random_genotype.reduce, 4, os.path.join(args.save, str(i), "reduce_random"), random_genotype.reduce_concat)
        file_merger = PdfFileMerger()

        file_merger.append(os.path.join(args.save, str(i), "normal_random.pdf"))
        file_merger.append(os.path.join(args.save, str(i), "reduce_random.pdf"))

        file_merger.write(os.path.join(args.save, str(i), "disguised_random.pdf"))

        os.remove( os.path.join(args.save, str(i), "normal_random"))
        os.remove( os.path.join(args.save, str(i), "reduce_random"))

                    
        str1 = '\n'
        f=open(os.path.join(args.save, str(i),"supernet_result.txt"),"w")
        f.write('acc_clean_target: {}'.format(float(tmp1['target_acc_clean'])))
        f.write(str1)
        f.write('acc_clean_surrogate: {}'.format(float(tmp1['surrogate_acc_clean'])))
        f.write(str1)
        f.write('acc_adv_surrogate: {}'.format(float(tmp1['adv_acc_baseline']) + tmp1['reward']))
        f.write(str1)
        f.write('acc_adv_baseline: {}'.format(float(tmp1['adv_acc_baseline'])))
        f.write(str1)
        f.close()

        str1 = '\n'
        f=open(os.path.join(args.save, str(i),"predictor_result.txt"),"w")
        f.write('acc_clean_target: {}'.format(float(tmp['target_acc_clean'])))
        f.write(str1)
        f.write('acc_clean_surrogate: {}'.format(float(tmp['surrogate_acc_clean'])))
        f.write(str1)
        f.write('acc_adv_surrogate: {}'.format(float(tmp['adv_acc_baseline']) + tmp['reward']))
        f.write(str1)
        f.write('acc_adv_baseline: {}'.format(float(tmp['adv_acc_baseline'])))
        f.write(str1)
        f.close()

        str1 = '\n'
        f=open(os.path.join(args.save, str(i),"random_result.txt"),"w")
        f.write('acc_clean_target: {}'.format(float(tmp2['target_acc_clean'])))
        f.write(str1)
        f.write('acc_clean_surrogate: {}'.format(float(tmp2['surrogate_acc_clean'])))
        f.write(str1)
        f.write('acc_adv_surrogate: {}'.format(float(tmp2['adv_acc_baseline']) + tmp2['reward']))
        f.write(str1)
        f.write('acc_adv_baseline: {}'.format(float(tmp2['adv_acc_baseline'])))
        f.write(str1)
        f.close()



    #     for step, (input, target) in enumerate(test_queue):
    #         if step >= 1:
    #             break
    #         size_ = input.shape
    #     surrogate_model_supernet.to(device)
    #     surrogate_model_predictor.to(device)
    #     surrogate_model_random.to(device)
    #     target_model.to(device)

    #     # target_flops = utils.compute_flops(target_model, size_,target_model.arch_normal ,target_model.arch_reduce, 'null')
    #     # surrogate_flops = utils.compute_flops(surrogate_model, size_,surrogate_model.arch_normal, surrogate_model.arch_reduce, 'null')
        

    #     # target_model.test_acc_single()
    #     start_time = time.process_time()
    #     for step, (input, target) in enumerate(test_queue):
    #         if step >= 1000:
    #             break
    #         input = input.to(device)
    #         target = target.to(device)
    #         logits = target_model._inner_forward(input, target_model.arch_normal, target_model.arch_reduce)
    #     end_time = time.process_time()
    #     run_time1 = end_time - start_time

    #     start_time = time.process_time()
    #     for step, (input, target) in enumerate(test_queue):
    #         if step >= 1000:
    #             break
    #         input = input.to(device)
    #         target = target.to(device)
    #         # logits = target_model._inner_forward(input, target_model.arch_normal, target_model.arch_reduce)
    #     end_time = time.process_time()
    #     run_time2 = end_time - start_time

    #     target_run_time = (run_time1 - run_time2) / 1000

    #     start_time = time.process_time()
    #     for step, (input, target) in enumerate(test_queue):
    #         if step >= 1000:
    #             break
    #         input = input.to(device)
    #         target = target.to(device)
    #         logits = surrogate_model_predictor._inner_forward(input, surrogate_model_predictor.arch_normal, surrogate_model_predictor.arch_reduce)
    #     end_time = time.process_time()
    #     run_time1 = end_time - start_time

    #     start_time = time.process_time()
    #     for step, (input, target) in enumerate(test_queue):
    #         if step >= 1000:
    #             break
    #         input = input.to(device)
    #         target = target.to(device)
    #         # logits = target_model._inner_forward(input, target_model.arch_normal, target_model.arch_reduce)
    #     end_time = time.process_time()
    #     run_time2 = end_time - start_time

    #     surrogate_predictor_run_time = (run_time1 - run_time2) / 1000

    #     start_time = time.process_time()
    #     for step, (input, target) in enumerate(test_queue):
    #         if step >= 1000:
    #             break
    #         input = input.to(device)
    #         target = target.to(device)
    #         logits = surrogate_model_supernet._inner_forward(input, surrogate_model_supernet.arch_normal, surrogate_model_supernet.arch_reduce)
    #     end_time = time.process_time()
    #     run_time1 = end_time - start_time

    #     start_time = time.process_time()
    #     for step, (input, target) in enumerate(test_queue):
    #         if step >= 1000:
    #             break
    #         input = input.to(device)
    #         target = target.to(device)
    #         # logits = target_model._inner_forward(input, target_model.arch_normal, target_model.arch_reduce)
    #     end_time = time.process_time()
    #     run_time2 = end_time - start_time

    #     surrogate_supernet_run_time = (run_time1 - run_time2) / 1000

    #     start_time = time.process_time()
    #     for step, (input, target) in enumerate(test_queue):
    #         if step >= 1000:
    #             break
    #         input = input.to(device)
    #         target = target.to(device)
    #         logits = surrogate_model_random._inner_forward(input, surrogate_model_random.arch_normal, surrogate_model_random.arch_reduce)
    #     end_time = time.process_time()
    #     run_time1 = end_time - start_time

    #     start_time = time.process_time()
    #     for step, (input, target) in enumerate(test_queue):
    #         if step >= 1000:
    #             break
    #         input = input.to(device)
    #         target = target.to(device)
    #         # logits = target_model._inner_forward(input, target_model.arch_normal, target_model.arch_reduce)
    #     end_time = time.process_time()
    #     run_time2 = end_time - start_time

    #     surrogate_random_run_time = (run_time1 - run_time2) / 1000


    #     logging.info('target model {}: latency {} '.format(i, target_run_time))
    #     result1.append(str(target_run_time))
    #     logging.info('surrogate predictor model {}: latency {} '.format(i, surrogate_predictor_run_time))
    #     result2.append(str(surrogate_predictor_run_time))
    #     logging.info('surrogate supernet model {}: latency {} '.format(i, surrogate_supernet_run_time))
    #     result3.append(str(surrogate_supernet_run_time))
    #     logging.info('surrogate random model {}: latency {} '.format(i, surrogate_random_run_time))
    #     result0.append(str(surrogate_random_run_time))

    #     size_ = [1, 3, 32, 32]

    #     target_flops = utils.compute_flops(target_model, size_,target_model.arch_normal ,target_model.arch_reduce, 'null')
    #     surrogate_predictor_flops = utils.compute_flops(surrogate_model_predictor, size_,surrogate_model_predictor.arch_normal, surrogate_model_predictor.arch_reduce, 'null')
    #     surrogate_supernet_flops = utils.compute_flops(surrogate_model_supernet, size_,surrogate_model_supernet.arch_normal, surrogate_model_supernet.arch_reduce, 'null')
    #     surrogate_model_flops = utils.compute_flops(surrogate_model_random, size_,surrogate_model_random.arch_normal, surrogate_model_random.arch_reduce, 'null')

    #     logging.info('target model {}: flops {} '.format(i, target_flops))
    #     flops1.append(str(target_flops))
    #     logging.info('surrogate predictor model {}: flops {} '.format(i, surrogate_predictor_flops))
    #     flops2.append(str(surrogate_predictor_flops))
    #     logging.info('surrogate supernet model {}: flops {} '.format(i, surrogate_supernet_flops))
    #     flops3.append(str(surrogate_supernet_flops))
    #     logging.info('surrogate random model {}: flops {} '.format(i, surrogate_model_flops))
    #     flops0.append(str(surrogate_model_flops))

    #     target_param = utils.compute_nparam(target_model, size_,target_model.arch_normal ,target_model.arch_reduce, 'null')
    #     surrogate_predictor_param = utils.compute_nparam(surrogate_model_predictor, size_,surrogate_model_predictor.arch_normal, surrogate_model_predictor.arch_reduce, 'null')
    #     surrogate_supernet_param = utils.compute_nparam(surrogate_model_supernet, size_,surrogate_model_supernet.arch_normal, surrogate_model_supernet.arch_reduce, 'null')
    #     surrogate_random_param = utils.compute_nparam(surrogate_model_random, size_,surrogate_model_random.arch_normal, surrogate_model_random.arch_reduce, 'null')

    #     logging.info('target model {}: flops {} '.format(i, target_param))
    #     param1.append(str(target_param))
    #     logging.info('surrogate predictor model {}: flops {} '.format(i, surrogate_predictor_param))
    #     param2.append(str(surrogate_predictor_param))
    #     logging.info('surrogate supernet model {}: flops {} '.format(i, surrogate_supernet_param))
    #     param3.append(str(surrogate_supernet_param))
    #     logging.info('surrogate supernet model {}: flops {} '.format(i, surrogate_random_param))
    #     param0.append(str(surrogate_random_param))
        
    # str1 = '\n'
    # f=open(os.path.join(args.save, "zy_target_latency.txt"),"w")
    # f.write(str1.join(result1))
    # f.close()
    # f=open(os.path.join(args.save, "zy_predictor_latency.txt"),"w")
    # f.write(str1.join(result2))
    # f.close()
    # f=open(os.path.join(args.save, "zy_supernet_latency.txt"),"w")
    # f.write(str1.join(result3))
    # f.close()
    # f=open(os.path.join(args.save, "zy_random_latency.txt"),"w")
    # f.write(str1.join(result0))
    # f.close()
    # f=open(os.path.join(args.save, "zy_target_flops.txt"),"w")
    # f.write(str1.join(flops1))
    # f.close()
    # f=open(os.path.join(args.save, "zy_predictor_flops.txt"),"w")
    # f.write(str1.join(flops2))
    # f.close()
    # f=open(os.path.join(args.save, "zy_supernet_flops.txt"),"w")
    # f.write(str1.join(flops3))
    # f.close()
    # f=open(os.path.join(args.save, "zy_random_flops.txt"),"w")
    # f.write(str1.join(flops0))
    # f.close()
    # f=open(os.path.join(args.save, "zy_target_param.txt"),"w")
    # f.write(str1.join(param1))
    # f.close()
    # f=open(os.path.join(args.save, "zy_predictor_param.txt"),"w")
    # f.write(str1.join(param2))
    # f.close()
    # f=open(os.path.join(args.save, "zy_supernet_param.txt"),"w")
    # f.write(str1.join(param3))
    # f.close()
    # f=open(os.path.join(args.save, "zy_random_param.txt"),"w")
    # f.write(str1.join(param0))
    # f.close()
def mean(a):
    return sum(a) / len(a)

        


        # for step, (input, target) in enumerate(test_queue):
        #     # if step >= args.accu_batch:
        #     #     break
        #     n = input.size(0)
        #     input = input.to(device)
        #     target = target.to(device)
        #     target_model.eval()
        #     target_model_baseline.eval()
        #     surrogate_model.eval()
        #     input_adv, (acc_clean, acc_adv) = surrogate_model.generate_adv_input(input, target, 0.031)
        #     acc_clean, acc_adv = surrogate_model.eval_transfer(input_adv, input, target)
        #     surrogate_acc_clean.update(acc_clean.item(), n)
        #     surrogate_acc_adv.update(acc_adv.item(), n)
        #     input_adv_, (acc_clean, acc_adv) = target_model_baseline.generate_adv_input(input, target, 0.031)
        #     # logging.info("acc_adv_target_white=%.2f", acc_adv.item())
        #     (acc_clean, acc_adv_) = target_model.eval_transfer(input_adv_, input, target)
        #     target_acc_clean_baseline.update(acc_clean.item(), n)
        #     adv_acc_baseline.update(acc_adv_.item(), n)
        #     (acc_clean, acc_adv) = target_model.eval_transfer(input_adv, input, target)
        #     target_acc_clean.update(acc_clean.item(), n)
        #     target_acc_adv.update(acc_adv.item(), n)
        #     reward.update(acc_adv.item() - acc_adv_.item(), n)
        #     if step % args.report_freq == 0:
        #         # logging.info('common save to %s', args.common_save)
        #         logging.info('Step=%03d: surrogate_acc_clean=%.4f surrogate_acc_adv=%.4f target_acc_clean_baseline=%.4f adv_acc_baseline=%.4f target_acc_clean=%.4f target_acc_adv=%.4f reward=%.4f',\
        #             step, surrogate_acc_clean.avg, surrogate_acc_adv.avg, target_acc_clean_baseline.avg, adv_acc_baseline.avg, target_acc_clean.avg, target_acc_adv.avg, reward.avg )
        # logging.info('Final: surrogate_acc_clean=%.4f surrogate_acc_adv=%.4f target_acc_clean_baseline=%.4f adv_acc_baseline=%.4f target_acc_clean=%.4f target_acc_adv=%.4f reward=%.4f',\
        #             surrogate_acc_clean.avg, surrogate_acc_adv.avg, target_acc_clean_baseline.avg, adv_acc_baseline.avg, target_acc_clean.avg, target_acc_adv.avg, reward.avg )
        # logging.info('final_train_reward=%.4f', reward.avg)
        # save_dict = {}
        # save_dict['target_arch'] = (target_model.arch_normal, target_model.arch_reduce)
        # save_dict['surrogate_arch'] = (surrogate_model.arch_normal, surrogate_model.arch_reduce)
        # save_dict['target_latency'] = target_run_time
        # # save_dict['target_run_time'] = target_run_time
        # save_dict['surrogate_latency'] = surrogate_run_time
        # # save_dict['surrogate_run_time'] = surrogate_run_time
        # # save_dict['adv_acc_baseline'] = adv_acc_baseline.avg
        # # save_dict['reward'] = reward.avg
        # torch.save(save_dict, os.path.join(args.save, 'save_dict'))

        

        # genotype = arch_to_genotype(target_model.arch_normal, target_model.arch_reduce, target_model._steps, target_model.op_type, [5], [5])
        # transformed_genotype = arch_to_genotype(surrogate_model.arch_normal, surrogate_model.arch_reduce, target_model._steps, target_model.op_type, [5], [5])

        # draw_genotype(genotype.normal, target_model._steps, os.path.join(args.save, "normal_target"), genotype.normal_concat)
        # draw_genotype(genotype.reduce, target_model._steps, os.path.join(args.save, "reduce_target"), genotype.reduce_concat)
        # draw_genotype(transformed_genotype.normal, target_model._steps, os.path.join(args.save, "disguised_normal"), transformed_genotype.normal_concat)
        # draw_genotype(transformed_genotype.reduce, target_model._steps, os.path.join(args.save, "disguised_reduce"), transformed_genotype.reduce_concat)
        # file_merger = PdfFileMerger()

        # file_merger.append(os.path.join(args.save, "normal_target.pdf"))
        # file_merger.append(os.path.join(args.save, "reduce_target.pdf"))

        # file_merger.write(os.path.join(args.save, "target.pdf"))

        # file_merger = PdfFileMerger()

        # file_merger.append(os.path.join(args.save, "disguised_normal.pdf"))
        # file_merger.append(os.path.join(args.save, "disguised_reduce.pdf"))

        # file_merger.write(os.path.join(args.save, "disguised_target.pdf"))

        # os.remove(os.path.join(args.save, "normal_target.pdf"))
        # os.remove(os.path.join(args.save, "normal_target"))
        # os.remove(os.path.join(args.save, "reduce_target"))
        # os.remove(os.path.join(args.save, "reduce_target.pdf"))
        # os.remove(os.path.join(args.save, "disguised_normal.pdf"))
        # os.remove(os.path.join(args.save, "disguised_normal"))
        # os.remove(os.path.join(args.save, "disguised_reduce.pdf"))
        # os.remove(os.path.join(args.save, "disguised_reduce"))


if __name__ == '__main__':
    main()







