import os
import shutil
import sys
STEM_WORK_DIR = os.path.join(os.getcwd(), "../..")
sys.path.append(STEM_WORK_DIR)
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
import random
from final_test.compile_based.models import NetworkCIFAR
from basic_parts.basic_nat_models import ArchMaster
from genotypes import LOOSE_END_PRIMITIVES
from collections import OrderedDict
'''
This file trains supernet and twin supernet for 50 epochs and save them
You can bypass the first 50 epochs of training by loading pretrained models
'''
localtime = time.asctime( time.localtime(time.time()))
x = re.split(r"[\s,(:)]",localtime)
default_EXP = " ".join(x[1:-1])
parser = argparse.ArgumentParser("DeepGuiser")
parser.add_argument('--data', type=str, default=os.path.join(STEM_WORK_DIR, '../data'), help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', "-lr", type=float, default=0.05, help='init learning rate')
parser.add_argument('--learning_rate_min', "-lr_min", type=float, default=0.01, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='number of training epochs')
parser.add_argument('--init_channels', type=int, default=20, help='number of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--save', type=str, default=default_EXP, help='experiment name')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--max_num', "-num", type=int, default=10000, help=' ')
parser.add_argument('--prefix', type=str, default='.', help='parent save path')
parser.add_argument('--scheduler', type=str, default='naive_cosine', help='type of LR scheduler')
parser.add_argument('--store', type=int, default=1, help='Whether to store the model')
parser.add_argument('--num_nodes', type=int, default=4, help='number of intermediate nodes')
parser.add_argument('--op_type', type=str, default='LOOSE_END_PRIMITIVES', help='LOOSE_END_PRIMITIVES | FULLY_CONCAT_PRIMITIVES')
parser.add_argument('--debug', action='store_true', default=False, help=' ')
parser.add_argument('--attack_info', type=str, default=os.path.join(STEM_WORK_DIR, 'final_test/attack/pgd1.yaml'), help='yaml file contains information about attack')# 
args = parser.parse_args()

args.seed = int(str(time.time()).split('.')[-1])
if args.op_type=='LOOSE_END_PRIMITIVES':
    args.loose_end = True
else:
    args.loose_end = False

args.cutout = False
utils.preprocess_exp_dir(args)
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
    logging.info('seed = %d', args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    logging.info("args = %s" % args)

    attack_info = utils.load_yaml(args.attack_info)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    train_queue, valid_queue, _ = utils.get_cifar_data_queue(args)

    train_dataset = []

    num_ops = len(LOOSE_END_PRIMITIVES)

    arch_master = ArchMaster(num_ops, args.num_nodes, device)
    arch_master.use_demo()
    arch_master.to(device)


    for its in range(args.max_num):

        sub_save_path = os.path.join(args.save, "{}".format(args.max_num))

        os.mkdir(sub_save_path)

        target_arch = [[], []]
        target_arch[0] = arch_master.forward()
        target_arch[1] = arch_master.forward()
        target_genotype = utils.arch_to_genotype(target_arch[0], target_arch[1], args.num_nodes, args.op_type, [5], [5])

        target_model = NetworkCIFAR( \
         args.init_channels, CIFAR_CLASSES, args.layers, False, target_genotype
        )

        logging.info('Training target model')

        target_optimizer = torch.optim.SGD(
            target_model.model_parameters(),
            args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    target_optimizer, float(args.epochs), eta_min=args.learning_rate_min
                    )

        utils.train_model(target_model, train_queue, device, criterion, target_optimizer, scheduler, args.epochs, logger)

        utils.save_compiled_based(target_model, os.path.join(sub_save_path, "target_model.pt"))

        baseline_model = NetworkCIFAR( \
         args.init_channels, CIFAR_CLASSES, args.layers, False, target_genotype
        )

        baseline_optimizer = torch.optim.SGD(
            target_model.model_parameters(),
            args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    baseline_optimizer, float(args.epochs), eta_min=args.learning_rate_min
                    )

        utils.train_model(baseline_model, train_queue, device, criterion, baseline_optimizer, scheduler, args.epochs, logger)

        utils.save_compiled_based(baseline_model, os.path.join(sub_save_path, "baseline_model.pt"))

        for j in range(4):
            surrogate_arch = [[], []]
            surrogate_arch[0] = arch_master.forward()
            surrogate_arch[1] = arch_master.forward()
            surrogate_genotype = utils.arch_to_genotype(surrogate_arch[0], surrogate_arch[1], args.num_nodes, args.op_type, [5], [5])
            
            surrogate_model = NetworkCIFAR( 
         args.init_channels, CIFAR_CLASSES, args.layers, False, surrogate_genotype
        )
           
            logging.info('Training Surrogate Model')

            surrogate_optimizer = torch.optim.SGD(
                surrogate_model.model_parameters(),
                args.learning_rate,
                momentum=args.momentum,
                weight_decay=args.weight_decay
            )

            if args.scheduler == "naive_cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                surrogate_optimizer, float(args.epochs), eta_min=args.learning_rate_min
                )
            else:
                assert False, "unsupported scheduler type: %s" % args.scheduler
            
            utils.train_model(surrogate_model, train_queue, device, criterion, surrogate_optimizer, scheduler, args.epochs, logger)

            utils.save_compiled_based(surrogate_model, os.path.join(sub_save_path, "surrogate_model.pt"))

            logging.info('Training Completed, Testing')

            acc_clean_target = utils.test_clean_accuracy(target_model, valid_queue, logger, device)[0]
            acc_clean_baseline = utils.test_clean_accuracy(baseline_model, valid_queue, logger, device)[0]
            acc_clean_surrogate = utils.test_clean_accuracy(surrogate_model, valid_queue, logger, device)[0]

            acc_adv_baseline, acc_adv_surrogate = utils.compiled_pgd_test(target_model, surrogate_model, baseline_model,  valid_queue, attack_info, logger = None)
            

            data_point = OrderedDict()
            
            data_point["target_genotype"] = "{}".format(utils.arch_to_genotype(target_arch[0], target_arch[1], args.num_nodes, args.op_type, [5], [5]))
            data_point["surrogate_genotype"] = "{}".format(utils.arch_to_genotype(surrogate_arch[0], surrogate_arch[1], args.num_nodes, args.op_type, [5], [5]))
            data_point["clean_accuracy"] = {"target": acc_clean_target, "baseline": acc_clean_baseline, "surrogate":acc_clean_surrogate}
            data_point["adversarial_accuracy"] = {"baseline": acc_adv_baseline, "surrogate":acc_adv_surrogate}
            data_point["train_info"] = {"model_path": {"target":os.path.join(sub_save_path, "target_model.pt") , "surrogate": os.path.join(sub_save_path, "surrogate_model.pt"), "baseline": os.path.join(sub_save_path, "baseline_model.pt")}, "log": args.save}

            train_dataset.append(data_point)

            utils.save_yaml(train_dataset, os.path.join(args.save, "dataset.yaml"))


if __name__ == '__main__':
    main()






