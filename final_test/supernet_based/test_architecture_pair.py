import os
import sys
STEM_WORK_DIR = os.path.join(os.getcwd(), "..", "..")
sys.path.append(STEM_WORK_DIR)
import glob
import random
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
import utils
import genotypes
from final_test.supernet_based.models import LooseEndModel
'''
This files tests the transferbility isotonicity on supernets and trained-from-scratch models
'''

parser = argparse.ArgumentParser("DeepGuiser")
parser.add_argument('--data', type=str, default=os.path.join(STEM_WORK_DIR, '../data'), help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--gpu', type=int, default=4, help='gpu device id')#
parser.add_argument('--epochs', type=int, default=100, help='number of signle model training epochs')
# parser.add_argument('--init_channels', type=int, default=20, help='number of init channels')
# parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--save', type=str, default=utils.localtime_as_dirname(), help='experiment name')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--prefix', type=str, default='.', help='parent save path')
parser.add_argument('--scheduler', type=str, default='naive_cosine', help='type of LR scheduler')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--arch_info', type=str, default='example.yaml', help='yaml file contains information about archs be tested')#
parser.add_argument('--pretrained_weight', '-pw', type=str, default=' ', help='pretrained weight file dir')#
parser.add_argument('--attack_info', type=str, default=os.path.join(STEM_WORK_DIR, 'final_test/attack/pgd1.yaml'), help='yaml file contains information about attack')#
parser.add_argument('--cifar_classes', type=int, default=10, help='hidden dimension')
parser.add_argument("--debug", action="store_true", default=False, help="debug mode")
args = parser.parse_args()

args.report_freq = 50
utils.preprocess_exp_dir(args)
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py')+glob.glob('*.sh')+glob.glob('*.yml'))

logger = utils.initialize_logger(args)
CIFAR_CLASSES = args.cifar_classes

args.cutout = False

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

    if args.pretrained_weight != " ":
        if not os.path.exists(args.pretrained_weight):
            assert 0
        else:
            bypass_training = True
            load_pretrained_weight = True
        


    train_queue, test_queue = utils.get_final_test_data(args, CIFAR_CLASSES)

    assert os.path.exists(args.arch_info)

    arch_info = utils.load_yaml(args.arch_info)

    target_genotype = eval(arch_info['target'][0]['genotype'])
    surrogate_genotpye = eval(arch_info['surrogate'][0]['genotype'])

    target_model = LooseEndModel(device, CIFAR_CLASSES, target_genotype)

    assert 0

    # if load_pretrained_weight:
    #     utils.load_supernet()

    target_model.to(device)

    logging.info('training Target Model')

    if not bypass_training:

        target_optimizer = torch.optim.SGD(
            target_model.model_parameters(),
            args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )

        target_model.to(device)

        if args.scheduler == "naive_cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            target_optimizer, float(args.epochs), eta_min=args.learning_rate_min
            )
        else:
            assert False, "unsupported scheduler type: %s" % args.scheduler

        utils.train_model(target_model, train_queue, device, criterion, target_optimizer, scheduler, args.epochs, logger)

    utils.save_supernet_based(target_model, os.path.join(args.save, 'target_model.pt'))
    acc_clean_target, _ = utils.test_clean_accuracy(target_model, test_queue, logger)

    logging.info('training Surrogate Model')

    surrogate_model = LooseEndModel(device, CIFAR_CLASSES, surrogate_genotpye)
    surrogate_optimizer = torch.optim.SGD(
        surrogate_model.model_parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    surrogate_model.to(device)

    if args.scheduler == "naive_cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        surrogate_optimizer, float(args.epochs), eta_min=args.learning_rate_min
        )
    else:
        assert False, "unsupported scheduler type: %s" % args.scheduler

    utils.train_model(surrogate_model, train_queue, device, criterion, surrogate_optimizer, scheduler, args.epochs, logger)

    utils.save_supernet_based(surrogate_model, os.path.join(args.save, 'surrogate_model.pt'))
    acc_clean_surrogate, _ = utils.test_clean_accuracy(surrogate_model, test_queue, logger)


    logging.info('training Target Baseline Model')

    baseline_model = LooseEndModel(device, CIFAR_CLASSES, target_genotype)

    target_baseline_optimizer = torch.optim.SGD(
        baseline_model.model_parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    baseline_model.to(device)

    if args.scheduler == "naive_cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        target_baseline_optimizer, float(args.epochs), eta_min=args.learning_rate_min
        )
    else:
        assert False, "unsupported scheduler type: %s" % args.scheduler

    utils.train_model(baseline_model, train_queue, device, criterion, target_baseline_optimizer, scheduler, args.epochs, logger)

    utils.save_supernet_based(baseline_model, os.path.join(args.save, 'baseline_model.pt'))
    acc_clean_baseline, _ = utils.test_clean_accuracy(baseline_model, test_queue, logger)

    logging.info('training completed')

    logging.info('testing adversarial attack transferbitlity')

    attack_info = utils.load_yaml(args.attack_info)

    acc_adv_baseline, acc_adv_surrogate = utils.compiled_pgd_test(target_model, surrogate_model, baseline_model, test_queue, attack_info, logger)
    save_dict = {}
    save_dict['target_genotype'] = '{}'.format(target_genotype)
    save_dict['surrogate_arch'] = '{}'.format(surrogate_genotpye)
    save_dict['acc_clean'] = {'target': acc_clean_target, 'surrogate': acc_clean_surrogate, 'baseline': acc_clean_baseline}
    save_dict['acc_adv'] = {'surrogate': acc_adv_surrogate, 'baseline': acc_adv_baseline}
    utils.save_yaml(save_dict, os.path.join(args.save, 'result.yaml'))

    utils.draw_clean(target_genotype, args.save, 'target')
    utils.draw_clean(surrogate_genotpye, args.save, 'surrogate')


if __name__ == '__main__':
    main()






