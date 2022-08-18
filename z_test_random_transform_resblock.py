import os
import shutil
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
import random
from single_model import FinalNetwork as FinalNet
from collections import OrderedDict
from genotypes import ResBlock
'''
This file trains supernet and twin supernet for 50 epochs and save them
You can bypass the first 50 epochs of training by loading pretrained models
'''
localtime = time.asctime( time.localtime(time.time()))
x = re.split(r"[\s,(:)]",localtime)
default_EXP = " ".join(x[1:-1])
parser = argparse.ArgumentParser("DeepGuiser")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--lr', type=float, default=0.1, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=int, default=50, help='report frequency')
parser.add_argument('--test_freq', type=int, default=10, help='test frequency')
parser.add_argument('--save_freq', type=int, default=10, help='test frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
parser.add_argument('--init_channels', type=int, default=20, help='number of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--save', type=str, default=default_EXP, help='experiment name')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--n_archs', type=int, default=1, help='number of candidate archs')
parser.add_argument('--num', type=int, default=10000, help=' ')
parser.add_argument('--prefix', type=str, default='.', help='parent save path')
parser.add_argument('--controller_hid', type=int, default=100, help='controller hidden dimension')
parser.add_argument('--entropy_coeff', nargs='+', type=float, default=[0.003, 0.003], help='coefficient for entropy: [normal, reduce]')
parser.add_argument('--gamma', type=float, default=0.99, help='time decay for baseline update')
parser.add_argument('--controller_start_training', type=int, default=0, help='Epoch that the training of controller starts')
parser.add_argument('--scheduler', type=str, default='naive_cosine', help='type of LR scheduler')
parser.add_argument('--lr_min', type=float, default=0.001, help='min learning rate')
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
parser.add_argument('--transfer', action='store_true', default=True, help='eval the transferability')
# parser.add_argument('--pw', type=str, default='LOOSE_END_supernet', help='The path to pretrained weight if there is(dir)')
parser.add_argument('--debug', action='store_true', default=False, help=' ')
parser.add_argument('--surrogate_only', action='store_true', default=False, help=' ')
parser.add_argument('--accu_batch', type=int, default=10, help=' ')
parser.add_argument('--rt', type=str, default='a', help='reward_type')  
parser.add_argument('--train_portion', type=float, default=0.8, help='data portion for training weights') 
args = parser.parse_args()

# args.seed = int(str(time.time()).split('.')[-1])
if args.op_type=='LOOSE_END_PRIMITIVES':
    args.loose_end = True
else:
    args.loose_end = False

if not os.path.exists(args.prefix):
    os.makedirs(args.prefix)


if args.debug:
    args.save = os.path.join(args.prefix, "z/random_transform_resblock/debug", args.save)
else:
    args.save = os.path.join(args.prefix, "z/random_transform_resblock", args.save)
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
    logging.info('seed = %d', args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    logging.info("args = %s" % args)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    # train_queue, valid_queue, test_queue = utils.get_train_queue(args)

    train_transform, valid_transform = utils._data_transforms_cifar10(args)

    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

    split = int(np.floor(0.8 * len(train_data)))

    test_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    indices_test = list(range(len(test_data)))
    # random.seed(seed)
    random.shuffle(indices)
    random.shuffle(indices_test)
    # random.seed(args.seed)

    valid_queue = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices_test),
        pin_memory=True, num_workers=2
    )

    train_queue = torch.utils.data.DataLoader(
            train_data, args.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices),
            # shuffle= False,
            pin_memory=True, num_workers=2
    )
    
    train_dataset = []

    for its in range(1):

        target_model = FinalNet( \
         args.init_channels, CIFAR_CLASSES, args.layers, criterion, device, steps=args.num_nodes, controller_hid=args.controller_hid, entropy_coeff=args.entropy_coeff, edge_hid = args.edge_hid, transformer_nfeat = args.transformer_nfeat, transformer_nhid = args.transformer_nhid, transformer_dropout = args.transformer_dropout, transformer_normalize = args.transformer_normalize, loose_end = args.loose_end, op_type=args.op_type\
        )

        

        logging.info('Training target model')

        target_optimizer = torch.optim.SGD(
            target_model.model_parameters(),
            args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )

        target_model.to(device)

        # target_model.arch_normal_master.demo = True
        # target_model.arch_reduce_master.demo = True


        target_model.arch_normal, target_model.arch_reduce = utils.genotype_to_arch(ResBlock, target_model.op_type)
        train_data_j = []

        for j in range(20):
            surrogate_model = FinalNet( \
         args.init_channels, CIFAR_CLASSES, args.layers, criterion, device, steps=args.num_nodes, controller_hid=args.controller_hid, entropy_coeff=args.entropy_coeff, edge_hid = args.edge_hid, transformer_nfeat = args.transformer_nfeat, transformer_nhid = args.transformer_nhid, transformer_dropout = args.transformer_dropout, transformer_normalize = args.transformer_normalize, loose_end = args.loose_end, op_type=args.op_type\
        )
            # target_model.arch_normal_master.demo = True
            # target_model.arch_reduce_master.demo = True

            surrogate_model.arch_normal = target_model.uni_random_transform(target_model.arch_normal)
            surrogate_model.arch_reduce = target_model.uni_random_transform(target_model.arch_reduce)
            if j == -1:

                target_model._model_optimizer = target_optimizer

                if args.scheduler == "naive_cosine":
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    target_optimizer, float(args.epochs), eta_min=args.lr_min
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
                        target_model.train()
                        target_model._model_optimizer.zero_grad()
                        n = input.size(0)
                        input = input.to(device)
                        target = target.to(device)
                        logits = target_model._inner_forward(input, target_model.arch_normal, target_model.arch_reduce)
                        loss = target_model._criterion(logits, target)
                        loss.backward()
                        target_model._model_optimizer.step()
                        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
                        objs.update(loss.item(), n)
                        top1.update(prec1.item(), n)
                        top5.update(prec5.item(), n)
                        if step % args.report_freq == 0:
                            logging.info('target_model %d: Step=%03d Loss=%e Top1=%f Top5=%f', its, step, objs.avg, top1.avg, top5.avg)
                
                logging.info('Training target_model %d: Loss=%e Top1=%f Top5=%f', its, objs.avg, top1.avg, top5.avg)

                target_acc_clean, _ = target_model.test_acc_single(valid_queue, logger, args)

                if not os.path.exists(os.path.join(args.save, str(its))):
                    os.mkdir(os.path.join(args.save, str(its)))

                utils.save(target_model, os.path.join(args.save, str(its), 'target.pt'))

                logging.info('Training target_baseline')

                target_model_baseline=FinalNet(
                    args.init_channels, CIFAR_CLASSES, args.layers, criterion, device, steps=args.num_nodes, controller_hid=args.controller_hid, entropy_coeff=args.entropy_coeff, edge_hid = args.edge_hid, transformer_nfeat = args.transformer_nfeat, transformer_nhid = args.transformer_nhid, transformer_dropout = args.transformer_dropout, transformer_normalize = args.transformer_normalize, loose_end = args.loose_end, op_type=args.op_type
                )

                target_model_baseline.arch_normal = target_model.arch_normal
                target_model_baseline.arch_reduce = target_model.arch_reduce

                target_baseline_optimizer = torch.optim.SGD(
                    target_model_baseline.model_parameters(),
                    args.lr,
                    momentum=args.momentum,
                    weight_decay=args.weight_decay
                )
                target_model_baseline.to(device)
                target_model_baseline._model_optimizer = target_baseline_optimizer

                if args.scheduler == "naive_cosine":
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    target_baseline_optimizer, float(args.epochs), eta_min=args.lr_min
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
                        target_model_baseline.train()
                        target_model_baseline._model_optimizer.zero_grad()
                        n = input.size(0)
                        input = input.to(device)
                        target = target.to(device)
                        logits = target_model_baseline._inner_forward(input, target_model_baseline.arch_normal, target_model_baseline.arch_reduce)
                        loss = target_model_baseline._criterion(logits, target)
                        loss.backward()
                        target_model_baseline._model_optimizer.step()
                        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
                        objs.update(loss.item(), n)
                        top1.update(prec1.item(), n)
                        top5.update(prec5.item(), n)
                        if step % args.report_freq == 0:
                            logging.info('Target_Model_Baseline %d: Step=%03d Loss=%e Top1=%f Top5=%f', its, step, objs.avg, top1.avg, top5.avg)
                logging.info('Target_Model_Baseline Final %d: Loss=%e Top1=%f Top5=%f',its, objs.avg, top1.avg, top5.avg)

                utils.save(target_model_baseline, os.path.join(args.save, str(its), 'target_baseline.pt'))
                target_acc_clean_baseline, _ = target_model_baseline.test_acc_single(valid_queue, logger, args)
            

            logging.info('Training surrogate model')

            surrogate_optimizer = torch.optim.SGD(
                surrogate_model.model_parameters(),
                args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay
            )

            surrogate_model.to(device)
            surrogate_model._model_optimizer = surrogate_optimizer

            if args.scheduler == "naive_cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                surrogate_optimizer, float(args.epochs), eta_min=args.lr_min
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
                        logging.info('Surrogate %d: Step=%03d Loss=%e Top1=%f Top5=%f', its, step, objs.avg, top1.avg, top5.avg)
            logging.info('Surrogate %d: Loss=%e Top1=%f Top5=%f', its, objs.avg, top1.avg, top5.avg)

            utils.save(surrogate_model, os.path.join(args.save, str(its), 'surrogate_model.pt'))
            utils.save(surrogate_model, os.path.join(args.save, str(its), 'surrogate_model_{}.pt'.format(j)))
            surrogate_acc_clean, _ = surrogate_model.test_acc_single(valid_queue, logger, args)

            logging.info('Training Completed, testing reward')

            utils.load(target_model, os.path.join(args.save, str(its), 'target.pt'))
            utils.load(target_model_baseline, os.path.join(args.save, str(its), 'target_baseline.pt'))
            utils.load(surrogate_model, os.path.join(args.save, str(its), 'surrogate_model.pt'))

            target_model.single = True
            target_model_baseline.single = True
            surrogate_model.single = True

            reward = utils.AvgrageMeter()
            acc_adv = utils.AvgrageMeter()
            optimized_acc_adv = utils.AvgrageMeter()
            acc_clean_, _ = target_model.test_acc_single(valid_queue, logger, args)
            optimized_acc_clean_, _ = surrogate_model.test_acc_single(valid_queue, logger, args)
            baseline_acc_clean_, _ = target_model_baseline.test_acc_single(valid_queue, logger, args)

            for step, (input, target) in enumerate(valid_queue):
                n = input.size(0)
                # if step >= args.accu_batch:
                #     break
                input = input.to(device)
                target = target.to(device)
                optimized_acc_adv_ = target_model.evaluate_transfer(surrogate_model, input, target)
                acc_adv_ = target_model.evaluate_transfer(target_model_baseline, input, target)
                reward_ = optimized_acc_adv_ - acc_adv_
                reward.update(reward_, n)
                acc_adv.update(acc_adv_, n)
                optimized_acc_adv.update(optimized_acc_adv_, n)
            logging.info("num: %d absolute_reward=%.2f, relative_reward=%.2f, target_acc_clean=%.2f, surrogate_acc_clean=%.2f, acc_adv=%.2f, surrogate_acc_adv=%.2f", its,\
                reward.avg, (optimized_acc_adv.avg - acc_adv.avg) / (acc_clean_ / 100 - acc_adv.avg), acc_clean_, optimized_acc_clean_, acc_adv.avg, optimized_acc_adv.avg)

            data_point = OrderedDict()
            
            data_point["target_arch"] = (target_model.arch_normal, target_model.arch_reduce)
            data_point["surrogate_arch"] = (surrogate_model.arch_normal, surrogate_model.arch_reduce)
            data_point["target_acc_clean"] = target_acc_clean
            data_point["surrogate_acc_clean"] = surrogate_acc_clean
            data_point["constrain"] = True
            data_point["train_info"] = {"index": its, "info": args.save}

            # data_point["target_arch"] = (target_normal, target_reduce)
            # data_point["surrogate_arch"] = (surrogate_normal, surrogate_reduce)
            data_point["absolute_reward"] = reward.avg
            data_point["relative_reward"] = (optimized_acc_adv.avg - acc_adv.avg) / (acc_clean_ / 100- acc_adv.avg)
            # data_point["target_acc_clean"] = acc_clean.avg
            # data_point["surrogate_acc_clean"] = optimized_acc_clean.avg
            data_point["acc_adv"] = acc_adv.avg
            data_point["surrogate_acc_adv"] = optimized_acc_adv.avg
            data_point["constrain"] = True
            # data_point["train_info"] = {"index": iteration, "info": args.save}

            train_dataset.append(data_point)
            train_data_j.append(data_point)


            if len(train_dataset) % args.save_freq == 0:
                torch.save(train_dataset, os.path.join(args.save, "train_data_constrain_" + str(len(train_dataset))))

            torch.save(train_dataset, os.path.join(args.save, "train_data_constrain_"))
            torch.save(train_data_j, os.path.join(args.save, str(its), 'train_data_{}'.format(its)))




if __name__ == '__main__':
    main()






