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
from predictors import PredictorDataSet
from torch.utils.tensorboard import SummaryWriter
localtime = time.asctime( time.localtime(time.time()))
x = re.split(r"[\s,(:)]",localtime)
default_EXP = " ".join(x[1:-1])
parser = argparse.ArgumentParser("DeepGuiser")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--lr', type=float, default=1e-3, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--save_freq', type=int, default=100, help='save frequency')
parser.add_argument('--report_freq', type=int, default=1, help=' ')
parser.add_argument('--test_freq', type=int, default=20, help='test frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=300, help='')
parser.add_argument('--ta', type=int, default=68000, help='total archs')
parser.add_argument('--fp', type=float, default=0.5, help='free portion')
parser.add_argument('--init_channels', type=int, default=20, help='number of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--save', type=str, default=default_EXP, help='experiment name')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--train_portion', type=float, default=0.95, help='data portion for training weights')
# parser.add_argument('--n_archs', type=int, default=10, help='number of candidate archs')
parser.add_argument('--prefix', type=str, default='.', help='parent save path')
parser.add_argument('--controller_hid', type=int, default=100, help='controller hidden dimension')
parser.add_argument('--entropy_coeff', nargs='+', type=float, default=[0.000, 0.000], help='coefficient for entropy: [normal, reduce]')
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
parser.add_argument('--transformer_dropout', type=float, default=0, help='dropout rate for transformer')
parser.add_argument('--transformer_normalize', action='store_true', default=False, help='use normalize in GCN')
parser.add_argument('--num_nodes', type=int, default=4, help='number of intermediate nodes')
parser.add_argument('--op_type', type=str, default='LOOSE_END_PRIMITIVES', help='LOOSE_END_PRIMITIVES | FULLY_CONCAT_PRIMITIVES')
parser.add_argument('--rt', type=str, default='a', help='reward_type')
parser.add_argument('--debug', action='store_true', default=False, help=' ')
parser.add_argument('--use_sch', action='store_true', default=False, help=' ')
parser.add_argument('--data_dir', type=str, default='predictor/lf_data', help=' ')
parser.add_argument('--loss_type', type=str, default='mse', help=' ')
# parser.add_argument('--pw', type=str, default='LOOSE_END_supernet', help='The path to pretrained weight if there is(dir)')
args = parser.parse_args()
args.mode = 'low_fidelity'
if args.op_type=='LOOSE_END_PRIMITIVES':
    args.loose_end = True
else:
    args.loose_end = False

if not os.path.exists(args.prefix):
    os.makedirs(args.prefix)
tmp = 'predictor/pretrain'
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

    logging.info('Building dataset')

    args.load_dataset = False

    if args.load_dataset == False:
        train_data = PredictorDataSet('predictor/lf_data/lf_train_data', \
            'no_name/lf_data', \
            args)
        if not args.debug:
            torch.save(train_data, os.path.join(args.data_dir, 'train_data'))
        else:
            torch.save(train_data, os.path.join(args.data_dir, 'train_data_debug'))
    else:
        # train_data = torch.load(train_data)
        if not args.debug:
            train_data = torch.load(os.path.join(args.data_dir, 'train_data'), map_location='cpu')
            # train_queuq = torch.load(os.path.join(args.data_dir, 'train_queu'), map_location='cpu')
        else:
            train_data = torch.load(os.path.join(args.data_dir, 'train_data_debug'), map_location='cpu')

    logging.info('Building Completed length = %d', len(train_data))
    
    num_train = len(train_data)
    indices = list(range(num_train))
    random.shuffle(indices)
    split = int(np.floor(num_train * args.train_portion))

    if not args.debug:

        train_queue = torch.utils.data.DataLoader(
            train_data, args.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
            # shuffle= False,
            pin_memory=True, num_workers=2
        )

        valid_queue = torch.utils.data.DataLoader(
            train_data, args.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
            pin_memory=True, num_workers=2
        )
    else:
        train_queue = torch.utils.data.DataLoader(
            train_data, args.batch_size,
            sampler=torch.utils.data.sampler.SequentialSampler(indices[:split]),
            # shuffle= False,
            pin_memory=True, num_workers=2
        )

        valid_queue = torch.utils.data.DataLoader(
            train_data, args.batch_size,
            sampler=torch.utils.data.sampler.SequentialSampler(indices[split:num_train]),
            pin_memory=True, num_workers=2
        )
    if not args.debug:
        torch.save(train_queue, os.path.join(args.save, 'train_queue'))
        torch.save(valid_queue, os.path.join(args.save, 'valid_queue'))

    # utils.load(model, os.path.join(args.pw, 'model.pt'))
    # model.re_initialize_arch_transformer()
    # utils.load(model_twin, os.path.join(args.pw, 'model_twin.pt'))
    # model_twin.re_initialize_arch_transformer()
    # transformer = Transformer(model, model_twin, args)
    model._initialize_predictor(args, "WarmUp")

    # tmp = torch.load('predictor/pretrain/debug/Apr 24 14 22 55/predictor_state_dict_3.pt', map_location='cpu')

    # model.predictor.load_state_dict(tmp)

    model.to(device)
    # model_twin.to(device)

    # assert (args.rt == 'a' or args.rt == 'r')

    model.reward_type = 'absolute' if args.rt == 'a' else 'relative'

    data_trace = []
    convergence = []

    for epoch in range(args.epochs):
        model.predictor.epoch = epoch
        for step, data_point in enumerate(train_queue):
            # if step > 0:
            #     break
            # data_point = data_point.to(device)
            if args.rt == 'a':
                label = data_point["absolute_reward"]
            elif args.rt == 'r':
                label = data_point["relative_reward"]
            else:
                label = data_point["acc_adv"]
                # label = data_point['absolute_reward_supernet']
            # label = label.to(device)
            loss, score, kendall = model.predictor.step(data_point, label)
            summaryWriter.add_scalar('train_loss', loss, epoch * len(train_queue) + step)
            summaryWriter.add_scalar('train_kendall', kendall, epoch * len(train_queue) + step)
            data_trace.append({'data_point': data_point, 'step': step, 'loss': loss, 'kendall': kendall}) 
            if step % 100 == 0:
                logging.info('epoch %d: step=%d loss=%.4f score=%.4f label=%.4f kendall=%.4f', epoch, step, loss.item(), score[0].item(), label[0].item(), kendall)
            if step % args.test_freq == 0 or step == len(train_queue) - 1:
                avg_loss, patk, kendall = model.predictor.test(valid_queue, logger)
                summaryWriter.add_scalar('valid_loss', avg_loss, epoch * len(train_queue) + step)
                summaryWriter.add_scalar('valid_patk', patk, epoch * len(train_queue) + step)
                summaryWriter.add_scalar('valid_kendall', kendall, epoch * len(train_queue) + step)
            # if step % args.save_freq == 0:
            #     torch.save(model.predictor.state_dict(), os.path.join(args.save, 'predictor_state_dict_{}_{}.pt'.format(epoch, step)))
            #     torch.save(model.predictor.state_dict(), os.path.join(args.save, 'predictor_state_dict.pt'))
            #     torch.save(data_trace, os.path.join(args.save, 'data_trace'))
        if args.use_sch:
            model.predictor.scheduler.step()
        torch.save(model.predictor.state_dict(), os.path.join(args.save, 'predictor_state_dict_{}.pt'.format(epoch)))
        torch.save(model.predictor.state_dict(), os.path.join(args.save, 'predictor_state_dict.pt'))
        logging.info('epoch %d is done', epoch)
    torch.save(model.predictor.state_dict(), os.path.join(args.save, 'predictor_state_dict.pt'))

if __name__ == '__main__':
    main()

