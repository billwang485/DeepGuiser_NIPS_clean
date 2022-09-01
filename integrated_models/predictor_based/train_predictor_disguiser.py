import os
import sys
STEM_WORK_DIR = os.path.join(os.getcwd(), "..", "..")
sys.path.append(STEM_WORK_DIR)
import shutil
import glob
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import utils
from integrated_models.predictor_based.predictor_based_disguiser import PredictorBasedDisguiser
from learners.predictor_disguiser_learner import PredictorBasedLearner
import random
parser = argparse.ArgumentParser("DeepGuiser")
parser.add_argument('--save_freq', type=int, default=1000, help='save frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--iteration', "-its", type=int, default=10000, help='number of training iteration')
parser.add_argument('--save', type=str, default=utils.localtime_as_dirname(), help='experiment name')
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

args.cutout = False
utils.preprocess_exp_dir(args)
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py')+glob.glob('*.sh')+glob.glob('*.yml'))
logger = utils.initialize_logger(args)
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

    optimizer_config = utils.load_yaml(args.optimizer_config)
    shutil.copy(args.optimizer_config, os.path.join(args.save, "optimizer_config.yaml"))


    learner = PredictorBasedLearner(model, optimizer_config, strategy_config)

    model.to(device)

    # assert (args.rt == 'a' or args.rt == 'r')

    # model.reward_type = 'absolute' if args.rt == 'a' else 'relative'

    # model.flops_limit = args.flps
    # model.op_diversity = args.opdiv
    # model.num_limit = args.nlit

    # model.set_thre(args)


    for iteration in range(args.iteration):
        loss, reward, ent, auxillary_info = learner.step()
        summaryWriter.add_scalar('reward', reward, iteration)
        summaryWriter.add_scalar('loss', loss, iteration)
        # summaryWriter.add_scalar('flops_limit', flops_limit, iteration)
        # summaryWriter.add_scalar('op_div', op_div, iteration)
        if "num_changes" in auxillary_info.keys():
            summaryWriter.add_scalar('num_changes', auxillary_info["num_changes"], iteration)
        if iteration % 100 == 0:
            logging.info('iteration %d lr %e', iteration, learner.current_lr)
            logging.info('Updating Theta iteration=%d reward=%.2f, ent = %.2f,loss=%.3f', iteration, reward, ent, loss)
        if iteration % args.save_freq == 0:
            utils.save_predictor_based_disguiser(model, os.path.join(args.save, 'model_{}.pt'.format(iteration)))
            # torch.save_predictor_based_disguiser(model.best_pair, os.path.join(args.save, "best_transform_pair"))
        # transformer.scheduler.step()

    if args.store == 1:
        utils.save_predictor_based_disguiser(model, os.path.join(args.save, 'model_final.pt'))
        # torch.save_predictor_based_disguiser(model.best_pair, os.path.join(args.save, "best_transform_pair"))

if __name__ == '__main__':
    main()