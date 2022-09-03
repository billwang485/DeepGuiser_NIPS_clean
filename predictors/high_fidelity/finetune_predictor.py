import os
import sys
STEM_WORK_DIR = os.path.join(os.getcwd(), "../..")
sys.path.append(STEM_WORK_DIR)
import glob
import shutil
import random
import logging
import argparse
import torch
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import utils
from predictors.predictors import VanillaGatesPredictor
from predictors.predictor_dataset import PredictorDataSet
from predictors.predictor_sampler import PredictorSampler

parser = argparse.ArgumentParser("DeepGuiser")
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
# parser.add_argument('--dataset', type=int, default=0, help='0 means build dataset from yaml data, 1 means load preprocessed dataqueue, 0 mode is princpled, 1 mode is fast')
parser.add_argument('--report_freq', type=int, default=100, help=' ')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=300, help='')
parser.add_argument('--save', type=str, default=utils.localtime_as_dirname(), help='experiment name')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--prefix', type=str, default='.', help='parent save path')
parser.add_argument('--scheduler', type=str, default='naive_cosine', help='type of LR scheduler')
parser.add_argument('--store', type=int, default=1, help='Whether to store the model')
parser.add_argument('--debug', action='store_true', default=False, help=' ')
parser.add_argument('--train_data_path', type=str, default=os.path.join(STEM_WORK_DIR, "transbench/data/high_fidelity/train.yaml"), help=' ')
parser.add_argument('--test_data_path', type=str, default=os.path.join(STEM_WORK_DIR, "transbench/data/high_fidelity/test.yaml"), help=' ')
parser.add_argument('--optimizer_config', '-oc', type=str, default='./optimizer_config.yaml', help='path to optimizer config file')
parser.add_argument('--predictor_config', "-pc", type=str, default='./predictor_config.yaml', help='path to predictor config file')
MIN_BATCH_SIZE = 4
args = parser.parse_args()
args.mode = 'high_fidelity'

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
    # save parameters
    
    predictor_config = utils.load_yaml(args.predictor_config)

    model = VanillaGatesPredictor( \
        device, predictor_config["op_type"], loss_type = predictor_config['loss_type'], dropout=predictor_config['dropout'], mode = predictor_config['mode'], concat= predictor_config['gates_concat']
    )

    logging.info('Building Dataset')

    args.load_dataset = False

    if args.load_dataset == False:
        train_data = PredictorDataSet(args.train_data_path)
    else:
        assert 0

    if args.load_dataset == False:
        test_data = PredictorDataSet(args.test_data_path)
    else:
        assert 0

    train_queue = torch.utils.data.DataLoader(
        train_data, args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(list(range(len(train_data)))),
        pin_memory=True, num_workers=4
    )
    valid_queue = torch.utils.data.DataLoader(
        test_data, 4,
        sampler=torch.utils.data.sampler.SequentialSampler(list(range(len(test_data)))),
        pin_memory=True, num_workers=4
    )

    logging.info('Building Dataset Completed: Train Set has %d pairs, Valid Set has %d pairs', len(train_data), len(test_data))

    

    shutil.copy(args.predictor_config, os.path.join(args.save, "predictor_config.yaml"))
    shutil.copy(args.optimizer_config, os.path.join(args.save, "optimzier_config.yaml"))
    torch.save(train_queue, os.path.join(args.save, 'train_queue.pt'))
    torch.save(valid_queue, os.path.join(args.save, 'valid_queue.pt'))

    if predictor_config["pretrained_weight"] is not None:
        tmp = torch.load(predictor_config["pretrained_weight"], map_location='cpu')
        model.load_state_dict(tmp, strict = False)

    model.to(device)

    optimizer_config = utils.load_yaml(args.optimizer_config)

    model_optimizer = utils.initialize_optimizer(model.parameters(), optimizer_config["learning_rate"], optimizer_config["momentum"], optimizer_config["weight_decay"], optimizer_config["type"])
    model.set_optimizer(model_optimizer)
    # data_trace = []
    valid_trace = []

    for epoch in range(args.epochs):
        logging.info('Epoch %d Starts', epoch)
        for step, data_point in enumerate(train_queue):
            loss, score, kendall = model.step(data_point, data_point["label"])
            data_point = utils.data_point_2_cpu(data_point)
            summaryWriter.add_scalar('train_loss', loss, epoch * len(train_queue) + step)
            summaryWriter.add_scalar('train_kendall', kendall, epoch * len(train_queue) + step)
            # data_trace.append({'data_point': utils.data_point_2_cpu(data_point), 'step': step, 'loss': loss.cpu(), 'kendall': kendall}) 
            tmp = random.randint(0, args.batch_size - 1)
            if step % args.report_freq == 0:
                logging.info('Epoch %d: Step=%d Loss=%.4f Score=%.4f Label=%.4f Kendall=%.4f', epoch, step, loss.item(), score[tmp].item(), data_point["label"][tmp].item(), kendall)
        if epoch % 1 == 0:
            avg_loss, patk, kendall = model.test(valid_queue, logger)
            summaryWriter.add_scalar('valid_loss', avg_loss, epoch * len(train_queue) + step)
            summaryWriter.add_scalar('valid_patk', patk, epoch * len(train_queue) + step)
            summaryWriter.add_scalar('valid_kendall', kendall, epoch * len(train_queue) + step)
            valid_trace.append({'epoch': epoch, 'loss': avg_loss, 'kendall': kendall, 'valid_patk': patk})
        # if epoch % 10 == 0:
        #     torch.save(model.predictor.state_dict(), os.path.join(args.save, 'predictor_state_dict_{}.pt'.format(epoch)))
        #     torch.save(model.predictor.state_dict(), os.path.join(args.save, 'predictor_state_dict.pt'))
            
        #     torch.save(valid_trace, os.path.join(args.save, 'valid_trace'))
        #     torch.save(data_trace, os.path.join(args.save, 'data_trace'))
        # logging.info('epoch %d is done', epoch)
        if args.store:
            torch.save(valid_trace, os.path.join(args.save, 'valid_trace'))
            # torch.save(data_trace, os.path.join(args.save, 'data_trace'))
            torch.save(model.state_dict(), os.path.join(args.save, 'predictor_state_dict.pt'))

if __name__ == '__main__':
    main()

