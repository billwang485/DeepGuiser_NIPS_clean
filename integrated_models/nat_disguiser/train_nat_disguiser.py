import os
import sys
STEM_WORK_DIR = os.path.join(os.getcwd(), "..", "..")
sys.path.append(STEM_WORK_DIR)
import glob
import random
import shutil
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import utils
from integrated_models.nat_disguiser.nat_disguiser import NATDisguiser
from learners.nat_disguiser_leaner import NATDisguiserLearner


parser = argparse.ArgumentParser("DeepGuiser")
parser.add_argument("--data", type=str, default=os.path.join(STEM_WORK_DIR, "../data"), help="location of the data corpus")
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument("--learning_rate", "-lr", type=float, default=0.05, help="init learning rate")
parser.add_argument("--learning_rate_min", "-lr_min", type=float, default=0.01, help="min learning rate")
parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
parser.add_argument("--weight_decay", type=float, default=3e-4, help="weight decay")
parser.add_argument("--save_freq", type=int, default=100, help="save frequency")
parser.add_argument("--gpu", type=int, default=0, help="gpu device id")
parser.add_argument("--iterations", "-its", type=int, default=1, help="number of training iteration")
parser.add_argument("--init_channels", type=int, default=20, help="number of init channels")
parser.add_argument("--layers", type=int, default=8, help="total number of layers")
parser.add_argument("--save", type=str, default=utils.localtime_as_dirname(), help="experiment name")
parser.add_argument("--seed", type=int, default=1234, help="random seed")
parser.add_argument("--prefix", type=str, default=".", help="parent save path")
parser.add_argument("--controller_hid", type=int, default=100, help="controller hidden dimension")
parser.add_argument("--entropy_coeff", nargs="+", type=float, default=[0.000, 0.000], help="coefficient for entropy: [normal, reduce]")
parser.add_argument("--gamma", type=float, default=0.99, help="time decay for baseline update")
parser.add_argument("--controller_start_training", type=int, default=0, help="Epoch that the training of controller starts")
parser.add_argument("--scheduler", type=str, default="naive_cosine", help="type of LR scheduler")
parser.add_argument("--store", type=int, default=1, help="Whether to store the model")
parser.add_argument("--edge_hid", type=int, default=100, help="edge hidden dimension")
parser.add_argument("--transformer_weight_decay", type=float, default=5e-4, help="learning rate for pruner")
parser.add_argument("--transformer_nfeat", type=int, default=1024, help="feature dimension of each node")
parser.add_argument("--transformer_nhid", type=int, default=100, help="hidden dimension")
parser.add_argument("--transformer_dropout", type=float, default=0, help="dropout rate for transformer")
parser.add_argument("--transformer_normalize", action="store_true", default=False, help="use normalize in GCN")
parser.add_argument("--num_nodes", type=int, default=4, help="number of intermediate nodes")
parser.add_argument("--op_type", type=str, default="LOOSE_END_PRIMITIVES", help="LOOSE_END_PRIMITIVES | FULLY_CONCAT_PRIMITIVES")
parser.add_argument("--pretrained_weight", "-pw", type=str, default=os.path.join(STEM_WORK_DIR, "supernet", "selected_supernets"), help="The path to pretrained supernet")
parser.add_argument("--reward_type", "-rt", type=str, default="absolute", help="reward type")
parser.add_argument("--sample_strategy", type=str, default="demo", help="sample strategy")
parser.add_argument("--debug", action="store_true", default=False, help=" ")
# parser.add_argument("--imitation", action="store_true", default=False, help=" ")
parser.add_argument("--accumulate_batch", "-accu_batch", type=int, default=10, help=" ")
parser.add_argument("--pgd_step", type=int, default=10, help=" ")
parser.add_argument("--pretrained_arch_embedder", "-pae", type=str, default=" ", help="The path to pretrained weight of arch embedder")
args = parser.parse_args()

if args.op_type == "LOOSE_END_PRIMITIVES":
    args.loose_end = True
else:
    args.loose_end = False
args.cutout = False
utils.preprocess_exp_dir(args)
utils.create_exp_dir(args.save, scripts_to_save=glob.glob("*.py") + glob.glob("*.sh") + glob.glob("*.yml"))

utils.initialize_logger(args)

CIFAR_CLASSES = 10

summaryWriter = SummaryWriter(os.path.join(args.save, "runs"))


def main():
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed(args.seed)
        cudnn.benchmark = True
        cudnn.enabled = True
        logging.info("GPU device = %d" % args.gpu)
    else:
        logging.info("no GPU available, use CPU!!")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    logging.info("args = %s" % args)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    model = NATDisguiser(
        args.init_channels,
        CIFAR_CLASSES,
        args.layers,
        criterion,
        device,
        steps=args.num_nodes,
        controller_hid=args.controller_hid,
        entropy_coeff=args.entropy_coeff,
        edge_hid=args.edge_hid,
        transformer_nfeat=args.transformer_nfeat,
        transformer_nhid=args.transformer_nhid,
        transformer_dropout=args.transformer_dropout,
        transformer_normalize=args.transformer_normalize,
        loose_end=args.loose_end,
        op_type=args.op_type,
        pdg_step=args.pgd_step,
    )

    twin_supernet = model.get_twin_model()

    _, valid_queue, _ = utils.get_cifar_data_queue(args)

    utils.load_supernet(model, os.path.join(args.pretrained_weight, "supernet.pt"))
    utils.load_supernet(twin_supernet, os.path.join(args.pretrained_weight, "supernet_twin.pt"))

    os.mkdir(os.path.join(args.save, "selected_supernets"))
    shutil.copy(os.path.join(args.pretrained_weight, "supernet.pt"), os.path.join(args.save, "selected_supernets", "supernet.pt"))
    shutil.copy(os.path.join(args.pretrained_weight, "supernet_twin.pt"), os.path.join(args.save, "selected_supernets", "supernet_twin.pt"))
    model.initialize_arch_transformer()

    learner = NATDisguiserLearner(model, twin_supernet, args)

    model.to(device)
    twin_supernet.to(device)
    if args.pretrained_arch_embedder != " ":
        utils.load_pretrained_arch_embedder(model, args.pretrained_arch_embedder)

    assert args.reward_type == "absolute" or args.reward_type == "relative"

    model.reward_type = "absolute" if args.reward_type == "absolute" else "relative"

    for iteration in range(args.iterations):

        logging.info("iteration %d lr %e", iteration, learner.current_lr)

        for step, (input, target) in enumerate(valid_queue):
            n = input.size(0)
            if step >= args.accumulate_batch:
                break
            input = input.to(device)
            target = target.to(device)
            reward, optimized_acc_adv, acc_adv, ent, loss = learner.step(input, target)
            if step == args.accumulate_batch - 1:
                summaryWriter.add_scalar("reward", reward, iteration)
                summaryWriter.add_scalar("loss", loss, iteration)
                logging.info("Updating Theta iteration=%d optimized_acc_adv=%.2f acc_adv=%.2f reward=%.2f, ent = %.2f,loss=%.3f", iteration, optimized_acc_adv, acc_adv, reward, ent, loss)
        if iteration % args.save_freq == 0:
            utils.save_nat_disguiser(model, os.path.join(args.save, "model_{}.pt".format(iteration)))
        # transformer.scheduler.step()

    if args.store == 1:
        utils.save_nat_disguiser(model, os.path.join(args.save, "model_final.pt"))
        # torch.save(model.best_pair, os.path.join(args.save, "best_transform_pair"))


if __name__ == "__main__":
    main()
