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
from genotypes import ResBlock
from search_model_predictor import NASNetwork as Network
import random
import shutil
import genotypes
import time
import os
import re

localtime = time.asctime(time.localtime(time.time()))
x = re.split(r"[\s,(:)]", localtime)
default_EXP = " ".join(x[1:-1])
parser = argparse.ArgumentParser("DeepGuiser")
parser.add_argument("--gpu", type=int, default=0, help="gpu device id")
parser.add_argument("--budget", nargs="+", type=float, default=[4.9, 5.2], help=" ")
parser.add_argument("--init_channels", type=int, default=20, help="number of init channels")
parser.add_argument("--layers", type=int, default=8, help="total number of layers")
parser.add_argument("--save", type=str, default=default_EXP, help="experiment name")
parser.add_argument("--seed", type=int, default=1234, help="random seed")
parser.add_argument("--num", type=int, default=5, help="random seed")
parser.add_argument("--prefix", type=str, default=".", help="parent save path")
parser.add_argument("--controller_hid", type=int, default=100, help="temperature for lstm")
parser.add_argument("--entropy_coeff", nargs="+", type=float, default=[0.003, 0.003], help="coefficient for entropy: [normal, reduce]")
parser.add_argument("--edge_hid", type=int, default=100, help="edge hidden dimension")
parser.add_argument("--transformer_nfeat", type=int, default=1024, help="feature dimension of each node")
parser.add_argument("--transformer_nhid", type=int, default=100, help="hidden dimension")
parser.add_argument("--transformer_dropout", type=float, default=0, help="dropout rate for transformer")
parser.add_argument("--transformer_normalize", action="store_true", default=False, help="use normalize in GCN")
parser.add_argument("--arch", type=str, default="ResBlock", help="which architecture to use")
parser.add_argument("--num_nodes", type=int, default=4, help="number of intermediate nodes")
parser.add_argument("--op_type", type=str, default="L", help="LOOSE_END_PRIMITIVES | FULLY_CONCAT_PRIMITIVES")
parser.add_argument("--pw", type=str, default="./train_search/train_search_predictor/May 17 07 39 58/model_5000.pt", help=" ")
parser.add_argument("--pwp", type=str, default="./predictor/finetune/May 14 10 52 44/predictor_state_dict_30.pt", help=" ")
parser.add_argument("--debug", action="store_true", default=False, help=" ")
parser.add_argument("--rt", type=str, default="x", help=" ")
args = parser.parse_args()

assert args.pw != " "
if args.op_type == "L":
    args.op_type = "LOOSE_END_PRIMITIVES"
elif args.op_type == "B":
    args.op_type = "BOTTLENECK_PRIMITIVES"
else:
    assert 0
if args.op_type == "LOOSE_END_PRIMITIVES":
    args.loose_end = True
else:
    args.loose_end = False

if not os.path.exists(args.prefix):
    os.makedirs(args.prefix)
tmp = "rebuttal/pareto"
if args.debug:
    tmp = os.path.join(tmp, "debug")
args.save = os.path.join(args.prefix, tmp, args.save)

utils.create_exp_dir(args.save, scripts_to_save=glob.glob("*.py") + glob.glob("*.sh") + glob.glob("*.yml"))
size_ = [1, 3, 32, 32]
log_format = "%(asctime)s %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt="%m/%d %I:%M:%S %p")
fh = logging.FileHandler(os.path.join(args.save, "log.txt"))
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

    genotype = eval("genotypes.%s" % args.arch)

    model = Network(
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
    )

    model.re_initialize_arch_transformer()
    model._initialize_predictor(args, "WarmUp")
    utils.load(model, args.pw)
    model.predictor.load_state_dict(torch.load(args.pwp, map_location="cpu"))

    min_flops = args.budget[0]
    max_flops = args.budget[1]
    gap = max_flops - min_flops

    model.to(device)
    model.predictor.eval()
    result_list_list = []

    for j in range(1, 4):
        i = 0
        result_list = []
        find_times = 0
        while i < args.num:
            find_times = find_times + 1
            if find_times > 10000:
                break
            target_normal, target_reduce = utils.genotype_to_arch(ResBlock, model.op_type)
            surrogate_normal = model.uni_random_transform(target_normal, 100)
            surrogate_reduce = model.uni_random_transform(target_reduce, 100)
            arch_normal_ = []
            arch_reduce_ = []
            optimized_normal_ = []
            optimized_reduce_ = []
            for j in range(len(target_normal)):
                arch_normal_.append(([target_normal[j][0]], [target_normal[j][1]], [target_normal[j][2]]))
                arch_reduce_.append(([target_reduce[j][0]], [target_reduce[j][1]], [target_reduce[j][2]]))
                optimized_normal_.append(([surrogate_normal[j][0]], [surrogate_normal[j][1]], [surrogate_normal[j][2]]))
                optimized_reduce_.append(([surrogate_reduce[j][0]], [surrogate_reduce[j][1]], [surrogate_reduce[j][2]]))
            score0 = model.predictor.forward([arch_normal_, arch_reduce_], [optimized_normal_, optimized_reduce_]).item()
            target_flops = utils.compute_flops(model, size_, target_normal, target_reduce, "null")
            surrogate_flops = utils.compute_flops(model, size_, surrogate_normal, surrogate_reduce, "null")
            budget = surrogate_flops / target_flops
            # print(min_flops)
            # print(max_flops)
            if budget < min_flops or budget >= max_flops:

                continue
            else:
                # print(budget >= max_flops, budget, max_flops)
                save_dict = {}
                save_dict["target_arch"] = (target_normal, target_reduce)
                save_dict["surrogate_arch"] = (surrogate_normal, surrogate_reduce)
                save_dict["target_flops"] = target_flops
                save_dict["surrogate_flops"] = surrogate_flops
                save_dict["budget"] = budget
                save_dict["acc_adv"] = score0
                result_list.append(save_dict)
                logging.info("Transform %d, budget %f, score %f", len(result_list), budget, score0)
                i = i + 1
        result_list_list.append(result_list)
        min_flops = max_flops
        max_flops = max_flops + gap
    torch.save(result_list_list, os.path.join(args.save, "result_list_list"))


if __name__ == "__main__":
    main()
