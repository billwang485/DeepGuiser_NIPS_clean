import enum
import os
import shutil
import sys
STEM_WORK_DIR = os.path.join(os.getcwd(), "../..")
sys.path.append(STEM_WORK_DIR)
import torch
from PyPDF2 import PdfFileMerger
import utils
import numpy as np
import genotypes
from genotypes import Genotype
import io
LOOSE_END_PRIMITIVES = [
    "max_pool_3x3",
    "avg_pool_3x3",
    "skip_connect",
    "sep_conv_3x3",
    "sep_conv_5x5",
    "conv_1x1",
    "conv_3x3",
    "conv_5x5",
    "null",
]
COUNT_NAME = "skip_connect"

RANGE_ = 20
sorted_data = utils.load_yaml(os.path.join(STEM_WORK_DIR, "experiments/transbench/sorted_reward_all_queue.yaml"))
sc_count = [0, 0]
sc_transform_count = []
for i in range(len(LOOSE_END_PRIMITIVES)):
    sc_transform_count.append([0, 0])
for i, data_point in enumerate(sorted_data):
    if i >= RANGE_:
        break
    target_genotype = eval(data_point["target_genotype"])
    for x in target_genotype.normal:
        if x[0] == COUNT_NAME:
            sc_count[0] += 1
    for x in target_genotype.reduce:
        if x[0] == COUNT_NAME:
            sc_count[1] += 1
    surrogate_genotype = eval(data_point["surrogate_genotype"])
    for i, x in enumerate(target_genotype.normal):
        if x[0] == COUNT_NAME:
            tmp = surrogate_genotype.normal[i][0]
            tmp = LOOSE_END_PRIMITIVES.index(tmp)
            # print(tmp)
            # print(len(sc_transform_count))
            sc_transform_count[tmp][0] += 1
    for i, x in enumerate(target_genotype.reduce):
        if x[0] == COUNT_NAME:
            tmp = surrogate_genotype.reduce[i][0]
            tmp = LOOSE_END_PRIMITIVES.index(tmp)
            sc_transform_count[tmp][1] += 1

save_dict = {"{}_count".format(COUNT_NAME):{"normal": sc_count[0] / (RANGE_ * 9), "reduce": sc_count[1] / (RANGE_ * 9)}}
save_dict["transform_count"] = dict()
save_dict["transform_count"]["normal"] = dict()
save_dict["transform_count"]["reduce"] = dict()
for i, x in enumerate(LOOSE_END_PRIMITIVES):
    save_dict["transform_count"]["normal"][x] = sc_transform_count[i][0] / sc_count[0]
    save_dict["transform_count"]["reduce"][x] = sc_transform_count[i][1] / sc_count[1]

utils.save_yaml(save_dict, os.path.join(STEM_WORK_DIR, "experiments/transbench/count/{}_top_{}.yaml".format(COUNT_NAME, RANGE_)))
sc_count = [0, 0]
sc_transform_count = []
for i in range(len(LOOSE_END_PRIMITIVES)):
    sc_transform_count.append([0, 0])
for i, data_point in enumerate(sorted_data[-RANGE_:]):
    # print(i)
    # print(data_point)
    target_genotype = eval(data_point["target_genotype"])
    for x in target_genotype.normal:
        if x[0] == COUNT_NAME:
            sc_count[0] += 1
    for x in target_genotype.reduce:
        if x[0] == COUNT_NAME:
            sc_count[1] += 1
    surrogate_genotype = eval(data_point["surrogate_genotype"])
    for i, x in enumerate(target_genotype.normal):
        if x[0] == COUNT_NAME:
            tmp = surrogate_genotype.normal[i][0]
            tmp = LOOSE_END_PRIMITIVES.index(tmp)
            # print(tmp)
            # print(len(sc_transform_count))
            sc_transform_count[tmp][0] += 1
    for i, x in enumerate(target_genotype.reduce):
        if x[0] == COUNT_NAME:
            tmp = surrogate_genotype.reduce[i][0]
            tmp = LOOSE_END_PRIMITIVES.index(tmp)
            sc_transform_count[tmp][1] += 1

save_dict = {"{}_count".format(COUNT_NAME):{"normal": sc_count[0] / (RANGE_ * 9), "reduce": sc_count[1] / (RANGE_ * 9)}}
save_dict["transform_count"] = dict()
save_dict["transform_count"]["normal"] = dict()
save_dict["transform_count"]["reduce"] = dict()
for i, x in enumerate(LOOSE_END_PRIMITIVES):
    save_dict["transform_count"]["normal"][x] = sc_transform_count[i][0] / sc_count[0]
    save_dict["transform_count"]["reduce"][x] = sc_transform_count[i][1] / sc_count[1]

utils.save_yaml(save_dict, os.path.join(STEM_WORK_DIR, "experiments/transbench/count/{}_last_{}.yaml".format(COUNT_NAME, RANGE_)))

sc_count = [0, 0]
sc_transform_count = []
for i in range(len(LOOSE_END_PRIMITIVES)):
    sc_transform_count.append([0, 0])
for i, data_point in enumerate(sorted_data):
    # print(i)
    # print(data_point)
    target_genotype = eval(data_point["target_genotype"])
    for x in target_genotype.normal:
        if x[0] == COUNT_NAME:
            sc_count[0] += 1
    for x in target_genotype.reduce:
        if x[0] == COUNT_NAME:
            sc_count[1] += 1
    surrogate_genotype = eval(data_point["surrogate_genotype"])
    for i, x in enumerate(target_genotype.normal):
        if x[0] == COUNT_NAME:
            tmp = surrogate_genotype.normal[i][0]
            tmp = LOOSE_END_PRIMITIVES.index(tmp)
            # print(tmp)
            # print(len(sc_transform_count))
            sc_transform_count[tmp][0] += 1
    for i, x in enumerate(target_genotype.reduce):
        if x[0] == COUNT_NAME:
            tmp = surrogate_genotype.reduce[i][0]
            tmp = LOOSE_END_PRIMITIVES.index(tmp)
            sc_transform_count[tmp][1] += 1

save_dict = {"{}_count".format(COUNT_NAME):{"normal": sc_count[0] / (len(sorted_data) * 9), "reduce": sc_count[1] / (len(sorted_data) * 9)}}
save_dict["transform_count"] = dict()
save_dict["transform_count"]["normal"] = dict()
save_dict["transform_count"]["reduce"] = dict()
for i, x in enumerate(LOOSE_END_PRIMITIVES):
    save_dict["transform_count"]["normal"][x] = sc_transform_count[i][0] / sc_count[0]
    save_dict["transform_count"]["reduce"][x] = sc_transform_count[i][1] / sc_count[1]

utils.save_yaml(save_dict, os.path.join(STEM_WORK_DIR, "experiments/transbench/count/{}_all.yaml".format(COUNT_NAME)))


