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

test_queue = utils.load_yaml(os.path.join(STEM_WORK_DIR, "experiments/transbench/all_queue.yaml"))
# train_queue = utils.load_yaml(os.path.join(STEM_WORK_DIR, "transbench/data/high_fidelity/train.yaml"))
# all_queue = list()
# test_queue.extend(train_queue)

# for data_point in test_queue:
#     target_normal, target_reduce = utils.genotype_to_arch(eval(data_point["target_genotype"]))
#     surrogate_normal, surrogate_reduce = utils.genotype_to_arch(eval(data_point["surrogate_genotype"]))
#     flag1 = utils.check_transform(target_normal, surrogate_normal)
#     flag2 = utils.check_transform(target_reduce, surrogate_reduce)
#     if flag1 and flag2:
#         all_queue.append(data_point)

# utils.save_yaml(all_queue, os.path.join(STEM_WORK_DIR, "experiments/transbench/all_queue.yaml"))

test_queue.sort(key = lambda x:  x["adversarial_accuracy"]["surrogate"] -  x["adversarial_accuracy"]["baseline"], reverse=True)
utils.save_yaml(test_queue, os.path.join(STEM_WORK_DIR, "experiments/transbench/sorted_reward_all_queue.yaml"))
save_path = "experiments/transbench/visualization"
for i, data_point in enumerate(test_queue):
    surrogate_genotype = eval('genotypes.{}'.format(data_point["surrogate_genotype"]))
    utils.draw_genotype(surrogate_genotype.normal, np.max(list(surrogate_genotype.normal_concat) + list(surrogate_genotype.reduce_concat)) -1, os.path.join(STEM_WORK_DIR, save_path, "surrogate_normal"), surrogate_genotype.normal_concat)
    utils.draw_genotype(surrogate_genotype.reduce, np.max(list(surrogate_genotype.normal_concat) + list(surrogate_genotype.reduce_concat)) -1, os.path.join(STEM_WORK_DIR, save_path, "surrogate_reduce"), surrogate_genotype.reduce_concat)


    file_merger = PdfFileMerger()

    file_merger.append(os.path.join(os.path.join(STEM_WORK_DIR, save_path, "surrogate_normal.pdf")))
    file_merger.append(os.path.join(os.path.join(STEM_WORK_DIR, save_path, "surrogate_reduce.pdf")))

    file_merger.write(os.path.join(os.path.join(STEM_WORK_DIR, save_path, "surrogate_{}.pdf".format(i))))

    os.remove(os.path.join(os.path.join(STEM_WORK_DIR, save_path, "surrogate_normal.pdf")))
    os.remove(os.path.join(os.path.join(STEM_WORK_DIR, save_path, "surrogate_normal")))
    os.remove(os.path.join(os.path.join(STEM_WORK_DIR, save_path, "surrogate_reduce.pdf")))
    os.remove(os.path.join(os.path.join(STEM_WORK_DIR, save_path, "surrogate_reduce")))

    target_genotype = eval('genotypes.{}'.format(data_point["target_genotype"]))
    utils.draw_genotype(target_genotype.normal, np.max(list(target_genotype.normal_concat) + list(target_genotype.reduce_concat)) -1, os.path.join(STEM_WORK_DIR, save_path, "target_normal"), target_genotype.normal_concat)
    utils.draw_genotype(target_genotype.reduce, np.max(list(target_genotype.normal_concat) + list(target_genotype.reduce_concat)) -1, os.path.join(STEM_WORK_DIR, save_path, "target_reduce"), target_genotype.reduce_concat)


    file_merger = PdfFileMerger()

    file_merger.append(os.path.join(os.path.join(STEM_WORK_DIR, save_path, "target_normal.pdf")))
    file_merger.append(os.path.join(os.path.join(STEM_WORK_DIR, save_path, "target_reduce.pdf")))

    file_merger.write(os.path.join(os.path.join(STEM_WORK_DIR, save_path, "target_{}.pdf".format(i))))

    os.remove(os.path.join(os.path.join(STEM_WORK_DIR, save_path, "target_normal.pdf")))
    os.remove(os.path.join(os.path.join(STEM_WORK_DIR, save_path, "target_normal")))
    os.remove(os.path.join(os.path.join(STEM_WORK_DIR, save_path, "target_reduce.pdf")))
    os.remove(os.path.join(os.path.join(STEM_WORK_DIR, save_path, "target_reduce")))

    file_merger = PdfFileMerger()

    file_merger.append(os.path.join(os.path.join(STEM_WORK_DIR, save_path, "target_{}.pdf".format(i))))
    file_merger.append(os.path.join(os.path.join(STEM_WORK_DIR, save_path, "surrogate_{}.pdf".format(i))))

    file_merger.write(os.path.join(os.path.join(STEM_WORK_DIR, save_path, "top_{}.pdf".format(i))))

    os.remove(os.path.join(os.path.join(STEM_WORK_DIR, save_path, "target_{}.pdf".format(i))))
    os.remove(os.path.join(os.path.join(STEM_WORK_DIR, save_path, "surrogate_{}.pdf".format(i))))

