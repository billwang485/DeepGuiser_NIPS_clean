import os
import sys
STEM_WORK_DIR = os.path.join(os.getcwd(), "../../..")
sys.path.append(STEM_WORK_DIR)
import utils

merge_dirs = ["transbench/high_fidelity/train_0_1000_new.yaml", 
            "transbench/high_fidelity/train_1000_2000_new.yaml",
            "transbench/high_fidelity/train_2000_3000_new.yaml",
            "transbench/high_fidelity/train_3000_4000_new.yaml",
            "transbench/high_fidelity/train_4000_5000_new.yaml",
            "transbench/high_fidelity/train_5000_6000_new.yaml",
            "transbench/high_fidelity/train_6000_7000_new.yaml",
            "transbench/high_fidelity/train_7000_8000_new.yaml",
            "transbench/high_fidelity/train_8000_8082_new.yaml"]
sub_length = 1000
train = list()
for i in range(len(merge_dirs)):
    tmp = utils.load_yaml(os.path.join(STEM_WORK_DIR, merge_dirs[i]))
    train.extend(tmp)

utils.save_yaml(train, "train_new.yaml")