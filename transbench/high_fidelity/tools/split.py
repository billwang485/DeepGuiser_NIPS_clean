import os
import shutil
import sys
STEM_WORK_DIR = os.path.join(os.getcwd(), "../../..")
sys.path.append(STEM_WORK_DIR)
import utils


sub_length = 1000

train_data = utils.load_yaml("../../data/high_fidelity/train.yaml")

flag = 1
count = 0
while flag:
    sub_train_data = train_data[count * sub_length: count * sub_length + sub_length]
    utils.save_yaml(sub_train_data, "train_{}_{}.yaml".format(count * sub_length , count * sub_length + sub_length))
    count += 1
    if count * sub_length + sub_length > len(train_data):
        sub_train_data = train_data[count * sub_length: ]
        utils.save_yaml(sub_train_data, "train_{}_{}.yaml".format(count * sub_length , len(train_data)))
        break
