import os
import sys
STEM_WORK_DIR = os.path.join(os.getcwd(), "../../..")
sys.path.append(STEM_WORK_DIR)
import utils

test_data = utils.load_yaml("test.yaml")
train_data = utils.load_yaml("train.yaml")
overlap = 0
for i, y in enumerate(test_data):
    for j, x in enumerate(train_data):
        if x["target_genotype"] == x["surrogate_genotype"]:
            overlap = 1
            print("Overlap! i = {}, j = {}".format(i, j))
            # break

