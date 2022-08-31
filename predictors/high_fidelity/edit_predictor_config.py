import sys
import os
sys.path.append(os.path.join(os.getcwd(), "../.."))
print(os.path.join(os.getcwd(), "../.."))

import utils

predictor_config = {"epochs": None, "type": "VanillaGates", "mode":"high_fidelity"}

utils.save_yaml(predictor_config, "predictor_config.yaml")