import torch
from torch.utils.data import Dataset
import utils
from genotypes import Genotype

class PredictorDataSet(Dataset):
    def __init__(self, data_path):
        self.data = utils.load_yaml(data_path)
        self._mode = "acc_adv_surrogate"
    
    def set_mode(self, mode):
        self._mode = mode

    def __len__(self):#返回整个数据集的大小
        return len(self.data)

    def __getitem__(self,index):
        if self._mode == "acc_adv_surrogate":
            target_arch = utils.genotype_to_arch(eval(self.data[index]['target_genotype']))
            surrogate_arch = utils.genotype_to_arch(eval(self.data[index]['surrogate_genotype']))
            acc_adv_surrogate = self.data[index]["adversarial_accuracy"]["surrogate"]
            return_data = {"target_arch": torch.tensor(target_arch), "surrogate_arch": torch.tensor(surrogate_arch), "label": torch.tensor(acc_adv_surrogate)}
        return return_data
