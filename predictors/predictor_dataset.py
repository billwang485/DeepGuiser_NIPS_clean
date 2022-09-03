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
            target_genotype = eval(self.data[index]['target_genotype'])
            target_arch = utils.genotype_to_arch(target_genotype)
            surrogate_genotype = eval(self.data[index]['surrogate_genotype'])
            surrogate_arch = utils.genotype_to_arch(surrogate_genotype)
            acc_adv_surrogate = self.data[index]["adversarial_accuracy"]["surrogate"]
            target_concat = torch.zeros([2, 4])
            surrogate_concat = torch.zeros([2, 4]) 
            for i, node in enumerate(target_genotype[1]):
                target_concat[0][i] = node
            for i, node in enumerate(target_genotype[3]):
                target_concat[1][i] = node
            for i, node in enumerate(surrogate_genotype[1]):
                surrogate_concat[0][i] = node
            for i, node in enumerate(surrogate_genotype[3]):
                surrogate_concat[1][i] = node
            return_data = {"target_arch": torch.tensor(target_arch), "surrogate_arch": torch.tensor(surrogate_arch), "label": torch.tensor(acc_adv_surrogate), 
            "target_concat": target_concat, "surrogate_concat": surrogate_concat}
        return return_data
