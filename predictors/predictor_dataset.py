import torch
from torch.utils.data import Dataset
import utils
from genotypes import Genotype
real_mean = 0.3192379331851974
real_std = 0.09124063405311836
supernet_mean = 0.19619138208364267
supernet_std = 0.0797107398048079
class PredictorDataSet(Dataset):
    def __init__(self, data_path, mode = "acc_adv_surrogate"):
        self.data = utils.load_yaml(data_path)
        self._mode = mode
        self.real_mean = real_mean
        self.real_std = real_std
        self.supernet_mean = supernet_mean
        self.supernet_std = supernet_std
    
    def set_mode(self, mode):
        self._mode = mode

    def __len__(self):#返回整个数据集的大小
        return len(self.data)
    
    def set_mean_std(self, real_mean, real_std, supernet_mean, supernet_std):
        self.real_mean = real_mean
        self.real_std = real_std
        self.supernet_mean = supernet_mean
        self.supernet_std = supernet_std

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
            "target_concat": target_concat, "surrogate_concat": surrogate_concat, "acc_adv_baseline": torch.tensor(self.data[index]["adversarial_accuracy"]["baseline"])}
        elif self._mode == "supernet_aided":
            target_genotype = eval(self.data[index]['target_genotype'])
            target_arch = utils.genotype_to_arch(target_genotype)
            surrogate_genotype = eval(self.data[index]['surrogate_genotype'])
            surrogate_arch = utils.genotype_to_arch(surrogate_genotype)
            acc_adv_surrogate = self.data[index]["adversarial_accuracy"]["surrogate"]
            acc_adv_supernet = self.data[index]["supernet"]["adversarial_accuracy"]["surrogate"]
            label = (acc_adv_surrogate - self.real_mean) / self.real_std - (acc_adv_supernet - self.supernet_mean) / self.supernet_std
            # label = (acc_adv_surrogate - self.real_mean * acc_adv_supernet / self.supernet_mean)
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
            return_data = {"target_arch": torch.tensor(target_arch), "surrogate_arch": torch.tensor(surrogate_arch), "label": torch.tensor(label), 
            "target_concat": target_concat, "surrogate_concat": surrogate_concat, 
            "acc_adv_surrogate":torch.tensor(acc_adv_surrogate), "acc_adv_supernet": torch.tensor(acc_adv_supernet),
            "acc_adv_baseline": torch.tensor(self.data[index]["adversarial_accuracy"]["baseline"])}
        return return_data
