import torch
from torch.utils.data import Dataset

class PredictorDataSet(Dataset):
    def __init__(self, free_pair_path, constrain_pair_path, args):
        if args.mode == 'low_fidelity':
            self.free_data = torch.load(free_pair_path, map_location='cpu')
            self.constrain_data = torch.load(constrain_pair_path, map_location='cpu')
            self.constrain_data.extend(self.free_data)
        elif args.mode == 'high_fidelity':
            self.constrain_data = torch.load(constrain_pair_path, map_location='cpu')
        else:
            assert 0       
        self.data = self.constrain_data

    def __len__(self):#返回整个数据集的大小
        return len(self.data)

    def __getitem__(self,index):
        return self.data[index]
