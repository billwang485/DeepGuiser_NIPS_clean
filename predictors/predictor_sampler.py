import torch
import torch.utils
from torch.utils.data import Sampler
from typing import Iterator, Sequence

class PredictorSampler(Sampler[int]):
    indices: Sequence[int]

    def __init__(self, indices: Sequence[int], generator=None, batch_size = 4) -> None:
        self.indices = indices
        self.generator = generator
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[int]:
        for i in torch.randperm(int(len(self.indices) / self.batch_size), generator=self.generator):
            for j in torch.randperm(self.batch_size, generator=self.generator):
                yield self.indices[i * self.batch_size + j]

    def __len__(self) -> int:
        return len(self.indices)