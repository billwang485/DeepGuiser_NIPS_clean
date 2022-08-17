
import os
import torch

name = 'random_concat'
x = 'random_transform/Mar 11 16 44 25/archs'
num = 1
archs = torch.load(x)
arch_list = []
for i, arch in enumerate(archs):
    if i >= num:
        break
    arch_list.append((i, 0, arch[0].item(), arch[1], arch[2]))
torch.save(arch_list, os.path.join(os.getcwd(), name))