import os
import re
import sys
import json
import time
import yaml
import shutil
import random
import difflib
import logging
import itertools
from copy import deepcopy
from collections import defaultdict
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as dset
from graphviz import Digraph
from PyPDF2 import PdfFileMerger
import numpy as np
import scipy.sparse as sp
import genotypes
from genotypes import Genotype
from genotypes import TRANSFORM_MASK_LOOSE_END, TRANSFORM_MASK_BOTTLENECK


def compute_nparam(module: nn.Module, size, arch_normal, arch_reduce, skip_pattern):
    def size_hook(module: nn.Module, input: torch.Tensor, output: torch.Tensor):
        module.param_hook = True

    hooks = []
    for name, m in module.named_modules():
        hooks.append(m.register_forward_hook(size_hook))
    with torch.no_grad():
        training = module.training
        module.eval()
        module._inner_forward(torch.rand(size, device=module._device), arch_normal, arch_reduce)
        module.train(mode=training)
    for hook in hooks:
        hook.remove()

    params = 0
    for name, m in module.named_modules():
        if skip_pattern in name:
            continue
        if isinstance(m, nn.Conv2d) and hasattr(m, "param_hook"):
            for p in m.parameters():
                params += p.numel()
            delattr(m, "param_hook")
        if isinstance(module, nn.Linear) and m.param_hook:
            for p in m.parameters():
                params += p.numel()
            delattr(m, "param_hook")

    return params


def compute_flops(module: nn.Module, size, arch_normal, arch_reduce, skip_pattern="null"):
    def size_hook(module: nn.Module, input: torch.Tensor, output: torch.Tensor):
        *_, h, w = output.shape
        module.output_size = (h, w)

    hooks = []
    for name, m in module.named_modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(size_hook))
    with torch.no_grad():
        training = module.training
        module.eval()
        module._inner_forward(torch.rand(size, device=module._device), arch_normal, arch_reduce)
        module.train(mode=training)
    for hook in hooks:
        hook.remove()

    flops = 0
    for name, m in module.named_modules():
        if skip_pattern in name:
            continue
        if isinstance(m, nn.Conv2d) and hasattr(m, "output_size"):
            h, w = m.output_size
            kh, kw = m.kernel_size
            flops += h * w * m.in_channels * m.out_channels * kh * kw / m.groups
            delattr(m, "output_size")
        if isinstance(module, nn.Linear):
            flops += m.in_features * m.out_features

    return flops


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def imagewise_accuracy(output, target, pid):

    res = {}
    r = zip(output, target, pid)
    for o, t, p in r:
        tokens = p.split("_")
        organ = tokens[0]
        prob = o[t]
        if organ in res:
            res[organ].append(float(prob >= 0.5))
        else:
            res[organ] = [float(prob >= 0.5)]

    result = {}
    all = 0
    n = 0
    for k, v in res.items():
        s = np.sum(v)
        m = np.mean(v)
        result[k] = m
        all += s
        n += len(v)
    all_mean = all / n
    result["all"] = all_mean
    return result


def subjectiwise_accuracy(output, target, pid):

    res = {}
    r = zip(output, target, pid)
    for key, value in itertools.groupby(r, key=lambda x: x[-1]):
        tokens = key.split("_")
        organ = tokens[0]
        patentID = tokens[1]
        prob = [p[l] for p, l, id in value]
        max_p = max(prob)
        min_p = min(prob)
        mean_p = sum(prob) / len(prob)
        if organ in res:
            res[organ].append([float(max_p >= 0.5), float(min_p >= 0.5), float(mean_p >= 0.5)])
        else:
            res[organ] = [[float(max_p >= 0.5), float(min_p >= 0.5), float(mean_p >= 0.5)]]

    result = {}
    all = np.zeros((3,))
    n = 0
    for k, v in res.items():
        v = np.array(v)
        s = np.sum(v, axis=0)
        m = np.mean(v, axis=0)
        result[k] = list(m)
        all += s
        n += len(v)
    all_mean = all / n
    result["all"] = list(all_mean)
    return result


def _data_transforms_mura(args):
    MURA_MEAN = [0.1524366]
    MURA_STD = [0.1807950]

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(512, padding=args.padding),
            transforms.RandomRotation(args.rotation),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MURA_MEAN, MURA_STD),
        ]
    )
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    valid_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MURA_MEAN, MURA_STD),])
    return train_transform, valid_transform


def _data_transforms_cifar10(args):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(),])
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    valid_transform = transforms.Compose([transforms.ToTensor(),])
    return train_transform, valid_transform


def _data_transforms_imagenet(args):

    train_transform = transforms.Compose([transforms.RandomResizedCrop(64), transforms.RandomHorizontalFlip(), transforms.ToTensor(),])
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    valid_transform = transforms.Compose([transforms.Resize(64), transforms.ToTensor(),])
    return train_transform, valid_transform


def count_parameters_in_MB(model):
    if isinstance(model, nn.DataParallel):
        return np.sum(np.prod(v.size()) for v in model.module.model_parameters()) / 1e6
    else:
        return np.sum(np.prod(v.size()) for v in model.model_parameters()) / 1e6


def count_parameters_woaux_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6


def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, "checkpoint.pth")
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, "model_best.pth")
        shutil.copyfile(filename, best_filename)

def save_as_json(model_dir, params, name='params.json'):
    """Save params to a .json file. Params is a dictionary of parameters."""
    path = os.path.join(model_dir, name)
    with open(path, 'w') as f:
        json.dump(params, f, indent=2, sort_keys=True)

# def save(model, model_path):
#     # torch.save(model.state_dict(), model_path)
#     model_dict = {"state_dict": model.state_dict()}
#     if hasattr(model, "arch_normal") and hasattr(model, "arch_reduce"):
#         model_dict["arch_normal"] = model.arch_normal
#         model_dict["arch_reduce"] = model.arch_reduce
#     torch.save(model_dict, model_path)
def save_compiled_based(model, save_path):
    model_dict = {"type": "compiled_based", "weight": model.state_dict(), "genotype":"{}".format(model._genotype), "C" : model._C, "layers": model._layers, "auxiliary": model._auxiliary}
    torch.save(model_dict, save_path)

def save_predictor_based_disguiser(model, save_path):
    model_dict = {"type": "predictor_based_disguiser", "predictor_state_dict": model.predictor.state_dict(), "arch_transformer_state_dict": model.arch_transformer.state_dict()}
    torch.save(model_dict, save_path)

def load_predictor_based_disguiser(model, save_path):
    model_dict = torch.load(save_path, map_location="cpu")
    model.predictor.load_state_dict(model_dict["predictor_state_dict"])
    model.arch_transformer.load_state_dict(model_dict["arch_transformer_state_dict"])

def save_predictor(model, save_path):
    model_dict = {"type": "predictor", "state_dict": model.state_dict()}
    torch.save(model_dict, save_path)

def load_predictor(model, save_path):
    model_dict = torch.load(save_path, map_location="cpu")
    if model.NAME == "predictor":
        model.load_state_dict(model_dict['state_dict'])
    

def save_supernet(integrated_model, save_path):
    assert hasattr(integrated_model, "stem") and hasattr(integrated_model, "cells")
    model_dict = {"type":"supernet", "stem": integrated_model.stem.state_dict(), "cells": integrated_model.cells.state_dict()}
    torch.save(model_dict, save_path)

def load_supernet(integrated_model, model_path):
    model_dict = torch.load(model_path, map_location="cpu")
    assert model_dict['type'] == 'supernet'
    assert hasattr(integrated_model, 'stem')
    integrated_model.stem.load_state_dict(model_dict['stem'])
    assert hasattr(integrated_model, 'cells')
    integrated_model.cells.load_state_dict(model_dict['cells'])

def save_nat_disguiser(integrated_model, save_path):
    supernet_dict = {"type":"supernet", "stem": integrated_model.stem.state_dict(), "cells": integrated_model.cells.state_dict()}
    model_dict = {"type":"nat_disguiser", "supernet": supernet_dict, "arch_transformer": integrated_model.arch_transformer.state_dict(), "arch_embedder": integrated_model.arch_transformer.arch_embedder.state_dict()}
    torch.save(model_dict, save_path)

def load_pretrained_arch_embedder(integrated_model, model_path):
    model_dict = torch.load(model_path, map_location="cpu")
    if model_dict['type'] == 'arch_embedder':
        integrated_model.arch_transformer.arch_embedder.load_state_dict(model_dict['arch_embedder'])
# def load(model, model_path, only_arch=False):
#     model_dict = torch.load(model_path, map_location="cpu")
#     if "state_dict" in model_dict:
#         if not only_arch:
#             model.load_state_dict(model_dict["state_dict"], strict=False)
#             if hasattr(model, "arch_normal") and hasattr(model, "arch_reduce"):
#                 model.arch_normal = model_dict["arch_normal"]
#                 model.arch_reduce = model_dict["arch_reduce"]
#                 model.single = True
#         else:
#             assert hasattr(model, "arch_normal") and hasattr(model, "arch_reduce")
#             model.arch_normal = model_dict["arch_normal"]
#             model.arch_reduce = model_dict["arch_reduce"]
#     else:
#         model.load_state_dict(model_dict)


def drop_path(x, drop_prob):
    if drop_prob > 0.0:
        keep_prob = 1.0 - drop_prob
        mask = torch.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob).to(x.device)
        x.div_(keep_prob)
        x.mul_(mask)
    return x


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print("Experiment dir : {}".format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, "scripts"))
        for script in scripts_to_save:
            dst_file = os.path.join(path, "scripts", os.path.basename(script))
            shutil.copyfile(script, dst_file)


def draw_genotype(genotype, n_nodes, filename, concat=None):
    """

    :param genotype:
    :param filename:
    :return:
    """
    g = Digraph(
        format="pdf",
        edge_attr=dict(fontsize="20", fontname="times"),
        node_attr=dict(style="filled", shape="rect", align="center", fontsize="20", height="0.5", width="0.5", penwidth="2", fontname="times"),
        engine="dot",
    )
    g.body.extend(["rankdir=LR"])

    g.node("-2", fillcolor="darkseagreen2")
    g.node("-1", fillcolor="darkseagreen2")
    steps = n_nodes

    for i in range(steps):
        g.node(str(i), fillcolor="lightblue")

    for op, source, target in genotype:
        if source == 0:
            u = "-2"
        elif source == 1:
            u = "-1"
        else:
            u = str(source - 2)
        v = str(target - 2)
        op = "null" if op == "none" else op
        # op = op.replace('dil_conv', 'dil_sep_conv') if 'dil_conv' in op else op
        g.edge(u, v, label=op, fillcolor="gray")

    g.node("out", fillcolor="palegoldenrod")
    if concat is not None:
        for i in concat:
            if i - 2 >= 0:
                g.edge(str(i - 2), "out", fillcolor="gray")
    else:
        for i in range(steps):
            g.edge(str(i), "out", fillcolor="gray")

    g.render(filename, view=False)

def draw_clean(genotype, save_path, name, steps = 4):
    draw_genotype(genotype.normal, steps, os.path.join(save_path, "normal"), genotype.normal_concat)
    draw_genotype(genotype.reduce, steps, os.path.join(save_path, "reduce"), genotype.reduce_concat)
    file_merger = PdfFileMerger()

    file_merger.append(os.path.join(save_path, "normal.pdf"))
    file_merger.append(os.path.join(save_path, "reduce.pdf"))

    file_merger.write(os.path.join(save_path,"{}.pdf".format(name)))

    os.remove(os.path.join(save_path, "normal.pdf"))
    os.remove(os.path.join(save_path, "normal"))
    os.remove(os.path.join(save_path, "reduce"))
    os.remove(os.path.join(save_path, "reduce.pdf"))

def arch_to_genotype(arch_normal, arch_reduce, n_nodes, cell_type, normal_concat=None, reduce_concat=None, hanag=False):
    try:
        primitives = eval("genotypes.{}".format(cell_type))
    except:
        assert False, "not supported op type %s" % (cell_type)

    if hanag:
        tmp = arch_normal[0]
        arch_reduce = arch_normal[1]
        arch_normal = tmp
        # (arch_normal, arch_reduce) = arch_normal

    gene_normal = [(primitives[op], f, t) for op, f, t in arch_normal]
    gene_reduce = [(primitives[op], f, t) for op, f, t in arch_reduce]
    if normal_concat is not None:
        _normal_concat = normal_concat
    else:
        _normal_concat = range(2, 2 + n_nodes)
    if reduce_concat is not None:
        _reduce_concat = reduce_concat
    else:
        _reduce_concat = range(2, 2 + n_nodes)
    genotype = Genotype(normal=gene_normal, normal_concat=_normal_concat, reduce=gene_reduce, reduce_concat=_reduce_concat)
    return genotype


def infinite_get(data_iter, data_queue):
    try:
        data = next(data_iter)
    except StopIteration:
        data_iter = iter(data_queue)
        data = next(data_iter)
    return data, data_iter


def get_variable(inputs, device, **kwargs):
    if type(inputs) in [list, np.ndarray]:
        inputs = torch.tensor(inputs)
    out = Variable(inputs.to(device), **kwargs)
    return out


def arch_to_string(arch):
    return ", ".join(["(op:%d,from:%d,to:%d)" % (o, f, t) for o, f, t in arch])


def get_index_item(inputs):
    if isinstance(inputs, torch.Tensor):
        inputs = int(inputs.item())
    return inputs


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def arch_to_matrix(arch):
    f_list = []
    t_list = []
    for _, f, t in arch:
        f_list.append(f)
        t_list.append(t)
    return np.array(f_list), np.array(t_list)


def parse_arch(arch, num_op):
    f_list, t_list = arch_to_matrix(arch)
    adj = sp.coo_matrix((np.ones(f_list.shape[0]), (t_list, f_list)), shape=(num_op, num_op), dtype=np.float32)
    adj = adj.multiply(adj > 0)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj


def sum_normalize(input):
    return input / torch.sum(input, -1, keepdim=True)


def convert_output(n_nodes, prev_nodes, prev_ops):
    """

    :param n_nodes: number of nodes
    :param prev_nodes: vector, each element is the node ID, int64, in the range of [0,1,...,n_node]
    :param prev_ops: vector, each element is the op_id, int64, in the range [0,1,...,n_ops-1]
    :return: arch list, (op, f, t) is the elements
    """
    assert len(prev_nodes) == 2 * n_nodes
    assert len(prev_ops) == 2 * n_nodes
    arch_list = []
    for i in range(n_nodes):
        t_node = i + 2
        f1_node = prev_nodes[i * 2].item()
        f2_node = prev_nodes[i * 2 + 1].item()
        f1_op = prev_ops[i * 2].item()
        f2_op = prev_ops[i * 2 + 1].item()
        arch_list.append((f1_op, f1_node, t_node))
        arch_list.append((f2_op, f2_node, t_node))
    return arch_list


def convert_lstm_output(n_nodes, prev_nodes, prev_ops):
    """

    :param n_nodes: number of nodes
    :param prev_nodes: vector, each element is the node ID, int64, in the range of [0,1,...,n_node]
    :param prev_ops: vector, each element is the op_id, int64, in the range [0,1,...,n_ops-1]
    :return: arch list, (op, f, t) is the elements
    """
    assert len(prev_nodes) == 2 * n_nodes
    assert len(prev_ops) == 2 * n_nodes
    arch_list = []
    for i in range(n_nodes):
        t_node = i + 2
        f1_node = prev_nodes[i * 2].item()
        f2_node = prev_nodes[i * 2 + 1].item()
        f1_op = prev_ops[i * 2].item()
        f2_op = prev_ops[i * 2 + 1].item()
        arch_list.append((f1_op, f1_node, t_node))
        arch_list.append((f2_op, f2_node, t_node))
    return arch_list


def translate_arch(arch, action, op_type="FULLY_CONCAT_PRIMITIVES"):
    # print(action)
    try:
        COMPACT_PRIMITIVES = eval("genotypes.{}".format(op_type))
    except:
        assert False, "not supported op type %s" % (op_type)
    arch_list = []
    for idx, (op, f, t) in enumerate(arch):
        f_op = op
        arch_list.append((action[idx], f, t))
    return arch_list


def genotype_to_arch(genotype, op_type="LOOSE_END_PRIMITIVES"):
    try:
        COMPACT_PRIMITIVES = eval("genotypes.{}".format(op_type))
    except:
        assert False, "not supported op type %s" % (op_type)
    arch_normal = [(COMPACT_PRIMITIVES.index(op), f, t) for op, f, t in genotype.normal]
    arch_reduce = [(COMPACT_PRIMITIVES.index(op), f, t) for op, f, t in genotype.reduce]
    return arch_normal, arch_reduce

def initialize_scheduler(optimizer, scheduler_config):
    if scheduler_config["type"] == "StepLR":
        return torch.optim.lr_scheduler.StepLR(optimizer, scheduler_config["step_size"], gamma=scheduler_config["gamma"])

def initialize_optimizer(parameter, learning_rate, momentum, weight_decay, _type = "Adam"):
    if _type == "Adam" and momentum == "default":
        return torch.optim.Adam(parameter, lr=learning_rate, weight_decay=weight_decay)
    
def str_diff_num(a, b):
    counter = 0
    for i, s in enumerate(difflib.ndiff(a, b)):
        if s[0] == " ":
            continue
        elif s[0] == "-" or s[0] == "+":
            counter += 1
    return int(counter / 2)


def concat_archs(arch1, arch2, op_type):
    arch = deepcopy(arch1)
    if op_type == "LOOSE_END_PRIMITIVES":
        TRANSFORM_MASK = TRANSFORM_MASK_LOOSE_END
    elif op_type == "BOTTLENECK_PRIMITIVES":
        TRANSFORM_MASK = TRANSFORM_MASK_BOTTLENECK
    ft = []
    for idx, (op, f, t) in enumerate(arch):
        ft.append((f, t))
    for idx, (op, f, t) in enumerate(arch2):
        if (f, t) in ft and TRANSFORM_MASK[arch[ft.index((f, t))][0]][op]:
            arch[ft.index((f, t))] = (op, f, t)
    return arch


def primitives_translation(op_type="LOOSE_END_PRIMTIVES"):
    op_list = eval("genotypes.{}".format(op_type))
    null_index = op_list.index("null")
    gate_op_list = ["None"]
    transform2gates = [0] * len(op_list)
    transform2gates[null_index] = 0
    transform2nat = [0] * len(op_list)
    transform2nat[0] = null_index
    for i, op in enumerate(op_list):
        if i == null_index:
            continue
        gate_op_list.append(op)
        transform2gates[i] = len(gate_op_list) - 1
        transform2nat[transform2gates[i]] = i

    assert max(transform2gates) == len(op_list) - 1

    return transform2gates, transform2nat, gate_op_list


def nat_arch_to_gates(nat_normal, nat_reduce, transform2gates, batch_size=1):
    nat_normal.sort(key=lambda x: x[-1])
    nat_reduce.sort(key=lambda x: x[-1])
    gates_normal = [[], []]
    gates_reduce = [[], []]
    for i, (op, f, t) in enumerate(nat_normal):
        gates_normal[0].append(f)
        gates_normal[1].append(transform2gates[op])

    for i, (op, f, t) in enumerate(nat_reduce):
        gates_reduce[0].append(f)
        gates_reduce[1].append(transform2gates[op])

    return [gates_normal, gates_reduce]

def data_point_2_cpu(data_point):
    return_data = {"target_arch": data_point["target_arch"].cpu(), "surrogate_arch": data_point["surrogate_arch"].cpu(), "label": data_point["label"].cpu()}
    torch.cuda.empty_cache()
    return return_data

def nat_arch_to_gates_p(nat_normal, nat_reduce, transform2gates, batch_size=1):
    # nat_normal.sort(key=lambda x: x[-1])
    # nat_reduce.sort(key=lambda x: x[-1])
    batch_size = nat_normal.shape[0]
    gates = []
    for i in range(batch_size):
        nat_normal_ = nat_normal[i]
        nat_reduce_ = nat_reduce[i]
        gates_normal = [[], []]
        gates_reduce = [[], []]
        for i, (op, f, t) in enumerate(nat_normal_):
            gates_normal[0].append(f)
            gates_normal[1].append(transform2gates[op])

        for i, (op, f, t) in enumerate(nat_reduce_):
            gates_reduce[0].append(f)
            gates_reduce[1].append(transform2gates[op])
        gates.append([gates_normal, gates_reduce])

    return torch.tensor(gates, dtype=torch.long)


def BinarySoftmax(X, V):
    X = X - max(X * V)
    X = torch.clamp(X, min=-100, max=1)
    X_exp_bi = X.exp() * V
    partition = X_exp_bi.sum(dim=-1, keepdim=True) + 1e-5
    return X_exp_bi / partition


def make_one_hot(label_mat, primitive_num, op_num):
    one_hot_label_mat = torch.zeros(primitive_num, op_num)
    for i, label in enumerate(label_mat):
        one_hot_label_mat[label[0]][i] = 1.0
    return one_hot_label_mat.requires_grad_()


def imitation_loss(label_normal, label_reduce, probs_normal, probs_reduce, device):
    primitive_num, op_num = probs_normal.shape
    loss_function = nn.CrossEntropyLoss()
    normal_mat = make_one_hot(label_normal, primitive_num, op_num).to(device)
    reduce_mat = make_one_hot(label_reduce, primitive_num, op_num).to(device)
    loss = torch.zeros(1, requires_grad=True, device=device)
    for x in range(op_num):
        loss = loss + loss_function(probs_normal[:, x].unsqueeze(dim=0), normal_mat[:, x].unsqueeze(dim=0)) + loss_function(probs_reduce[:, x].unsqueeze(dim=0), reduce_mat[:, x].unsqueeze(dim=0))
    loss = loss / 50
    return loss.to(device)



def update_arch(best_pair_list, arch_normal, arch_reduce, optimized_normal, optimized_reduce, reward, acc_clean, acc_adv, optimized_acc):
    tmp = {}
    tmp["reward"] = deepcopy(reward)
    tmp["target_arch"] = [deepcopy(arch_normal), deepcopy(arch_reduce)]
    tmp["surrogate_arch"] = [deepcopy(optimized_normal), deepcopy(optimized_reduce)]
    tmp["acc_clean"] = deepcopy(acc_clean)
    tmp["acc_adv"] = deepcopy(acc_adv)
    tmp["optimized_acc"] = deepcopy(optimized_acc)
    best_pair_list.append(deepcopy(tmp))


def check_transform(arch_original, arch_transform, op_type="LOOSE_END_PRIMITIVES"):
    # from genotypes import LooseEnd_Transition_Dict, FullyConcat_Transition_Dict
    COMPACT_PRIMITIVES = eval("genotypes.{}".format(op_type))
    transition_dict = genotypes.LooseEnd_Transition_Dict if op_type == "LOOSE_END_PRIMITIVES" else None
    assert transition_dict != None
    for i, ((op_1, f_1, t_1), (op_2, f_2, t_2)) in enumerate(zip(arch_original, arch_transform)):
        flag = [False, False, False]
        select_op = transition_dict[COMPACT_PRIMITIVES[op_1]]
        if COMPACT_PRIMITIVES[op_2] in select_op:
            flag[0] = True
        if f_1 == f_2:
            flag[1] = True
        if t_1 == t_2:
            flag[2] = True
        if not all(flag):
            # assert 0, 'transform doesn\'t match trnasition_dict'
            break
    return all(flag)


def z_load(path):
    return torch.load(path, map_location="cpu")


def op_diversity(arch):
    op_list = [0] * 9
    for i, (op, f, t) in enumerate(arch):
        op_list[op] = op_list[op] + 1
    op_list = op_list / len(arch)
    entropy = 0
    for i, prob in op_list:
        entropy = entropy + prob * np.log(prob)
    return entropy


def patk(true_scores, predict_scores, k=10):
    true_inds = np.argsort(true_scores)[::-1]
    true_scores = np.array(true_scores)
    reorder_true_scores = true_scores[true_inds]
    predict_scores = np.array(predict_scores)
    reorder_predict_scores = predict_scores[true_inds]
    ranks = np.argsort(reorder_predict_scores)[::-1]
    num_archs = len(ranks)
    # calculate precision at each point
    cur_inds = np.zeros(num_archs)
    passed_set = set()
    for i_rank, rank in enumerate(ranks):
        cur_inds[i_rank] = (cur_inds[i_rank - 1] if i_rank > 0 else 0) + int(i_rank in passed_set) + int(rank <= i_rank)
        passed_set.add(rank)
    patks = cur_inds / (np.arange(num_archs) + 1)
    # THRESH = 1000
    # p_corrs = []
    # for prec in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
    #     k = np.where(patks[THRESH:] >= prec)[0][0] + THRESH
    #     arch_inds = ranks[:k][ranks[:k] < k]
    #     p_corrs.append((k, float(k)/num_archs, len(arch_inds), prec, stats.kendalltau(
    #         reorder_true_scores[arch_inds],
    #         reorder_predict_scores[arch_inds]).correlation))
    return patks[k - 1]


def compare_data(archs, accs, max_compare_ratio=4.0, compare_threshold=0.0):
    n_max_pairs = int(max_compare_ratio * len(archs))
    acc_diff = np.array(accs)[:, None] - np.array(accs)
    acc_abs_diff_matrix = np.triu(np.abs(acc_diff), 1)
    ex_thresh_inds = np.where(acc_abs_diff_matrix > compare_threshold)
    ex_thresh_num = len(ex_thresh_inds[0])
    if ex_thresh_num > n_max_pairs:
        keep_inds = np.random.choice(np.arange(ex_thresh_num), n_max_pairs, replace=False)
        ex_thresh_inds = (ex_thresh_inds[0][keep_inds], ex_thresh_inds[1][keep_inds])
    archs_1, archs_2, better_lst = archs[ex_thresh_inds[1]], archs[ex_thresh_inds[0]], (acc_diff > 0)[ex_thresh_inds]

    return archs_1, archs_2, better_lst


def get_tim_data(args, seed=1234):
    train_transform, valid_transform = _data_transforms_imagenet(args)
    # print(os.path.join(args.data, 'train'))

    train_data = dset.ImageFolder(root=os.path.join(args.data, "train"), transform=train_transform)
    test_data = dset.ImageFolder(root=os.path.join(args.data, "val"), transform=valid_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    indices_test = list(range(len(test_data)))
    random.seed(seed)
    random.shuffle(indices)
    random.shuffle(indices_test)
    random.seed(args.seed)

    test_queue = torch.utils.data.DataLoader(test_data, batch_size=32, sampler=torch.utils.data.sampler.SubsetRandomSampler(indices_test), pin_memory=True, num_workers=2)

    train_queue = torch.utils.data.DataLoader(
        train_data,
        args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices),
        # shuffle= False,
        pin_memory=True,
        num_workers=2,
    )
    return train_queue, test_queue


def get_final_test_data(args, CIFAR_CLASSES=10, seed=1234):
    train_transform, valid_transform = _data_transforms_cifar10(args)
    if CIFAR_CLASSES == 10:

        train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

        test_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)
    else:
        train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)

        test_data = dset.CIFAR100(root=args.data, train=False, download=True, transform=valid_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    indices_test = list(range(len(test_data)))
    random.seed(seed)
    random.shuffle(indices)
    random.shuffle(indices_test)
    random.seed(args.seed)

    test_queue = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, sampler=torch.utils.data.sampler.SubsetRandomSampler(indices_test), pin_memory=True, num_workers=2)

    train_queue = torch.utils.data.DataLoader(
        train_data,
        args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices),
        # shuffle= False,
        pin_memory=True,
        num_workers=2,
    )
    return train_queue, test_queue


def get_cifar_data_queue(args, seed=1234):
    ''' get_train_queue'''

    train_transform, valid_transform = _data_transforms_cifar10(args)

    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

    split = int(np.floor(0.8 * len(train_data)))

    test_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    indices_test = list(range(len(test_data)))
    random.seed(seed)
    random.shuffle(indices)
    random.shuffle(indices_test)
    random.seed(args.seed)

    test_queue = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, sampler=torch.utils.data.sampler.SubsetRandomSampler(indices_test), pin_memory=True, num_workers=2)

    train_queue = torch.utils.data.DataLoader(
        train_data,
        args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        # shuffle= False,
        pin_memory=True,
        num_workers=2,
    )

    if not hasattr(args, "accu_batch"):
        valid_queue = torch.utils.data.DataLoader(train_data, args.batch_size, sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]), pin_memory=True, num_workers=2)
    else:
        valid_queue = torch.utils.data.DataLoader(train_data, args.batch_size, sampler=torch.utils.data.sampler.SequentialSampler(indices[split:num_train]), pin_memory=True, num_workers=2)

    return train_queue, valid_queue, test_queue


def check_connectivity(arch, num_nodes=4, op_num=9):
    connnectivity = []
    for _ in range(6):
        connnectivity.append(list())
    for step, (op, f, t) in enumerate(arch):
        if op != op_num - 1:
            connnectivity[t].append(f)
    flag = 0
    flag = check_connnectivity_(connnectivity, 5)
    return flag


def get_connectivity(arch, reverse=False, op_num=9):
    connnectivity = []
    for _ in range(6):
        connnectivity.append(list())
    for step, (op, f, t) in enumerate(arch):
        if op != op_num - 1:
            connnectivity[f].append(t)
    return connnectivity


def check_connectivity_transform(connnectivity, from_, num_nodes=4):
    if from_ == 5:
        return True

    if len(connnectivity[from_]) == 0:
        return False
    else:
        flag = False
        for t in connnectivity[from_]:
            if check_connectivity_transform(connnectivity, t):
                flag = True
                break
        return flag


def check_connnectivity_(connectivity, t_, f_=-1):
    if f_ != -1:
        if t_ == f_:
            return True
        else:
            flag = False
            for f in connectivity[t_]:
                if check_connnectivity_(connectivity, f, f_):
                    flag = True
                    break
            return flag
    if t_ == 0 or t_ == 1:
        return True
    else:
        flag = False
        for f in connectivity[t_]:
            if check_connnectivity_(connectivity, f):
                flag = True
                break
        return flag


def transform_times(arch1, arch2):
    count = 0
    for i, (op, f, t) in enumerate(arch1):
        assert f == arch2[i][1] and t == arch2[i][2]
        if op != arch2[i][0]:
            count = count + 1
    return count


def gradient_wrt_input(model, arch_normal, arch_reduce, inputs, targets, criterion=nn.CrossEntropyLoss()):
    inputs.requires_grad = True

    outputs = model._inner_forward(inputs, arch_normal, arch_reduce)
    loss = criterion(outputs, targets)
    model.zero_grad()
    loss.backward()

    data_grad = inputs.grad.data
    return data_grad.clone().detach()


def linf_pgd(model, arch_normal, arch_reduce, dat, lbl, eps, alpha, steps, is_targeted=False, rand_start=True, momentum=False, mu=1, criterion=nn.CrossEntropyLoss()):
    x_nat = dat.clone().detach()
    x_adv = None
    if rand_start:
        x_adv = dat.clone().detach() + torch.FloatTensor(dat.shape).uniform_(-eps, eps).cuda()
    else:
        x_adv = dat.clone().detach()
    x_adv = torch.clamp(x_adv, 0.0, 1.0)  # respect image bounds
    g = torch.zeros_like(x_adv)

    # Iteratively Perturb data
    for i in range(steps):
        # Calculate gradient w.r.t. data
        grad = gradient_wrt_input(model, arch_normal, arch_reduce, x_adv, lbl, criterion)
        with torch.no_grad():
            if momentum:
                # Compute sample wise L1 norm of gradient
                flat_grad = grad.view(grad.shape[0], -1)
                l1_grad = torch.norm(flat_grad, 1, dim=1)
                grad = grad / torch.clamp(l1_grad, min=1e-12).view(grad.shape[0], 1, 1, 1)
                # Accumulate the gradient
                new_grad = mu * g + grad  # calc new grad with momentum term
                g = new_grad
            else:
                new_grad = grad
            # Get the sign of the gradient
            sign_data_grad = new_grad.sign()
            if is_targeted:
                x_adv = x_adv - alpha * sign_data_grad  # perturb the data to MINIMIZE loss on tgt class
            else:
                x_adv = x_adv + alpha * sign_data_grad  # perturb the data to MAXIMIZE loss on gt class
            # Clip the perturbations w.r.t. the original data so we still satisfy l_infinity
            # x_adv = torch.clamp(x_adv, x_nat-eps, x_nat+eps) # Tensor min/max not supported yet
            x_adv = torch.max(torch.min(x_adv, x_nat + eps), x_nat - eps)
            # Make sure we are still in bounds
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv.clone().detach()

def load_yaml(yaml_path):
    with open(yaml_path) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data

def save_yaml(save_dict, yaml_path):
    with open(yaml_path, 'w') as f:
        yaml.dump(save_dict, f)

class NormalizeByChannelMeanStd(nn.Module):
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        self.mean = mean.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        self.std = std.unsqueeze(0).unsqueeze(2).unsqueeze(3)

    def forward(self, x):
        assert x.shape[1] == self.mean.shape[1]
        assert x.shape[1] == self.std.shape[1]

        return (x - self.mean) / self.std


def localtime_as_dirname():
    localtime = time.asctime(time.localtime(time.time()))
    x = re.split(r"[\s,(:)]", localtime)
    default_EXP = " ".join(x[1:-1])
    return default_EXP


def preprocess_exp_dir(args, name = "log"):
    if not os.path.exists(args.prefix):
        os.makedirs(args.prefix)
    log_dir = name
    if args.debug:
        log_dir = os.path.join(log_dir, "debug")
    if not os.path.exists(os.path.join(args.prefix, log_dir)):
        os.makedirs(os.path.join(args.prefix, log_dir))
    args.save = os.path.join(args.prefix, log_dir, args.save)


def initialize_logger(args):
    log_format = "%(asctime)s %(message)s"
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt="%m/%d %I:%M:%S %p")
    fh = logging.FileHandler(os.path.join(args.save, "log.txt"))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logger = logging.getLogger()
    return logger

def unified_forward(model, inputs):
    if model.model_type == "compiled_base":
        logits, _ = model(inputs)
    else:
        logits = model(inputs)
    
    return logits

def train_model(model, train_queue, device, criterion, optimizer, scheduler, epochs, logger = None):
    model.to(device)
    for epoch in range(epochs):
        
        lr = scheduler.get_lr()[0]
        if logger is not None:
            logger.info('epoch %d lr %e', epoch, lr)
        top1 = AvgrageMeter()
        top5 = AvgrageMeter()
        objs = AvgrageMeter()
        for step, (input, target) in enumerate(train_queue):
            model.train()
            optimizer.zero_grad()
            n = input.size(0)
            input = input.to(device)
            target = target.to(device)
            logits = unified_forward(model, input)
            
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            prec1, prec5 = accuracy(logits, target, topk=(1, 5))
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)
            if step % 100 == 0:
                logging.info('Step=%03d Loss=%e Top1=%f Top5=%f', step, objs.avg, top1.avg, top5.avg)
        scheduler.step()
    
    logging.info('Training Finished: Loss=%e Top1=%f Top5=%f', objs.avg, top1.avg, top5.avg)

def test_clean_accuracy(model, test_queue, logger = None,  device = None, ):
    if device is not None:
        model.to(device)
    else:
        device = next(model.parameters()).device
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()
    for step, (input, target) in enumerate(test_queue):
        model.eval()
        n = input.size(0)
        input = input.to(device)
        target = target.to(device)
        logits = unified_forward(model, input)
        prec1, prec5 = accuracy(logits, target, topk=(1, 5))
        top1.update(prec1.item() / 100.0, n)
        top5.update(prec5.item() / 100.0, n)
    if logger is not None:
        logger.info('Test Accuracy: Top1=%f Top5=%f', top1.avg, top5.avg)
    return top1.avg, top5.avg

def compiled_pgd_test(target_model, surrogate_model, baseline_model, test_queue, attack_info, logger = None):
    assert attack_info['type'] == 'PGD'
    device = next(target_model.parameters()).device
    acc_adv_baseline = AvgrageMeter()
    acc_adv_surrogate = AvgrageMeter()

    for step, (input, target) in enumerate(test_queue):
        n = input.size(0)
        input = input.to(device)
        target = target.to(device)
        target_model.eval()
        baseline_model.eval()
        surrogate_model.eval()

        input_adv0 = _generate_pgd_input(baseline_model, input, target, attack_info['eps'], attack_info['alpha'], attack_info['step'])
        # logits0, _ = target_model(input_adv0)
        logits0 = unified_forward(target_model, input_adv0)
        acc_adv_baseline_ = accuracy(logits0, target, topk=(1, 5))[0] / 100.0
        acc_adv_baseline.update(acc_adv_baseline_.item(), n)

        input_adv1 = _generate_pgd_input(surrogate_model, input, target, attack_info['eps'], attack_info['alpha'], attack_info['step'])
        # logging.info("acc_adv_target_white=%.2f", acc_adv.item())
        # logits1, _ = target_model(input_adv1)
        logits1 = unified_forward(target_model, input_adv1)
        acc_adv_surrogate_ = accuracy(logits1, target, topk=(1, 5))[0] / 100.0
        acc_adv_surrogate.update(acc_adv_surrogate_.item(), n)
    if logger is not None:
        logger.info('PGD Test Results: acc_adv_baseline=%f acc_adv_surrogate=%.2f',\
                acc_adv_baseline.avg, acc_adv_surrogate.avg)
    return acc_adv_baseline.avg, acc_adv_surrogate.avg

def _generate_pgd_input(generator_model, input, target, eps, alpha, steps, is_targeted=False, rand_start=True, momentum=False, mu=1, criterion=nn.CrossEntropyLoss()):
    def _gradient_wrt_input(model, inputs, targets, criterion=nn.CrossEntropyLoss()):
        inputs.requires_grad = True

        outputs = unified_forward(model ,inputs)
        loss = criterion(outputs, targets)
        model.zero_grad()
        loss.backward()

        data_grad = inputs.grad.data
        return data_grad.clone().detach()
    generator_model.eval()
    x_nat = input.clone().detach()
    x_adv = None
    if rand_start:
        x_adv = input.clone().detach() + torch.FloatTensor(input.shape).uniform_(-eps, eps).cuda()
    else:
        x_adv = input.clone().detach()
    x_adv = torch.clamp(x_adv, 0.0, 1.0)  # respect image bounds
    g = torch.zeros_like(x_adv)

    # Iteratively Perturb data
    for i in range(int(steps)):
        # Calculate gradient w.r.t. data
        # print("hhhh")
        grad = _gradient_wrt_input(generator_model, x_adv, target, criterion)
        with torch.no_grad():
            if momentum:
                # Compute sample wise L1 norm of gradient
                flat_grad = grad.view(grad.shape[0], -1)
                l1_grad = torch.norm(flat_grad, 1, dim=1)
                grad = grad / torch.clamp(l1_grad, min=1e-12).view(grad.shape[0], 1, 1, 1)
                # Accumulate the gradient
                new_grad = mu * g + grad  # calc new grad with momentum term
                g = new_grad
            else:
                new_grad = grad
            # Get the sign of the gradient
            sign_data_grad = new_grad.sign()
            if is_targeted:
                x_adv = x_adv - alpha * sign_data_grad  # perturb the data to MINIMIZE loss on tgt class
            else:
                x_adv = x_adv + alpha * sign_data_grad  # perturb the data to MAXIMIZE loss on gt class
            # Clip the perturbations w.r.t. the original data so we still satisfy l_infinity
            # x_adv = torch.clamp(x_adv, x_nat-eps, x_nat+eps) # Tensor min/max not supported yet
            x_adv = torch.max(torch.min(x_adv, x_nat + eps), x_nat - eps)
            # Make sure we are still in bounds
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv.clone().detach()

class AvgrageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1:y2, x1:x2] = 0.0
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


class keydefaultdict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret
