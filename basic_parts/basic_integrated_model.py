import genotypes
from operations import *
from predictors.predictors import VanillaGatesPredictor
import utils
import numpy as np
from utils import Linf_PGD
from copy import deepcopy
from basic_parts.basic_nat_models import NASCell, ArchMaster
from utils import NormalizeByChannelMeanStd
from basic_parts.basic_arch_transformers import ArchTransformerGates, ArchTransformer


class NASNetwork(nn.Module):
    def __init__(
        self,
        C,
        num_classes,
        layers,
        criterion,
        device,
        steps=4,
        stem_multiplier=3,
        controller_hid=None,
        entropy_coeff=[0.0, 0.0],
        edge_hid=100,
        transformer_nfeat=1024,
        transformer_nhid=512,
        transformer_dropout=0,
        transformer_normalize=False,
        loose_end=False,
        normal_concat=None,
        reduce_concat=None,
        op_type="FULLY_CONCAT_PRIMITIVES",
    ):
        super(NASNetwork, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        multiplier = steps
        self._device = device

        self.controller_hid = controller_hid
        self.entropy_coeff = entropy_coeff

        self.edge_hid = edge_hid
        self.transformer_nfeat = transformer_nfeat
        self.transformer_nhid = transformer_nhid
        self.transformer_dropout = transformer_dropout
        self.transformer_normalize = transformer_normalize
        self.op_type = op_type

        self.loose_end = loose_end

        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(nn.Conv2d(3, C_curr, 3, padding=1, bias=False), nn.BatchNorm2d(C_curr))

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        _concat = None
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
                if reduce_concat is not None:
                    _concat = reduce_concat
            else:
                reduction = False
                if normal_concat is not None:
                    _concat = normal_concat
            cell = NASCell(steps, device, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, loose_end=loose_end, concat=_concat, op_type=self.op_type)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        self._initialize_arch_master()
        # self._initialize_arch_transformer()
        # self.initialize_step()
        self.best_pair = []
        self.accu_batch = 1
        self.reward_type = "absolute"
        self.single = False
        mean = torch.tensor([0.4914, 0.4822, 0.4465], dtype=torch.float32).cuda()
        std = torch.tensor([0.2023, 0.1994, 0.2010], dtype=torch.float32).cuda()
        self.normalizer = NormalizeByChannelMeanStd(mean=mean, std=std)
        self.count = 0
        self.reward = utils.AvgrageMeter()
        self.optimized_acc_adv = utils.AvgrageMeter()
        self.acc_adv = utils.AvgrageMeter()
        self.acc_clean = utils.AvgrageMeter()
        self.pgd_step = 10
        self.tiny_imagenet = False

    def initialize_tiny_imagenet(self):
        self.tiny_imagenet = True
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).cuda()
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).cuda()
        self.normalizer = NormalizeByChannelMeanStd(mean=mean, std=std)

    def re_initialize_arch_transformer(self):
        # self.arch_transformer = New_ArchTransformer(self._steps, self._device, self.edge_hid, self.transformer_nfeat, self.transformer_nhid, self.transformer_dropout, self.transformer_normalize, op_type=self.op_type)
        self.arch_transformer = ArchTransformerGates(
            self._steps, self._device, self.edge_hid, self.transformer_nfeat, self.transformer_nhid, self.transformer_dropout, self.transformer_normalize, op_type=self.op_type
        )
        self._transformer_parameters = list(self.arch_transformer.parameters())

    def _initialize_predictor(self, args, name):
        if name == "WarmUp":
            self.predictor = VanillaGatesPredictor(self._device, args)
        elif name == "NAT":
            assert 0

    def _initialize_arch_master(self):
        try:
            COMPACT_PRIMITIVES = eval("genotypes.%s" % self.op_type)
        except:
            assert False, "not supported op type %s" % (self.op_type)

        num_ops = len(COMPACT_PRIMITIVES) - 1
        self.arch_normal_master = ArchMaster(num_ops, self._steps, self._device)
        self.arch_reduce_master = ArchMaster(num_ops, self._steps, self._device)
        self.arch_normal_master_demo = ArchMaster(num_ops + 1, self._steps, self._device)
        self.arch_reduce_master_demo = ArchMaster(num_ops + 1, self._steps, self._device)
        self.arch_normal_master_demo.demo = True
        self.arch_reduce_master_demo.demo = True
        self._arch_parameters = list(self.arch_normal_master.parameters()) + list(self.arch_reduce_master.parameters())

    def _initialize_arch_transformer(self):
        self.arch_transformer = ArchTransformer(
            self._steps, self._device, self.edge_hid, self.transformer_nfeat, self.transformer_nhid, self.transformer_dropout, self.transformer_normalize, op_type=self.op_type
        )
        self._transformer_parameters = list(self.arch_transformer.parameters())

    def _inner_forward(self, input, arch_normal, arch_reduce):
        input = self.normalizer(input)

        s0 = self.stem(input)
        s1 = s0
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                archs = arch_reduce
            else:
                archs = arch_normal
            s0, s1 = s1, cell(s0, s1, archs)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

    def _test_acc(self, test_queue, arch_normal, arch_reduce):
        # go over all the testing data to obtain the accuracy
        top1 = utils.AvgrageMeter()
        for step, (test_input, test_target) in enumerate(test_queue):
            test_input = test_input.to(self._device)
            test_target = test_target.to(self._device)
            n = test_input.size(0)
            logits = self._inner_forward(test_input, arch_normal, arch_reduce)
            accuracy = utils.accuracy(logits, test_target)[0]
            top1.update(accuracy.item(), n)
        return top1.avg

    def test(self, test_queue, n_optim, logger, folder, suffix):
        arch_normal = self.arch_normal_master()
        arch_reduce = self.arch_reduce_master()
        self.derive_optimized_arch(test_queue, arch_normal, arch_reduce, n_optim, logger, folder, suffix)

    def derive_optimized_arch(self, model_twin, test_queue, arch_normal, arch_reduce, n_optim, logger, folder, suffix, normal_concat=None, reduce_concat=None):
        best_optimized_acc_adv = -np.inf
        best_optimized_acc_clean = -np.inf
        # best_arch_normal = None
        # best_arch_reduce = None
        best_optimized_arch_normal = None
        best_optimized_arch_reduce = None

        acc_clean_baseline, acc_adv_baseline = self._test_transfer(model_twin, test_queue, arch_normal, arch_reduce, arch_normal, arch_reduce)
        logger.info("Sampling candidate architectures ...")
        for i in range(n_optim):
            optimized_normal, optimized_reduce, optimized_logP, optimized_entropy = self.arch_transformer.forward(arch_normal)
            optimized_acc_clean, optimized_acc_adv = self._test_transfer(model_twin, test_queue, arch_normal, arch_reduce, optimized_normal, optimized_reduce)
            if optimized_acc_adv > best_optimized_acc_adv:
                best_optimized_acc_adv = optimized_acc_adv
                best_optimized_acc_clean = optimized_acc_clean
                best_optimized_arch_normal = optimized_normal
                best_optimized_arch_reduce = optimized_reduce
                best_arch_logP = optimized_logP
                best_arch_ent = optimized_entropy
        logger.info("Target: acc_clean = %.2f acc_adv = %.2f", acc_clean_baseline, acc_adv_baseline)
        logger.info("Best surrogate: acc_clean = %.2f acc_adv = %.2f", best_optimized_acc_clean, best_optimized_acc_adv)
        logger.info("Absolute reward = %.2f Relative reward = %.2f", best_optimized_acc_adv - acc_adv_baseline, (best_optimized_acc_adv - acc_adv_baseline) / (acc_clean_baseline - acc_adv_baseline))

        result = {}
        result["target_arch"] = (deepcopy(arch_normal), deepcopy(arch_reduce))
        result["surrogate_arch"] = (deepcopy(best_optimized_arch_normal), deepcopy(best_optimized_arch_reduce))
        result["absolute_supernet_reward"] = best_optimized_acc_adv - acc_adv_baseline
        result["relative_supernet_reward"] = (best_optimized_acc_adv - acc_adv_baseline) / (acc_clean_baseline - acc_adv_baseline)
        result["acc_clean_baseline"] = acc_clean_baseline
        result["acc_adv_baseline"] = acc_adv_baseline
        result["best_optimized_acc_adv"] = best_optimized_acc_adv
        result["best_optimized_acc_clean"] = best_optimized_acc_clean
        result["best_arch_logP"] = best_arch_logP
        result["best_arch_ent"] = best_arch_ent
        return result

    def random_transform(self, arch, step=1, reduce=False):
        arch_ = deepcopy(arch)
        if reduce:
            tmp_transformer = self.arch_reduce_master
        else:
            tmp_transformer = self.arch_normal_master
        for i in range(step):
            tmp = tmp_transformer.forward()
            arch_ = utils.concat_archs(arch_, tmp, self.op_type)
        return arch_

    def uni_random_transform(self, arch, budget=100):
        arch_ = deepcopy(arch)
        COMPACT_PRIMITIVES = eval("genotypes.{}".format(self.op_type))
        transition_dict = genotypes.LooseEnd_Transition_Dict if self.op_type == "LOOSE_END_PRIMITIVES" else None
        assert transition_dict != None
        arch_1 = []
        for i, (op, f, t) in enumerate(arch):
            select_op = transition_dict[COMPACT_PRIMITIVES[op]]
            probs = F.softmax(torch.zeros(len(select_op)).to(self._device), dim=-1)
            tmp = probs.multinomial(num_samples=1)
            op_1 = COMPACT_PRIMITIVES.index(select_op[tmp])
            if np.random.randint(0, budget) > 10:
                arch_1.append((op_1, f, t))
            else:
                arch_1.append((op, f, t))
        utils.check_transform(arch, arch_1, self.op_type)
        return arch_1

    def transformer_forward(self, valid_input):
        arch_normal = self.arch_normal_master.forward()
        arch_reduce = self.arch_reduce_master.forward()
        optimized_normal, optimized_normal_logP, optimized_normal_entropy = self.arch_transformer.forward(arch_normal)
        optimized_reduce, optimized_reduce_logP, optimized_reduce_entropy = self.arch_transformer.forward(arch_reduce)
        logits = self._inner_forward(valid_input, arch_normal, arch_reduce)
        optimized_logits = self._inner_forward(valid_input, optimized_normal, optimized_reduce)
        return (logits, optimized_logits, optimized_normal, optimized_normal_logP, optimized_normal_entropy, optimized_reduce, optimized_reduce_logP, optimized_reduce_entropy)

    def step(self, valid_input, valid_target):
        arch_normal = self.arch_normal_master.forward()
        arch_reduce = self.arch_reduce_master.forward()
        self._model_optimizer.zero_grad()
        logits = self._inner_forward(valid_input, arch_normal, arch_reduce)
        loss = self._criterion(logits, valid_target)
        loss.backward()
        self._model_optimizer.step()
        return logits, loss

    def initialize_step(self):
        self.count = 0
        self.reward = utils.AvgrageMeter()
        self.optimized_acc_adv = utils.AvgrageMeter()
        self.acc_adv = utils.AvgrageMeter()
        self.acc_clean = utils.AvgrageMeter()
        self.arch_normal = None
        self.arch_reduce = None
        self.optimized_normal = None
        self.optimized_reduce = None

    def evaluate_transfer(self, model_twin, target_arch, surrogate_arch, input, target, eps=0.031, steps=10):
        if self.single:
            self.eval()
            model_twin.eval()
        optimized_normal = surrogate_arch[0]
        optimized_reduce = surrogate_arch[1]
        arch_normal = target_arch[0]
        arch_reduce = target_arch[1]
        input_adv = Linf_PGD(model_twin, optimized_normal, optimized_reduce, input, target, eps=eps, alpha=eps / 10, steps=steps, rand_start=True)
        input_adv_ = Linf_PGD(model_twin, arch_normal, arch_reduce, input, target, eps=eps, alpha=eps / 10, steps=steps, rand_start=True)

        logits = self._inner_forward(input, arch_normal, arch_reduce)
        acc_clean = utils.accuracy(logits, target, topk=(1, 5))[0] / 100.0

        logits = self._inner_forward(input, optimized_normal, optimized_reduce)
        optimized_acc_clean = utils.accuracy(logits, target, topk=(1, 5))[0] / 100.0

        logits = self._inner_forward(input_adv, arch_normal, arch_reduce)
        optimized_acc_adv = utils.accuracy(logits, target, topk=(1, 5))[0] / 100.0

        logits = self._inner_forward(input_adv_, arch_normal, arch_reduce)
        acc_adv_baseline = utils.accuracy(logits, target, topk=(1, 5))[0] / 100.0

        reward_old = optimized_acc_adv - acc_adv_baseline

        return reward_old, acc_clean, optimized_acc_clean, acc_adv_baseline, optimized_acc_adv

    def _loss_transformer(self, model_twin, input, target, baseline=None, eps=0.1, steps=10, stop=False):
        if self.count == 0:
            self.arch_normal = self.arch_normal_master.forward()
            self.arch_reduce = self.arch_reduce_master.forward()
            self.optimized_normal, self.optimized_normal_logP, self.optimized_normal_entropy, self.probs_normal = self.arch_transformer.forward(self.arch_normal)
            self.optimized_reduce, self.optimized_reduce_logP, self.optimized_reduce_entropy, self.probs_reduce = self.arch_transformer.forward(self.arch_reduce)
        self.count = self.count + 1
        arch_normal = self.arch_normal
        arch_reduce = self.arch_reduce
        optimized_normal = self.optimized_normal
        optimized_reduce = self.optimized_reduce
        reward_old, acc_clean, _, acc_adv, optimized_acc_adv = self.evaluate_transfer(
            model_twin, (self.arch_normal, self.arch_reduce), (self.optimized_normal, self.optimized_reduce), input, target, eps=0.1, steps=10
        )

        reward_old = reward_old if reward_old > 0 else reward_old
        reward = reward_old - baseline if baseline else reward_old

        n = input.size(0)
        self.acc_clean.update(acc_clean, n)

        self.reward.update(reward, n)
        self.optimized_acc_adv.update(optimized_acc_adv, n)
        self.acc_adv.update(acc_adv, n)
        if stop:
            if self.reward_type == "absolute":
                reward = self.reward.avg
            elif self.reward_type == "relative":
                reward = (self.optimized_acc_adv.avg - self.acc_adv.avg) / (self.acc_clean.avg - self.acc_adv.avg)
            policy_loss = -(self.optimized_normal_logP + self.optimized_reduce_logP) * reward - (
                self.entropy_coeff[0] * self.optimized_normal_entropy + self.entropy_coeff[1] * self.optimized_reduce_entropy
            )
            optimized_acc_adv = self.optimized_acc_adv.avg
            acc_adv = self.acc_adv.avg
            utils.update_arch(
                self.best_pair, self.arch_normal, self.arch_reduce, self.optimized_normal, self.optimized_reduce, reward, self.acc_clean.avg, self.acc_adv.avg, self.optimized_acc_adv.avg
            )
            self.initialize_step()
            return (
                policy_loss,
                reward,
                optimized_acc_adv,
                acc_adv,
                self.optimized_normal_entropy,
                self.optimized_reduce_entropy,
            )
        else:
            policy_loss = None
            return policy_loss, 0, 0, 0, 0, 0

    def arch_parameters(self):
        return self._arch_parameters

    def transformer_parameters(self):
        return self._transformer_parameters

    def model_parameters(self):
        for k, v in self.named_parameters():
            if "arch" not in k:
                yield v

    def eval_transfer(self, input_adv, input_clean, valid_target, arch_normal, arch_reduce):
        # self.eval()

        logits = self._inner_forward(input_clean, arch_normal, arch_reduce)
        acc_clean = utils.accuracy(logits, valid_target, topk=(1, 5))[0] / 100.0

        logits = self._inner_forward(input_adv, arch_normal, arch_reduce)
        acc_adv = utils.accuracy(logits, valid_target, topk=(1, 5))[0] / 100.0

        return (acc_clean, acc_adv)

    def _test_transfer(self, model_twin, test_queue, target_normal, target_reduce, surrogate_normal, surrogate_reduce):
        eps = 0.031
        steps = 10
        acc_clean_ = utils.AvgrageMeter()
        acc_adv_ = utils.AvgrageMeter()
        for step, (input, target) in enumerate(test_queue):
            if step >= self.accu_batch:
                break
            n = input.size(0)
            input = input.to(self._device)
            target = target.to(self._device)
            input_adv = Linf_PGD(model_twin, surrogate_normal, surrogate_reduce, input, target, eps=eps, alpha=eps / 10, steps=steps, rand_start=True)
            # input_adv_ = Linf_PGD(model_twin, target_normal, target_reduce, input, target, eps= eps, alpha= eps / 10, steps = steps, rand_start=True)
            logits = self._inner_forward(input, target_normal, target_reduce)
            acc_clean = utils.accuracy(logits, target, topk=(1, 5))[0] / 100.0
            acc_clean_.update(acc_clean, n)

            logits = self._inner_forward(input_adv, target_normal, target_reduce)
            optimized_acc_adv = utils.accuracy(logits, target, topk=(1, 5))[0] / 100.0
            acc_adv_.update(optimized_acc_adv, n)

        return acc_clean_.avg, acc_adv_.avg
