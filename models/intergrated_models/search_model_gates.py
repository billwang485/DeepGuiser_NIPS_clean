import genotypes
from operations import *
import utils
import numpy as np
from utils import (
    arch_to_genotype,
    check_connectivity_transform,
    draw_genotype,
    Linf_PGD,
)
import os
from copy import deepcopy
from search_model_twin import NASNetwork as Network
from basic_models.basic_gates_models import GCNFlowArchEmbedder
from genotypes import LooseEnd_Transition_Dict, FullyConcat_Transition_Dict
from PyPDF2 import PdfFileMerger


class ArchTransformer_gates(nn.Module):
    def __init__(self, steps, device, normalize=False, op_type="LOOSE_END_PRIMITIVES", transform_type_more=True):
        """

        :param nfeat: feature dimension of each node in the graph
        :param nhid: hidden dimension
        :param dropout: dropout rate for GCN
        """
        super(ArchTransformer_gates, self).__init__()
        self.steps = steps
        self.device = device
        self.normalize = normalize
        self.op_type = op_type
        (
            self.transform2gates,
            self.transform2nat,
            self.gates_op_list,
        ) = utils.primitives_translation(self.op_type)
        if transform_type_more:
            if op_type == "LOOSE_END_PRIMITIVES":
                num_ops = len(genotypes.LOOSE_END_PRIMITIVES)
            else:
                num_ops = len(genotypes.FULLY_CONCAT_PRIMITIVES)
        else:
            num_ops = len(genotypes.TRANSFORM_PRIMITIVES)
        self.arch_embedder = GCNFlowArchEmbedder(self.gates_op_list, node_dim=32, op_dim=32, use_bn=False, hidden_dim=32, gcn_out_dims=[64, 64, 64, 64], gcn_kwargs={"residual_only": 2})
        self.fcs = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_ops * self.steps * 4),
        )

        try:
            COMPACT_PRIMITIVES = eval("genotypes.%s" % op_type)
        except:
            assert False, "not supported op type %s" % (op_type)

        self.hanag_mask = True

    def forward(self, arch_normal, arch_reduce, derive=False):
        assert len(arch_normal) == len(arch_reduce)
        # initial the first two nodes
        gates_arch = utils.nat_arch_to_gates(arch_normal, arch_reduce, self.transform2gates)
        x = self.arch_embedder(gates_arch)
        logits = self.fcs(x)
        logits = logits.squeeze(dim=0)
        logits = logits.view(self.steps * 4, -1)
        entropy = 0
        log_p = 0
        arch_normal_list = []
        arch_reduce_list = []
        try:
            COMPACT_PRIMITIVES = eval("genotypes.{}".format(self.op_type))
        except:
            assert False, "not supported op type %s" % (self.op_type)
        transition_dict = LooseEnd_Transition_Dict if self.op_type == "LOOSE_END_PRIMITIVES" else FullyConcat_Transition_Dict

        probs_normal = torch.zeros(len(COMPACT_PRIMITIVES), len(arch_normal), device=self.device)
        probs_reduce = torch.zeros(len(COMPACT_PRIMITIVES), len(arch_normal), device=self.device)

        normal_connectivity = utils.get_connectivity(arch_normal)

        for idx, (op, f, t) in enumerate(arch_normal):
            select_op = transition_dict[COMPACT_PRIMITIVES[op]]
            selected_arch_index = [COMPACT_PRIMITIVES.index(i) for i in select_op]
            tmp = logits[idx]

            V = tmp.new_zeros(tmp.size(), requires_grad=False)
            V[selected_arch_index] = 1
            if (not check_connectivity_transform(normal_connectivity, f)) and self.hanag_mask:
                V = tmp.new_zeros(tmp.size(), requires_grad=False)
                V[op] = 1
            prob = utils.BinarySoftmax(tmp, V)
            # if idx == 1:
            # print(prob, idx)
            probs_normal[:, idx] = prob
            log_prob = torch.log(torch.clamp(prob, min=1e-5, max=1 - 1e-5))
            entropy += -(log_prob * prob).sum()
            f_op = prob.multinomial(num_samples=1)
            if derive:
                f_op = torch.argmax(prob)
                # print(derive)
            selected_log_p = log_prob.gather(-1, f_op)
            log_p += selected_log_p.sum()
            arch_normal_list.append((f_op, f, t))

        reduce_connectivity = utils.get_connectivity(arch_reduce)

        for idx, (op, f, t) in enumerate(arch_reduce):
            select_op = transition_dict[COMPACT_PRIMITIVES[op]]
            selected_arch_index = [COMPACT_PRIMITIVES.index(i) for i in select_op]
            tmp = logits[idx + len(arch_normal)]
            V = tmp.new_zeros(tmp.size(), requires_grad=False)
            V[selected_arch_index] = 1
            if not check_connectivity_transform(normal_connectivity, f) and self.hanag_mask:
                V = tmp.new_zeros(tmp.size(), requires_grad=False)
                V[op] = 1
            prob = utils.BinarySoftmax(tmp, V)
            # print(prob, idx)
            probs_reduce[:, idx] = prob
            log_prob = torch.log(torch.clamp(prob, min=1e-5, max=1 - 1e-5))
            entropy += -(log_prob * prob).sum()
            f_op = prob.multinomial(num_samples=1)
            if derive:
                f_op = torch.argmax(prob)
            selected_log_p = log_prob.gather(-1, f_op)
            log_p += selected_log_p.sum()
            arch_reduce_list.append((f_op, f, t))
        utils.check_transform(arch_normal, arch_normal_list, op_type=self.op_type)
        utils.check_transform(arch_reduce, arch_reduce_list, op_type=self.op_type)
        return arch_normal_list, arch_reduce_list, log_p, entropy, probs_normal, probs_reduce


class NASNetwork(Network):
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
        super(NASNetwork, self).__init__(
            C,
            num_classes,
            layers,
            criterion,
            device,
            steps,
            stem_multiplier,
            controller_hid,
            entropy_coeff,
            edge_hid,
            transformer_nfeat,
            transformer_nhid,
            transformer_dropout,
            transformer_normalize,
            loose_end,
            normal_concat,
            reduce_concat,
            op_type,
        )
        self.derive = False

    def re_initialize_arch_transformer(self):
        self.arch_transformer = ArchTransformer_gates(self._steps, self._device, self.transformer_normalize, op_type=self.op_type)
        self._transformer_parameters = list(self.arch_transformer.parameters())

    def derive_optimized_arch(self, model_twin, test_queue, arch_normal, arch_reduce, n_optim, logger, folder, suffix, normal_concat=None, reduce_concat=None):
        best_optimized_acc_adv = -np.inf
        best_optimized_acc_clean = -np.inf
        best_arch_logP = None
        best_arch_ent = None
        best_optimized_arch_normal = None
        best_optimized_arch_reduce = None

        acc_clean_baseline, acc_adv_baseline = self._test_transfer(model_twin, test_queue, arch_normal, arch_reduce, arch_normal, arch_reduce)
        logger.info("Sampling candidate architectures ...")
        for i in range(1):

            optimized_normal, optimized_reduce, optimized_logP, optimized_entropy, probs_normal, probs_reduce = self.arch_transformer.forward(arch_normal, arch_reduce, self.derive)
            # optimized_acc_clean, optimized_acc_adv = self._test_transfer(model_twin, test_queue, arch_normal, arch_reduce, optimized_normal, optimized_reduce)
            optimized_acc_clean, optimized_acc_adv = 0, 0
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

        genotype = arch_to_genotype(arch_normal, arch_reduce, self._steps, self.op_type, normal_concat, reduce_concat)
        transformed_genotype = arch_to_genotype(best_optimized_arch_normal, best_optimized_arch_reduce, self._steps, self.op_type, normal_concat, reduce_concat)
        draw_genotype(genotype.normal, self._steps, os.path.join(folder, "normal_%s" % suffix), genotype.normal_concat)
        draw_genotype(genotype.reduce, self._steps, os.path.join(folder, "reduce_%s" % suffix), genotype.reduce_concat)
        draw_genotype(transformed_genotype.normal, self._steps, os.path.join(folder, "disguised_normal_%s" % suffix), transformed_genotype.normal_concat)
        draw_genotype(transformed_genotype.reduce, self._steps, os.path.join(folder, "disguised__reduce_%s" % suffix), transformed_genotype.reduce_concat)
        file_merger = PdfFileMerger()

        file_merger.append(os.path.join(folder, "normal_%s.pdf" % suffix))
        file_merger.append(os.path.join(folder, "reduce_%s.pdf" % suffix))

        file_merger.write(os.path.join(folder, "target.pdf"))

        file_merger = PdfFileMerger()

        file_merger.append(os.path.join(folder, "disguised_normal_%s.pdf" % suffix))
        file_merger.append(os.path.join(folder, "disguised__reduce_%s.pdf" % suffix))

        file_merger.write(os.path.join(folder, "disguised_target.pdf"))

        logger.info("genotype = %s", genotype)
        logger.info("optimized_genotype = %s", transformed_genotype)
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

    def _loss_transformer(self, model_twin, input, target, baseline=None, eps=0.031, steps=10, stop=False):
        # logits, optimized_logits, optimized_normal, optimized_normal_logP, optimized_normal_entropy, optimized_reduce, optimized_reduce_logP, optimized_reduce_entropy = self.transformer_forward(input)
        # accuracy = utils.accuracy(logits, target)[0] / 100.0
        # optimized_accuracy = utils.accuracy(optimized_logits, target)[0] / 100.0
        if self.count == 0:
            self.arch_normal = self.arch_normal_master_demo.forward()
            self.arch_reduce = self.arch_normal_master_demo.forward()
            # self.arch_normal, self.arch_reduce = utils.genotype_to_arch(ResBlock, self.op_type)
            (self.optimized_normal, self.optimized_reduce, self.optimized_logP, self.optimized_entropy, self.probs_normal, self.probs_reduce) = self.arch_transformer.forward(self.arch_normal, self.arch_reduce)
        self.count = self.count + 1
        arch_normal = self.arch_normal
        arch_reduce = self.arch_reduce
        optimized_normal = self.optimized_normal
        optimized_reduce = self.optimized_reduce
        input_adv = Linf_PGD(model_twin, optimized_normal, optimized_reduce, input, target, eps=eps, alpha=eps / steps, steps=steps, rand_start=False)
        input_adv_ = Linf_PGD(model_twin, arch_normal, arch_reduce, input, target, eps=eps, alpha=eps / steps, steps=steps, rand_start=False)

        logits = self._inner_forward(input, arch_normal, arch_reduce)
        acc_clean = utils.accuracy(logits, target, topk=(1, 5))[0] / 100.0

        logits = self._inner_forward(input_adv, arch_normal, arch_reduce)
        optimized_acc_adv = utils.accuracy(logits, target, topk=(1, 5))[0] / 100.0

        logits = self._inner_forward(input_adv_, arch_normal, arch_reduce)
        acc_adv = utils.accuracy(logits, target, topk=(1, 5))[0] / 100.0

        reward_old = optimized_acc_adv - acc_adv
        reward_old = reward_old if reward_old > 0 else reward_old
        reward = reward_old - baseline if baseline else reward_old

        n = input.size(0)
        self.acc_clean.update(acc_clean, n)

        self.reward.update(reward, n)
        self.optimized_acc_adv.update(optimized_acc_adv, n)
        self.acc_adv.update(acc_adv, n)
        if stop:
            if self.reward_type == "absolute":
                reward = (self.optimized_acc_adv.avg - self.acc_adv.avg) * 1
            elif self.reward_type == "relative":
                reward = (self.optimized_acc_adv.avg - self.acc_adv.avg) / (self.acc_clean.avg - self.acc_adv.avg)
            if not self.imitation:
                policy_loss = -(self.optimized_logP) * reward - (self.entropy_coeff[0] * self.optimized_entropy)
            else:
                label_normal, label_reduce = utils.genotype_to_arch(genotypes.HANAG_ResBlock, self.op_type)
                policy_loss = utils.imitation_loss(label_normal, label_reduce, self.probs_normal, self.probs_reduce, self._device)
            optimized_acc_adv = self.optimized_acc_adv.avg
            acc_adv = self.acc_adv.avg

            utils.update_arch(self.best_pair, self.arch_normal, self.arch_reduce, self.optimized_normal, self.optimized_reduce, reward, self.acc_clean.avg, self.acc_adv.avg, self.optimized_acc_adv.avg)
            self.initialize_step()
            return (policy_loss, reward, optimized_acc_adv, acc_adv, self.optimized_entropy)
        else:
            policy_loss = None
            return policy_loss, 0, 0, 0, 0
