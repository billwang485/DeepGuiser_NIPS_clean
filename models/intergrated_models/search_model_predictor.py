from operations import *
import utils
import os
import numpy as np
from utils import arch_to_genotype, draw_genotype
from copy import deepcopy
from integrated_models.nat_disguier.search_model_gates import NASNetwork as Network
from PyPDF2 import PdfFileMerger


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
        self.baseline_avg = 10
        self.vpi = False
        self.flops_limit = False
        self.op_diversity = False
        self.num_limit = False
        self.ss = "nat"
        self.flag = True
        self.use_arch = False
        self.derive = False

    def set_arch(self, arch_normal, arch_reduce):
        self.arch_normal = arch_normal
        self.arch_reduce = arch_reduce

    def set_thre(self, args):
        self.thre = args.thre

    def derive_optimized_arch(self, arch_normal, arch_reduce, n_optim, logger, folder, suffix, normal_concat=None, reduce_concat=None):
        best_reward = -np.inf
        best_arch_logP = None
        best_arch_ent = None
        best_optimized_arch_normal = None
        best_optimized_arch_reduce = None
        size_ = [1, 3, 32, 32]
        for i in range(n_optim):
            optimized_normal, optimized_reduce, optimized_logP, optimized_entropy, probs_normal, probs_reduce = self.arch_transformer.forward(arch_normal, arch_reduce, self.derive)
            arch_normal_ = []
            arch_reduce_ = []
            optimized_normal_ = []
            optimized_reduce_ = []
            for j in range(len(arch_normal)):
                arch_normal_.append(([arch_normal[j][0]], [arch_normal[j][1]], [arch_normal[j][2]]))
                arch_reduce_.append(([arch_reduce[j][0]], [arch_reduce[j][1]], [arch_reduce[j][2]]))
                optimized_normal_.append([optimized_normal[j][0]], [optimized_normal[j][1]], [optimized_normal[j][2]])
                optimized_reduce_.append([optimized_reduce[j][0]], [optimized_reduce[j][1]], [optimized_reduce[j][2]])
            score0 = self.predictor.forward([arch_normal_, arch_reduce_], [optimized_normal_, optimized_reduce_]).item()
            target_flops = utils.compute_flops(self, size_, arch_normal, arch_reduce, "null")
            surrogate_flops = utils.compute_flops(self, size_, optimized_normal, optimized_reduce, "null")
            z = utils.transform_times(arch_normal, optimized_normal) + utils.transform_times(arch_reduce, optimized_reduce)
            if self.num_limit:
                z = 1 / (2 ** (z - self.thre) + 1)
            else:
                z = 1
            if (score0) * z > best_reward:
                best_reward = score0
                best_optimized_arch_normal = optimized_normal
                best_optimized_arch_reduce = optimized_reduce
                best_arch_logP = optimized_logP
                best_arch_ent = optimized_entropy

        genotype = arch_to_genotype(arch_normal, arch_reduce, self._steps, self.op_type, normal_concat, reduce_concat)
        transformed_genotype = arch_to_genotype(best_optimized_arch_normal, best_optimized_arch_reduce, self._steps, self.op_type, normal_concat, reduce_concat)
        draw_genotype(genotype.normal, self._steps, os.path.join(folder, "normal_%s" % suffix), genotype.normal_concat)
        draw_genotype(genotype.reduce, self._steps, os.path.join(folder, "reduce_%s" % suffix), genotype.reduce_concat)
        draw_genotype(transformed_genotype.normal, self._steps, os.path.join(folder, "disguised_normal_%s" % suffix), transformed_genotype.normal_concat)
        draw_genotype(transformed_genotype.reduce, self._steps, os.path.join(folder, "disguised_reduce_%s" % suffix), transformed_genotype.reduce_concat)
        file_merger = PdfFileMerger()

        file_merger.append(os.path.join(folder, "normal_%s.pdf" % suffix))
        file_merger.append(os.path.join(folder, "reduce_%s.pdf" % suffix))

        file_merger.write(os.path.join(folder, "target.pdf"))

        file_merger = PdfFileMerger()

        file_merger.append(os.path.join(folder, "disguised_normal_%s.pdf" % suffix))
        file_merger.append(os.path.join(folder, "disguised_reduce_%s.pdf" % suffix))

        file_merger.write(os.path.join(folder, "disguised_target.pdf"))

        result = {}
        result["target_arch"] = (deepcopy(arch_normal), deepcopy(arch_reduce))
        result["surrogate_arch"] = (deepcopy(best_optimized_arch_normal), deepcopy(best_optimized_arch_reduce))
        result["absolute_predictor_reward"] = best_reward
        result["best_arch_logP"] = best_arch_logP
        result["best_arch_ent"] = best_arch_ent
        result["prob_normal"] = probs_normal
        result["target_flops"] = target_flops
        result["surrogate_flops"] = surrogate_flops
        result["budget"] = surrogate_flops / target_flops
        logger.info("Budget: %f, acc_adv: %f ", surrogate_flops / target_flops, best_reward)
        # result = 0
        return result

    def get_baseline(self):
        baseline = []
        for i in range(self.baseline_avg):
            baseline_normal = self.uni_random_transform(self.arch_normal)
            baseline_reduce = self.uni_random_transform(self.arch_reduce)
            baseline_normal_ = []
            baseline_reduce_ = []
            for i in range(len(self.arch_normal)):
                baseline_normal_.append([baseline_normal[i][0]], [baseline_normal[i][1]], [baseline_normal[i][2]])
                baseline_reduce_.append([baseline_reduce[i][0]], [baseline_reduce[i][1]], [baseline_reduce[i][2]])
            score = self.predictor.forward([self.arch_normal_, self.arch_reduce_], [baseline_normal_, baseline_reduce_]).item()
            baseline.append(score)
        baseline.sort(reverse=True)
        baseline = sum(baseline) / len(baseline)
        baseline = baseline - self.predictor.forward([self.arch_normal_, self.arch_reduce_], [self.arch_normal_, self.arch_reduce_]).item()
        return baseline if baseline > 0 else 0

    def _loss_transformer(self, baseline=None):
        self.arch_reduce_master.demo = True
        self.arch_reduce_master.demo = True
        reward_old_ = utils.AvgrageMeter()
        policy_loss = torch.zeros(1)
        policy_loss = policy_loss.to(self._device)
        policy_loss.requires_grad_()

        if not self.use_arch:

            if self.ss == "null":
                self.arch_normal = self.arch_normal_master_demo.forward()
                self.arch_reduce = self.arch_normal_master_demo.forward()
            else:
                self.arch_normal = self.arch_normal_master.forward()
                self.arch_reduce = self.arch_reduce_master.forward()
        for i in range(1):
            (
                self.optimized_normal,
                self.optimized_reduce,
                self.optimized_logP,
                self.optimized_entropy,
                self.probs_normal,
                self.probs_reduce,
            ) = self.arch_transformer.forward(self.arch_normal, self.arch_reduce)
            arch_normal = list()
            arch_reduce = list()
            optimized_normal = list()
            optimized_reduce = list()

            for i in range(len(self.arch_normal)):
                arch_normal.append([self.arch_normal[i][0]], [self.arch_normal[i][1]], [self.arch_normal[i][2]])
                arch_reduce.append([self.arch_reduce[i][0]], [self.arch_reduce[i][1]], [self.arch_reduce[i][2]])
                optimized_normal.append([self.optimized_normal[i][0]], [self.optimized_normal[i][1]], [self.optimized_normal[i][2]])
                optimized_reduce.append([self.optimized_reduce[i][0]], [self.optimized_reduce[i][1]], [self.optimized_reduce[i][2]])

            self.arch_normal_ = arch_normal
            self.arch_reduce_ = arch_reduce

            acc_adv_surrogate = self.predictor.forward([arch_normal, arch_reduce], [optimized_normal, optimized_reduce]).item()
            acc_adv_target = self.predictor.forward([arch_normal, arch_reduce], [arch_normal, arch_reduce]).item()
            reward_old = acc_adv_surrogate

            if baseline is not None:
                if self.vpi:
                    baseline = self.get_baseline()

                reward = reward_old - baseline
            else:
                reward = reward_old

            if self.flops_limit:
                target_flops = utils.compute_flops(self, [1, 3, 32, 32], self.arch_normal, self.arch_reduce)
                surrogate_flops = utils.compute_flops(self, [1, 3, 32, 32], self.optimized_normal, self.optimized_reduce)
                x = target_flops / surrogate_flops
                x = 1 / x
                x = 1 / (100 ** (x - 1.5) + 1)
                reward = reward * x
            else:
                target_flops = 1
                surrogate_flops = 1
                x = 1

            if self.num_limit:
                z = utils.transform_times(self.arch_normal, self.optimized_normal) + utils.transform_times(self.arch_reduce, self.optimized_reduce)
                z = 1 / (2 ** (z - self.thre) + 1)
                reward = reward * z
            else:
                z = 1

            if self.op_diversity:
                target_diversity = utils.op_diversity(self.arch_normal) + utils.op_diversity(self.arch_reduce)
                surrogate_diversity = utils.op_diversity(self.optimized_normal) + utils.op_diversity(self.optimized_reduce)
                y = surrogate_diversity / target_diversity
                reward = reward * y
            else:
                surrogate_diversity = 1
                target_diversity = 1
                y = 1
            policy_loss = policy_loss - (self.optimized_logP) * reward - (self.entropy_coeff[0] * self.optimized_entropy)

        reward_old_.update(reward_old, 1)

        policy_loss = policy_loss / 1

        utils.update_arch(self.best_pair, self.arch_normal, self.arch_reduce, self.optimized_normal, self.optimized_reduce, reward_old, 0, 0, 0)

        return policy_loss, reward_old_.avg, self.optimized_entropy, 1 / (target_flops / surrogate_flops), y, z
