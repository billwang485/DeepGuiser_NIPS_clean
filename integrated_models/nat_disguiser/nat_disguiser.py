import os
from copy import deepcopy
import utils
import genotypes
from operations import *
import numpy as np
from PyPDF2 import PdfFileMerger
from basic_parts.basic_integrated_model import NASNetwork as Network
from basic_parts.basic_arch_transformers import ArchTransformerGates


class NATDisguiser(Network):
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
        pdg_step=10,
    ):
        super(NATDisguiser, self).__init__(
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
        self.pgd_step = pdg_step
        self.derive = False
        self.imitation = False

    def initialize_arch_transformer(self):
        self.arch_transformer = ArchTransformerGates(self._steps, self._device, self.transformer_normalize, op_type=self.op_type)
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

        genotype = utils.arch_to_genotype(arch_normal, arch_reduce, self._steps, self.op_type, normal_concat, reduce_concat)
        transformed_genotype = utils.arch_to_genotype(best_optimized_arch_normal, best_optimized_arch_reduce, self._steps, self.op_type, normal_concat, reduce_concat)
        utils.draw_genotype(genotype.normal, self._steps, os.path.join(folder, "normal_%s" % suffix), genotype.normal_concat)
        utils.draw_genotype(genotype.reduce, self._steps, os.path.join(folder, "reduce_%s" % suffix), genotype.reduce_concat)
        utils.draw_genotype(transformed_genotype.normal, self._steps, os.path.join(folder, "disguised_normal_%s" % suffix), transformed_genotype.normal_concat)
        utils.draw_genotype(transformed_genotype.reduce, self._steps, os.path.join(folder, "disguised__reduce_%s" % suffix), transformed_genotype.reduce_concat)
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
        if self.count == 0:
            self.arch_normal = self.arch_normal_master_demo.forward()
            self.arch_reduce = self.arch_normal_master_demo.forward()
            (self.optimized_normal, self.optimized_reduce, self.optimized_logP, self.optimized_entropy, self.probs_normal, self.probs_reduce) = self.arch_transformer.forward(
                self.arch_normal, self.arch_reduce
            )
        self.count = self.count + 1
        arch_normal = self.arch_normal
        arch_reduce = self.arch_reduce
        optimized_normal = self.optimized_normal
        optimized_reduce = self.optimized_reduce
        input_adv = utils.linf_pgd(model_twin, optimized_normal, optimized_reduce, input, target, eps=eps, alpha=eps / steps, steps=steps, rand_start=False)
        input_adv_ = utils.linf_pgd(model_twin, arch_normal, arch_reduce, input, target, eps=eps, alpha=eps / steps, steps=steps, rand_start=False)

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

            utils.update_arch(
                self.best_pair, self.arch_normal, self.arch_reduce, self.optimized_normal, self.optimized_reduce, reward, self.acc_clean.avg, self.acc_adv.avg, self.optimized_acc_adv.avg
            )
            self.initialize_step()
            return (policy_loss, reward, optimized_acc_adv, acc_adv, self.optimized_entropy)
        else:
            policy_loss = None
            return policy_loss, 0, 0, 0, 0
