import os
from copy import deepcopy
import numpy as np
from PyPDF2 import PdfFileMerger

import utils
from operations import *
import genotypes
from genotypes import Genotype
from basic_parts.basic_nat_models import NASCell, ArchMaster
from basic_parts.basic_arch_transformers import ArchTransformerGates
from predictors.predictors import VanillaGatesPredictor
class PredictorBasedDisguiser(nn.Module):

    NAME = "predictor_based_disguiser"

    def __init__(self, device, predictor_config, search_space_config, transformer_config, strategy_config, save_path):
        super(PredictorBasedDisguiser, self).__init__()
        # self.baseline_avg = 10
        # self.vpi = False
        # self.flops_limit = False
        # self.op_diversity = False
        # self.num_limit = False
        # self.ss = 'nat'
        # self.flag = True
        # self.use_arch = False
        # self.derive = False
        self._device = device
        
        self._predictor_config = predictor_config
        self._initialize_predictor(predictor_config)

        self._search_space_config = search_space_config
        self.op_type = search_space_config["op_type"]
        self._steps = search_space_config["steps"]
        self._initialize_arch_master(self._search_space_config)

        self._transformer_config = transformer_config
        self._initialize_arch_transformer(self._transformer_config)

        self._strategy_config = strategy_config

        self._save_path = save_path

        self._iteration = 0

        (
            self.transform2gates,
            self.transform2nat,
            self.gates_op_list,
        ) = utils.primitives_translation(self.op_type)

        # debug 参数
        self._one_arch = False
    
    def _initialize_arch_master(self, search_space_config):
        op_type = search_space_config["op_type"]
        steps = search_space_config["steps"]
        try:
            COMPACT_PRIMITIVES = eval("genotypes.%s" % op_type)
        except:
            assert False, "not supported op type %s" % (op_type)

        num_ops = len(COMPACT_PRIMITIVES) - 1
        self.arch_normal_master = ArchMaster(num_ops, steps, self._device)
        self.arch_reduce_master = ArchMaster(num_ops, steps, self._device)
        self._arch_parameters = list(self.arch_normal_master.parameters()) + list(self.arch_reduce_master.parameters())
    
    def _initialize_arch_transformer(self, transformer_config):
        self.arch_transformer = ArchTransformerGates(self._steps, self._device, transformer_config["transformer_normalize"], op_type=self.op_type)
        self._transformer_parameters = list(self.arch_transformer.parameters())

    def _initialize_predictor(self, predictor_config):
        self.predictor = VanillaGatesPredictor(self._device, predictor_config["op_type"], loss_type = predictor_config['loss_type'], dropout=predictor_config['dropout'], mode = predictor_config['mode'], concat= predictor_config['gates_concat'])
        self._predictor_parameters = list(self.predictor.parameters())
    
    def set_arch(self, arch_normal, arch_reduce):
        self.arch_normal = arch_normal
        self.arch_reduce = arch_reduce

    def set_thre(self, thre):
        self.thre = thre
    
    def derive_optimized_arch(self, arch_normal, arch_reduce, n_optim, logger, folder, suffix, normal_concat=None, reduce_concat=None):
        self.predictor.eval()
        self.arch_transformer.eval()
        best_reward = -np.inf
        # best_optimized_acc_clean = -np.inf
        best_arch_logP = None
        best_arch_ent = None
        best_optimized_arch_normal = None
        best_optimized_arch_reduce = None
        size_ = [1,3,32,32]
        # score = self.predictor.forward([arch_normal_, arch_reduce_], [optimized_normal_, optimized_reduce_])

        # acc_clean_baseline, acc_adv_baseline = self._test_transfer(model_twin, test_queue, arch_normal, arch_reduce, arch_normal, arch_reduce)
        # logger.info("Sampling candidate architectures ...")
        # tmp = torch.load('z/random_transform_resblock/May 10 07 42 35/train_data_constrain_10', map_location='cpu')
        for i in range(n_optim):
            optimized_normal, optimized_reduce, optimized_logP, optimized_entropy, probs_normal, probs_reduce = self.arch_transformer.forward(arch_normal, arch_reduce, self.derive)
            arch_normal_ = []
            arch_reduce_ = []
            optimized_normal_ = []
            optimized_reduce_ = []
            # from genotypes import HANAG_ResBlock
            # optimized_normal, optimized_reduce = utils.genotype_to_arch(HANAG_ResBlock, self.op_type)
            # optimized_normal, optimized_reduce = tmp[i]['surrogate_arch']
            for j in range(len(arch_normal)):
                arch_normal_.append(([arch_normal[j][0]], [arch_normal[j][1]], [arch_normal[j][2]]))
                arch_reduce_.append(([arch_reduce[j][0]], [arch_reduce[j][1]], [arch_reduce[j][2]]))
                optimized_normal_.append(([optimized_normal[j][0]], [optimized_normal[j][1]], [optimized_normal[j][2]]))
                optimized_reduce_.append(([optimized_reduce[j][0]], [optimized_reduce[j][1]], [optimized_reduce[j][2]]))
            score0 = self.predictor.forward([arch_normal_, arch_reduce_], [optimized_normal_, optimized_reduce_]).item()
            target_flops = utils.compute_flops(self, size_,arch_normal ,arch_reduce, 'null')
            surrogate_flops = utils.compute_flops(self, size_,optimized_normal, optimized_reduce, 'null')
            # score1 = self.predictor.forward([arch_normal_, arch_reduce_], [arch_normal_, arch_reduce_]).item()
            # print(score0-score1)
            # print(tmp[i]['absolute_reward'].item())
            # input()
            z = utils.transform_times(arch_normal, optimized_normal) + utils.transform_times(arch_reduce, optimized_reduce)
            # print(z)
            # z = 1
            if self.num_limit:
                z = 1 / (2**(z - self.thre) + 1)
            else:
                z = 1
            # print(z)
            # reward = reward * z
            if (score0)*z > best_reward:
                best_reward = (score0)
                # best_optimized_acc_clean = optimized_acc_clean
                best_optimized_arch_normal = optimized_normal
                best_optimized_arch_reduce = optimized_reduce
                best_arch_logP = optimized_logP
                best_arch_ent = optimized_entropy
        # logger.info("Target: acc_clean = %.2f acc_adv = %.2f", acc_clean_baseline, acc_adv_baseline )
        # logger.info("Best surrogate: acc_clean = %.2f acc_adv = %.2f", best_optimized_acc_clean, best_optimized_acc_adv )
        # logger.info("Absolute reward = %.2f", best_reward)

        genotype = utils.arch_to_genotype(arch_normal, arch_reduce, self._steps, self.op_type, normal_concat, reduce_concat)
        transformed_genotype = utils.arch_to_genotype(best_optimized_arch_normal, best_optimized_arch_reduce, self._steps, self.op_type, normal_concat, reduce_concat)
        utils.draw_genotype(genotype.normal, self._steps, os.path.join(folder, "normal_%s" % suffix), genotype.normal_concat)
        utils.draw_genotype(genotype.reduce, self._steps, os.path.join(folder, "reduce_%s" % suffix), genotype.reduce_concat)
        utils.draw_genotype(transformed_genotype.normal, self._steps, os.path.join(folder, "disguised_normal_%s" % suffix), transformed_genotype.normal_concat)
        utils.draw_genotype(transformed_genotype.reduce, self._steps, os.path.join(folder, "disguised_reduce_%s" % suffix), transformed_genotype.reduce_concat)
        file_merger = PdfFileMerger()

        file_merger.append(os.path.join(folder, "normal_%s.pdf" % suffix))
        file_merger.append(os.path.join(folder, "reduce_%s.pdf" % suffix))

        file_merger.write(os.path.join(folder, "target.pdf"))

        file_merger = PdfFileMerger()

        file_merger.append(os.path.join(folder, "disguised_normal_%s.pdf" % suffix))
        file_merger.append(os.path.join(folder, "disguised_reduce_%s.pdf" % suffix))

        file_merger.write(os.path.join(folder, "disguised_target.pdf"))

        # logger.info('genotype = %s', genotype)
        # logger.info('optimized_genotype = %s', transformed_genotype)
        # target_flops = utils.compute_flops(self, [1, 3, 32, 32],arch_normal ,arch_reduce, 'null')
        # surrogate_flops = utils.compute_flops(self, [1, 3, 32, 32], optimized_normal, optimized_reduce, 'null')
        # logger.info('target_flops')
        # logger.info(target_flops)
        # logger.info('surrogate_flops')
        # logger.info(surrogate_flops)
        result = {}
        result["target_arch"] = (deepcopy(arch_normal), deepcopy(arch_reduce))
        result["surrogate_arch"] = (deepcopy(best_optimized_arch_normal), deepcopy(best_optimized_arch_reduce))
        result["absolute_predictor_reward"] = best_reward
        result["best_arch_logP"] = best_arch_logP
        result["best_arch_ent"] = best_arch_ent
        # result["target_flops"] = target_flops
        # result["surrogate_flops"] = surrogate_flops
        result['prob_normal'] = probs_normal
        result['target_flops'] = target_flops
        result['surrogate_flops'] = surrogate_flops
        result['budget'] = surrogate_flops / target_flops
        logger.info('Budget: %f, acc_adv: %f ', surrogate_flops / target_flops, best_reward)
        # result = 0
        return result
    
    def update_counter(self):
        self._iteration += 1

    def get_baseline(self):
        baseline = []
        for i in range(self.baseline_avg):
            baseline_normal = self.uni_random_transform(self.arch_normal)
            baseline_reduce = self.uni_random_transform(self.arch_reduce)
            baseline_normal_ = []
            baseline_reduce_ = []
            for i in range(len(self.arch_normal)):
                baseline_normal_.append(([baseline_normal[i][0]], [baseline_normal[i][1]], [baseline_normal[i][2]]))
                baseline_reduce_.append(([baseline_reduce[i][0]], [baseline_reduce[i][1]], [baseline_reduce[i][2]]))
            score = self.predictor.forward([self.arch_normal_, self.arch_reduce_], [baseline_normal_, baseline_reduce_]).item()
            # score = self.predictor.forward([self.arch_normal_, self.arch_reduce_], [baseline_normal_, baseline_reduce_]).item()
            baseline.append(score)
        baseline.sort(reverse=True)
        # baseline = baseline[0:5]
        baseline = sum(baseline) / len(baseline)
        baseline = baseline - self.predictor.forward([self.arch_normal_, self.arch_reduce_], [self.arch_normal_, self.arch_reduce_]).item()
        return baseline if baseline > 0 else 0

    def save_trace(self, env_action, policy_loss, reward, autillary_info):
        if not os.path.exists(os.path.join(self._save_path, "data_trace.yaml")):
            save_list = []
        else:
            save_list = utils.load_yaml(os.path.join(self._save_path, "data_trace.yaml"))
        save_list.append({"env_action": env_action, 
                        "policy_loss":policy_loss.item(), 
                        "reward": reward,
                        "autillary_info": autillary_info})
        utils.save_yaml(save_list, os.path.join(self._save_path, "data_trace.yaml"))

    def _loss_transformer(self, baseline = None):
        policy_loss = torch.empty([1], device = self._device, requires_grad= True)
        
        self.arch_normal = self.arch_normal_master.forward()
        self.arch_reduce = self.arch_reduce_master.forward()
        data_trace = []
        env_action = []
        self.update_counter()
        
        self.optimized_normal, self.optimized_reduce, self.optimized_logP, self.optimized_entropy, self.probs_normal, self.probs_reduce = self.arch_transformer.forward(self.arch_normal, self.arch_reduce)
        env_action.append({"target_genotype":"{}".format(utils.arch_to_genotype(self.arch_normal, self.arch_reduce, self._steps, self.op_type, [5], [5])),
            "surrogate_genotype":"{}".format(utils.arch_to_genotype(self.optimized_normal, self.optimized_reduce, self._steps, self.op_type, [5], [5])),
            "iteration":self._iteration})

        [gates_normal, gates_reduce] = utils.nat_arch_to_gates(self.arch_normal, self.arch_reduce, self.transform2gates)
        [gates_optimized_normal, gates_optimized_reduce] = utils.nat_arch_to_gates(self.optimized_normal, self.optimized_reduce, self.transform2gates)
        
        acc_adv_surrogate = self.predictor.forward([gates_normal, gates_reduce], [gates_optimized_normal, gates_optimized_reduce], gates_input=True).item()
        # acc_adv_target = self.predictor.forward([arch_normal, arch_reduce], [arch_normal, arch_reduce]).item()
        reward_old = acc_adv_surrogate
        reward = reward_old
            # reward_baseline = self.predictor.forward([arch_normal, arch_reduce], [label_normal, label_reduce]).item()
            # torch.save(reward_baseline, os.path.join('no_name', 'reward_baseline'))

            # if baseline is not None:
            #     if self.vpi:
            #         baseline = self.get_baseline()

            #     reward = reward_old - baseline 
            # else:
            #     reward = reward_old
            
            # if self.flops_limit:
            #     target_flops = utils.compute_flops(self, [1, 3, 32, 32], self.arch_normal, self.arch_reduce)
            #     surrogate_flops = utils.compute_flops(self, [1, 3, 32, 32], self.optimized_normal, self.optimized_reduce)
            #     # print(target_flops)
            #     # print(surrogate_flops)
            #     x = (target_flops / surrogate_flops)
            #     x = 1 / x 
            #     x = 1 / (100**(x - 1.5) + 1)
            #     reward = reward * x
            # else:
            #     target_flops = 1
            #     surrogate_flops = 1
            #     x = 1

        auxillary_info = {}
        
        if self._strategy_config["compute_reward"]["hardware_constrain"] == "num_changes":
            z = utils.transform_times(self.arch_normal, self.optimized_normal) + utils.transform_times(self.arch_reduce, self.optimized_reduce)
            auxillary_info["num_changes"] = z
            z = 1 / (2**(z - self._strategy_config["compute_reward"]["threshold"]) + 1)
            reward = reward * z
        else:
            z = 1

        
        # if self.op_diversity:
        #     target_diversity = utils.op_diversity(self.arch_normal) + utils.op_diversity(self.arch_reduce)
        #     surrogate_diversity = utils.op_diversity(self.optimized_normal) + utils.op_diversity(self.optimized_reduce)
        #     y = surrogate_diversity / target_diversity
        #     reward = reward * y
        # else:
        #     surrogate_diversity = 1
        #     target_diversity = 1
        #     y = 1
        policy_loss = - (self.optimized_logP) * reward - (self._strategy_config["compute_loss"]["entropy_coeff"] * self.optimized_entropy)

                

        

        # policy_loss = policy_loss / 1

            # label_normal, label_reduce = utils.genotype_to_arch(genotypes.HANAG_ResBlock, self.op_type)
            # policy_loss = utils.imitation_loss(label_normal, label_reduce, self.probs_normal, self.probs_reduce, self._device)
        
        # self.save_trace(env_action, policy_loss, reward, auxillary_info)

        return policy_loss, reward_old, self.optimized_entropy , auxillary_info
    