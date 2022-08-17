from search_model_twin import NASNetwork
from operations import *
import utils
from distillation import Linf_PGD

'''
This is a fixed-arch supernet, which performs same function as a single architecture model
'''
class FinalNetwork(NASNetwork):
    def __init__(self, C, num_classes, layers, criterion, device, steps=4, controller_hid=None, entropy_coeff=[0.0, 0.0], stem_multiplier=3, edge_hid=100, transformer_nfeat=1024, transformer_nhid=512, transformer_dropout=0, transformer_normalize=False, loose_end=False, normal_concat=None, reduce_concat=None, op_type='FULLY_CONCAT_PRIMITIVES'):
        super(FinalNetwork, self).__init__(C, num_classes, layers, criterion, device, steps=4, controller_hid=controller_hid, entropy_coeff=entropy_coeff,stem_multiplier=stem_multiplier, edge_hid=edge_hid, transformer_nfeat=transformer_nfeat, transformer_nhid=transformer_nhid, transformer_dropout=transformer_dropout, transformer_normalize=transformer_normalize, loose_end=loose_end, normal_concat=normal_concat, reduce_concat=reduce_concat, op_type=op_type)
        self.arch_normal = None
        self.arch_reduce = None
        # if num_classes == 200:
        #     self.initialize_tiny_imagenet()

    def step(self, valid_input, valid_target):
        self._model_optimizer.zero_grad()
        logits = self._inner_forward(valid_input, self.arch_normal, self.arch_reduce)
        loss = self._criterion(logits, valid_target)
        loss.backward()
        self._model_optimizer.step()
        return logits, loss  

    def eval_transfer(self, input_adv, input_clean, valid_target):
        self.eval()

        logits = self._inner_forward(input_clean, self.arch_normal, self.arch_reduce)
        acc_clean = utils.accuracy(logits, valid_target, topk=(1, 5))[0] / 100.0

        logits = self._inner_forward(input_adv, self.arch_normal, self.arch_reduce)
        acc_adv = utils.accuracy(logits, valid_target, topk=(1, 5))[0] / 100.0

        return (acc_clean, acc_adv)
    
    def generate_adv_input(self, input_clean, valid_target, eps):
        self.eval()
        input_adv = Linf_PGD(self, self.arch_normal, self.arch_reduce, input_clean, valid_target, eps = eps, alpha=eps/10, steps=10)

        logits = self._inner_forward(input_clean, self.arch_normal, self.arch_reduce)
        acc_clean = utils.accuracy(logits, valid_target, topk=(1, 5))[0] / 100.0

        logits = self._inner_forward(input_adv, self.arch_normal, self.arch_reduce)
        acc_adv = utils.accuracy(logits, valid_target, topk=(1, 5))[0] / 100.0

        return input_adv, (acc_clean, acc_adv) 

    def test_acc_single(self, test_queue, logger, args):
        top1 = utils.AvgrageMeter()
        top5 = utils.AvgrageMeter()
        self.eval()
        for step, (input, target) in enumerate(test_queue):
            n = input.size(0)
            input = input.to(self._device)
            target = target.to(self._device)
            logits = self._inner_forward(input, self.arch_normal, self.arch_reduce)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            # if step % args.report_freq == 0:
            #     logger.info('Testing Model: Step=%03d Top1=%f Top5=%f ',
            #                 step, top1.avg, top5.avg, )
        logger.info('Testing Model: Top1=%f Top5=%f ',
                            top1.avg, top5.avg, )
        return top1.avg, top5.avg
    
    def evaluate_transfer(self, model_twin, input, target, eps=0.031, steps=10):
        self.eval()
        model_twin.eval()
        (optimized_normal, optimized_reduce) = (model_twin.arch_normal, model_twin.arch_reduce)
        (arch_normal, arch_reduce) = (self.arch_normal, self.arch_reduce)
        input_adv = Linf_PGD(model_twin, optimized_normal, optimized_reduce, input, target, eps= eps, alpha= eps / 10, steps = steps, rand_start=True)
        
        logits = self._inner_forward(input_adv, arch_normal, arch_reduce)
        optimized_acc_adv = utils.accuracy(logits, target, topk=(1, 5))[0] / 100.0

        return optimized_acc_adv



