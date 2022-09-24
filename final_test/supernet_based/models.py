from operations import *
import utils
from basic_parts.basic_nat_models import NASCell

'''
This is a fixed-arch supernet, which performs same function as a single architecture model
'''
class LooseEndModel(nn.Module):
    model_type = "supernet_based"
    def __init__(
        self,
        device,
        num_classes,
        genotype, 
        layers = 8,
        steps = 4,
        stem_multiplier = 3,
        C = 20,
        loose_end = True
    ):
        super(LooseEndModel, self).__init__()
        self._C = C # Chenyu: I don't know what c means but I just keep it here
        self._num_classes = num_classes
        self._layers = layers
        self._steps = steps
        multiplier = steps
        self._device = device
        self._stem_multiplier = stem_multiplier
        self._genotype = genotype
        self._loose_end = loose_end
        if loose_end == True:
            self.op_type = "LOOSE_END_PRIMITIVES" 
        else:
            self.op_type = "FULLY_CONCAT_PRIMITIVES"

        self._arch_normal, self._arch_reduce = utils.genotype_to_arch(self._genotype)

        C_curr = stem_multiplier * self._C
        self.stem = nn.Sequential(nn.Conv2d(3, C_curr, 3, padding=1, bias=False), nn.BatchNorm2d(C_curr))

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, self._C
        self.cells = nn.ModuleList()
        reduction_prev = False
        _concat = None
        # reduce_concat = None
        reduce_concat = [5]
        normal_concat = [5]
        # normal_concat = None
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
            cell = NASCell(steps, device, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, loose_end=True, concat=_concat, op_type=self.op_type)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        self.best_pair = []
        self.accu_batch = 1
        self.reward_type = "absolute"
        self.single = False
        mean = torch.tensor([0.4914, 0.4822, 0.4465], dtype=torch.float32).cuda()
        std = torch.tensor([0.2023, 0.1994, 0.2010], dtype=torch.float32).cuda()
        self.normalizer = utils.NormalizeByChannelMeanStd(mean=mean, std=std)
        self.count = 0
        self.reward = utils.AvgrageMeter()
        self.optimized_acc_adv = utils.AvgrageMeter()
        self.acc_adv = utils.AvgrageMeter()
        self.acc_clean = utils.AvgrageMeter()
        self.pgd_step = 10
        self.tiny_imagenet = False
        self.image_size = 32
        self.num_channels = 3
    
    def set_genotype(self, genotype):
        self._genotype = genotype
        self._arch_normal, self._arch_reduce = utils.genotype_to_arch(self._genotype)
    
    def model_parameters(self):
        for k, v in self.named_parameters():
            if 'arch' not in k:
                yield v

    def forward(self, input):
        input = self.normalizer(input)

        s0 = self.stem(input)
        s1 = s0
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                archs = self._arch_reduce
            else:
                archs = self._arch_normal
            s0, s1 = s1, cell(s0, s1, archs)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits