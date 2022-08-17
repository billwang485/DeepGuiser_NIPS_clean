import torch
import utils

class Transformer(object):
    def __init__(self, model, model_twin, args):
        self.args = args
        self.model = model
        self.model_twin = model_twin
        self.optimizer = torch.optim.Adam(self.model.transformer_parameters(),
                                          lr=args.lr, betas=(0.5, 0.999),
                                          weight_decay=args.transformer_weight_decay)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #      self.optimizer, float(args.iterations), eta_min=args.lr_min
        #  )
        self.baseline = 0
        self.gamma = args.gamma
        self.accu_batch = args.accu_batch
        self.initialize_step()
    
    def initialize_step(self):
        self.count = 0

    def update_baseline(self, reward):
        self.baseline = self.baseline * self.gamma + reward * (1-self.gamma)

    def accumulate(self):
        self.accumulation = True

    def step(self, input_valid, target_valid):
        n = input_valid.size(0)
        # if self.count == 0:
        #     self.optimizer.zero_grad()
        self.count = self.count + 1
        self.optimizer.zero_grad()
        loss, reward, optimized_acc_adv, acc_adv, normal_ent, reduce_ent = self.model._loss_transformer(self.model_twin, input_valid, target_valid, self.baseline, stop = (self.count % self.accu_batch == 0))
        
        if self.count % self.accu_batch == 0:
            
            loss.backward()
            self.optimizer.step()
            # self.optimizer.zero_grad()
            self.update_baseline(reward)
            self.initialize_step()
            return reward, optimized_acc_adv, acc_adv, normal_ent, reduce_ent, loss
        else:
            return 0, 0, 0, 0, 0, 0

        