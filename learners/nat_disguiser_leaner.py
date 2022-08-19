import torch
import utils

class NATDisguiserLearner(object):
    def __init__(self, model, twin_supernet, args):
        self.args = args
        self.model = model
        self.twin_supernet = twin_supernet
        self.optimizer = torch.optim.Adam(self.model.transformer_parameters(),
                                          lr=args.learning_rate, betas=(0.5, 0.999),
                                          weight_decay=args.transformer_weight_decay)
        gamma = (args.learning_rate_min / args.learning_rate)**(1 / args.iterations)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma= gamma)
        self.baseline = 0.0
        self.gamma = args.gamma
        self.accu_batch = args.accumulate_batch
        self.initialize_step()
        self.current_lr = args.learning_rate
    
    def initialize_step(self):
        self.count = 0

    def update_baseline(self, reward):
        self.baseline = self.baseline * self.gamma + reward * (1-self.gamma)

    def accumulate(self):
        self.accumulation = True
    
    # def multi_batch_step(self, input_valid, target_valid):
    #     n = input_valid.size(0)
    #     self.count = self.count + 1
    #     self.optimizer.zero_grad()
    #     loss, reward, optimized_acc_adv, acc_adv, ent = self.model._loss_transformer(self.twin_supernet, input_valid, target_valid, self.baseline, stop = (self.count % self.accu_batch == 0))
        
    #     if self.count % self.accu_batch == 0:
            
    #         loss.backward()

    #         self.optimizer.step()
    #         self.update_baseline(reward)
    #         self.initialize_step()
    #         self.scheduler.step()
    #         self.current_lr = self.scheduler.get_last_lr()[0]
    #         return reward, optimized_acc_adv, acc_adv, ent, loss
    #     else:
    #         return 0, 0, 0, 0, 0

    def step(self, input_valid, target_valid):
        n = input_valid.size(0)
        self.count = self.count + 1
        self.optimizer.zero_grad()
        loss, reward, optimized_acc_adv, acc_adv, ent = self.model._loss_transformer(self.twin_supernet, input_valid, target_valid, self.baseline, stop = (self.count % self.accu_batch == 0))
        
        if self.count % self.accu_batch == 0:
            
            loss.backward()

            self.optimizer.step()
            self.update_baseline(reward)
            self.initialize_step()
            self.scheduler.step()
            self.current_lr = self.scheduler.get_last_lr()[0]
            return reward, optimized_acc_adv, acc_adv, ent, loss
        else:
            return 0, 0, 0, 0, 0

        