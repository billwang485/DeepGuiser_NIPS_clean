from pyexpat import model
import torch
import utils

class Transformer(object):
    def __init__(self, model, args):
        self.args = args
        self.model = model
        if not args.only_mlp:
            self.optimizer = torch.optim.Adam(self.model.transformer_parameters(),
                                            lr=args.lr, betas=(0.5, 0.999),
                                            weight_decay=args.transformer_weight_decay)
        else:
            self.optimizer = torch.optim.Adam(self.model.arch_transformer.fcs.parameters(),
                                            lr=args.lr, betas=(0.5, 0.999),
                                            weight_decay=args.transformer_weight_decay)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #      self.optimizer, float(args.its), eta_min=args.lr_min
        #  )
        # gamma = (args.lr_min / args.lr)**(1 / args.its)
        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma= gamma)
        step_size = 10000
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size, gamma=0.25)
        self.baseline = 0
        self.gamma = args.gamma
        # self.initialize_step()
        self.current_lr = args.lr

    def update_baseline(self, reward):
        self.baseline = self.baseline * self.gamma + reward * (1-self.gamma)

    def step(self):
        # print(self.model.predictor.training)
        self.optimizer.zero_grad()
        loss, reward, ent, flops_limit, op_div, nlit = self.model._loss_transformer(self.baseline)
        loss.backward()
        # for name, parms in self.model.arch_transformer.named_parameters():
        #     print('-->name:', name, '-->grad_requirs:',parms.requires_grad, ' -->grad_value:',parms.grad)

        self.optimizer.step()
        # self.optimizer.zero_grad()
        self.update_baseline(reward)
        # self.scheduler.step()
        self.current_lr = self.scheduler.get_last_lr()[0]
        return reward, ent, loss, flops_limit, op_div, nlit

        