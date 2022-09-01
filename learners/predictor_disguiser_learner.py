import torch
import utils

class PredictorBasedLearner():
    def __init__(self, model, optimizer_config, strategy_config):
        self.model = model
        self.optimizer = utils.initialize_optimizer(model.arch_transformer.parameters(), optimizer_config["learning_rate"], optimizer_config["momentum"], optimizer_config["weight_decay"], optimizer_config["type"])
        self.scheduler = utils.initialize_scheduler(self.optimizer, optimizer_config["scheduler_info"])
        self.baseline = 0
        self.current_lr = optimizer_config["learning_rate"]
    
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def step(self):
        self.optimizer.zero_grad()
        loss, reward, ent, auxillary_info = self.model._loss_transformer(self.baseline)
        loss.backward()

        self.optimizer.step()
        # self.update_baseline(reward)
        # self.scheduler.step()
        self.current_lr = self.scheduler.get_last_lr()[0]
        return loss, reward, ent, auxillary_info

        