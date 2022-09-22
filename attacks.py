# classic pgd is in utils

from this import d
from autoattack import AutoAttack
from nn_robust_attacks.li_attack import CarliniLi
import tensorflow as tf
import torch
import torch.nn as nn
import utils

def adversarial_test(target_model, surrogate_model, baseline_model, test_queue, attack_info, logger):
    device = next(target_model.parameters()).device
    acc_adv_baseline = utils.AvgrageMeter()
    acc_adv_surrogate = utils.AvgrageMeter()

    for step, (input, target) in enumerate(test_queue):
        n = input.size(0)
        input = input.to(device)
        target = target.to(device)
        target_model.eval()
        baseline_model.eval()
        surrogate_model.eval()

        input_adv0 = unified_adversarial_generation(baseline_model, input, target, attack_info)
        # logits0, _ = target_model(input_adv0)
        logits0 = utils.unified_forward(target_model, input_adv0)
        acc_adv_baseline_ = utils.accuracy(logits0, target, topk=(1, 5))[0] / 100.0
        acc_adv_baseline.update(acc_adv_baseline_.item(), n)

        input_adv1 = unified_adversarial_generation(surrogate_model, input, target, attack_info)
        # logging.info("acc_adv_target_white=%.2f", acc_adv.item())
        # logits1, _ = target_model(input_adv1)
        logits1 = utils.unified_forward(target_model, input_adv1)
        acc_adv_surrogate_ = utils.accuracy(logits1, target, topk=(1, 5))[0] / 100.0
        acc_adv_surrogate.update(acc_adv_surrogate_.item(), n)
    if logger is not None:
        logger.info('PGD Test Results: acc_adv_baseline=%f acc_adv_surrogate=%.2f',\
                acc_adv_baseline.avg, acc_adv_surrogate.avg)
    return acc_adv_baseline.avg, acc_adv_surrogate.avg

def unified_adversarial_generation(generate_model, input, target, attack_info):
    device = next(generate_model.parameters()).device
    if attack_info["type"] == "PGD":
        x_adv =  classic_pgd_attack(generate_model, input, target, attack_info['eps'], attack_info['alpha'], attack_info['step'])
    elif attack_info["type"] == "auto_attack":
        x_adv = auto_attack(generate_model, attack_info['eps'], input, target)
    elif attack_info["type"] == "carlini_wagner":
        x_adv = carlini_wagner_attack(generate_model, input, target)
    else:
        assert 0

    return x_adv.to(device)


def auto_attack(model, epsilon, inputs, labels):
    assert model.model_type == "compiled_based" or model.model_type == "supernet_based"
    adversary = AutoAttack(model, norm='Linf', eps=epsilon, version='standard')
    x_adv = adversary.run_standard_evaluation(inputs, labels, bs=input.size(0))
    return x_adv

def carlini_wagner_attack(model, inputs, labels):
    assert model.model_type == "compiled_based" or model.model_type == "supernet_based"
    x_adv = CarliniLi(tf.session(), model).attack(inputs.cpu().numpy(), labels.cpu().numpy())
    return torch.tensor(x_adv)

def classic_pgd_attack(generator_model, input, target, eps, alpha, steps, is_targeted=False, rand_start=True, momentum=False, mu=1, criterion=nn.CrossEntropyLoss()):
    
    def _gradient_wrt_input(model, inputs, targets, criterion=nn.CrossEntropyLoss()):
        inputs.requires_grad = True

        outputs = utils.unified_forward(model ,inputs)
        loss = criterion(outputs, targets)
        model.zero_grad()
        loss.backward()

        data_grad = inputs.grad.data
        return data_grad.clone().detach()
    generator_model.eval()
    x_nat = input.clone().detach()
    x_adv = None
    if rand_start:
        x_adv = input.clone().detach() + torch.FloatTensor(input.shape).uniform_(-eps, eps).cuda()
    else:
        x_adv = input.clone().detach()
    x_adv = torch.clamp(x_adv, 0.0, 1.0)  # respect image bounds
    g = torch.zeros_like(x_adv)

    # Iteratively Perturb data
    for i in range(int(steps)):
        # Calculate gradient w.r.t. data
        # print("hhhh")
        grad = _gradient_wrt_input(generator_model, x_adv, target, criterion)
        with torch.no_grad():
            if momentum:
                # Compute sample wise L1 norm of gradient
                flat_grad = grad.view(grad.shape[0], -1)
                l1_grad = torch.norm(flat_grad, 1, dim=1)
                grad = grad / torch.clamp(l1_grad, min=1e-12).view(grad.shape[0], 1, 1, 1)
                # Accumulate the gradient
                new_grad = mu * g + grad  # calc new grad with momentum term
                g = new_grad
            else:
                new_grad = grad
            # Get the sign of the gradient
            sign_data_grad = new_grad.sign()
            if is_targeted:
                x_adv = x_adv - alpha * sign_data_grad  # perturb the data to MINIMIZE loss on tgt class
            else:
                x_adv = x_adv + alpha * sign_data_grad  # perturb the data to MAXIMIZE loss on gt class
            # Clip the perturbations w.r.t. the original data so we still satisfy l_infinity
            # x_adv = torch.clamp(x_adv, x_nat-eps, x_nat+eps) # Tensor min/max not supported yet
            x_adv = torch.max(torch.min(x_adv, x_nat + eps), x_nat - eps)
            # Make sure we are still in bounds
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv.clone().detach()
