from operations import *
import utils
import numpy as np
from basic_parts.basic_gates_models import GCNFlowArchEmbedder
from scipy.stats import kendalltau


class VanillaGatesPredictor(nn.Module):
    def __init__(self, device, args, normalize=False, dropout=0):
        super(VanillaGatesPredictor, self).__init__()
        self.steps = args.num_nodes
        self.device = device
        self.normalize = normalize
        self.op_type = args.op_type
        (self.transform2gates, self.transform2nat, self.gates_op_list) = utils.primitives_translation(self.op_type)
        self.arch_embedder = GCNFlowArchEmbedder(self.gates_op_list, node_dim=32, op_dim=32, use_bn=False, hidden_dim=32, gcn_out_dims=[64] * 4, gcn_kwargs={"residual_only": 2})

        self.fcs = nn.Sequential(
            nn.Linear(256, 64, bias=False),
            nn.ReLU(inplace=False),
            nn.Dropout(p=dropout),

            nn.Linear(64, 1, bias=False),
        )


        if hasattr(args, "mode") and hasattr(args, "loss_type") and hasattr(args, "use_sch"):
            self.mode = args.mode

            if self.mode == "low_fidelity":
                self.optimizer = torch.optim.Adam(
                    self.parameters(),
                    lr=args.lr,
                    betas=(0.5, 0.999),
                    weight_decay=args.transformer_weight_decay,
                )
            elif self.mode == "high_fidelity":
                self.optimizer = torch.optim.Adam(
                    self.parameters(),
                    args.lr,
                    weight_decay=args.transformer_weight_decay,
                )
            self.loss_type = args.loss_type
        else:
            self.eval()
        self.rt = args.rt
        self.compare_margin = 0.01
        self.max_grad_norm = 5.0
        self.margin_l2 = False
        self.epoch = 0

    def forward(self, target_arch, surrogate_arch, gates_input=False):
        if not gates_input:
            (arch_normal_t, arch_reduce_t) = target_arch
            (arch_normal_s, arch_reduce_s) = surrogate_arch
            assert len(arch_normal_t) == len(arch_reduce_t) and len(arch_normal_s) == len(arch_reduce_s)
            gates_arch_t = utils.nat_arch_to_gates_p(arch_normal_t, arch_reduce_t, self.transform2gates)
            gates_arch_s = utils.nat_arch_to_gates_p(arch_normal_s, arch_reduce_s, self.transform2gates)
        else:
            gates_arch_t = target_arch
            gates_arch_s = surrogate_arch
        x1 = self.arch_embedder(gates_arch_t)
        x2 = self.arch_embedder(gates_arch_s)
        # x = x1 - x2
        x = torch.cat((x1, x2), 1)
        score = self.fcs(x)
        score = torch.tanh(score)
        return score

    def step_mse(self, archs, label):
        label = label.to(self.device)
        scores = self.forward(archs["target_arch"], archs["surrogate_arch"])
        loss = self._mse_loss(scores, label) * 100
        return loss, scores

    def step_compare(self, archs, label):
        scores = self.forward(archs["target_arch"], archs["surrogate_arch"])
        (arch_normal_t, arch_reduce_t) = archs["target_arch"]
        (arch_normal_s, arch_reduce_s) = archs["surrogate_arch"]
        gates_arch_t = utils.nat_arch_to_gates_p(arch_normal_t, arch_reduce_t, self.transform2gates)
        gates_arch_s = utils.nat_arch_to_gates_p(arch_normal_s, arch_reduce_s, self.transform2gates)
        archs = torch.arange(gates_arch_t.shape[0])
        index_1, index_2, better_list = utils.compare_data(archs, label)
        scores1 = self.forward(gates_arch_t[index_1], gates_arch_s[index_1], gates_input=True)
        scores2 = self.forward(gates_arch_t[index_2], gates_arch_s[index_2], gates_input=True)
        loss = self._pair_loss(scores1.squeeze(), scores2.squeeze(), better_list)
        return loss, scores

    def step(self, archs, label):
        if not self.training:
            self.train()
        if self.loss_type == "mse":
            loss, scores = self.step_mse(archs, label)
        elif self.loss_type == "ranking":
            loss, scores = self.step_compare(archs, label)
        elif self.loss_type == "mix":
            loss1, scores = self.step_mse(archs, label)
            loss2, _ = self.step_compare(archs, label)
            loss = loss2 + 0.2 * loss1
        else:
            assert 0
        self.update_step(loss)
        kendall = kendalltau(label.squeeze().tolist(), scores.squeeze().tolist()).correlation
        return loss, scores, kendall

    def test(self, test_queue, logger):
        if self.training:
            self.eval()
        avg_loss = utils.AvgrageMeter()
        scores = []
        reals = []
        count = 0
        success_count = 0
        for step, data_point in enumerate(test_queue):
            if self.rt == "a":
                label = data_point["absolute_reward"]
            elif self.rt == "r":
                label = data_point["relative_reward"]
            else:
                if self.mode == "high_fidelity":
                    label = data_point["acc_adv_surrogate"]
                else:
                    label = data_point["acc_adv"]
            if self.loss_type == "mse":
                loss, score = self.step_mse(data_point, label)
            elif self.loss_type == "ranking":
                loss, score = self.step_compare(data_point, label)
            elif self.loss_type == "mix":
                loss1, score = self.step_mse(data_point, label)
                loss2, _ = self.step_compare(data_point, label)
                loss = loss2 + 0.2 * loss1
            else:
                assert 0
            scores.extend(score.squeeze().tolist())
            reals.extend(label.squeeze().tolist())
            avg_loss.update(loss.item(), 1)
            tmp1 = score.squeeze().tolist()
            tmp2 = label.squeeze().tolist()
            count = count + 1
            if tmp1.index(max(tmp1)) == tmp2.index(max(tmp2)):
                success_count = success_count + 1
        if self.mode == "low_fidelity":
            tmp = 100
            patk = utils.patk(reals, scores, tmp)
        else:
            tmp = 10
            patk = success_count / count

        kendalltau_, _ = kendalltau(reals, scores)
        logger.info("testing predictors on %d transforms with %s mode", len(scores), self.mode)
        logger.info(
            "average loss=%.4f patk=%.2f kendalltau=%.4f",
            avg_loss.avg,
            patk,
            kendalltau_,
        )
        return avg_loss.avg, patk, kendalltau_

    def _mse_loss(self, scores, labels):
        return F.mse_loss(scores.squeeze(), scores.new(labels.float()))

    def _pair_loss(self, scores1, scores2, better_list):
        s_1 = scores1.requires_grad_()
        s_2 = scores2.requires_grad_()
        better_pm = 2 * s_1.new(np.array(better_list, dtype=np.float32)) - 1
        zero_ = s_1.new([0.0])
        zero_.requires_grad_()
        margin = 0.1
        margin = s_1.new([margin])
        margin.requires_grad_()
        if not self.margin_l2:
            pair_loss = torch.mean(torch.max(zero_, margin - better_pm * (s_2 - s_1))) * len(s_1)
        else:
            pair_loss = torch.mean(torch.max(zero_, margin - better_pm * (s_2 - s_1)) ** 2 / np.maximum(1.0, margin))

        return pair_loss

    def update_step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self._clip_grads()
        self.optimizer.step()
        return loss.item()

    def _clip_grads(self):
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
