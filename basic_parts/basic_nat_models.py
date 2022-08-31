import utils
import genotypes
from operations import *


class NASOp(nn.Module):
    def __init__(self, C, stride, op_type):
        super(NASOp, self).__init__()
        self._ops = nn.ModuleList()
        try:
            COMPACT_PRIMITIVES = eval("genotypes.%s" % op_type)
        except:
            assert False, "not supported op type %s" % (op_type)
        for primitive in COMPACT_PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            if "pool" in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)

    def forward(self, x, index):
        return self._ops[index](x)


class NASCell(nn.Module):
    def __init__(self, steps, device, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, loose_end=False, concat=None, op_type="FULLY_CONCAT_PRIMITIVES"):
        super(NASCell, self).__init__()
        self.steps = steps
        self.device = device
        self.multiplier = multiplier
        self.C = C
        self.reduction = reduction
        self.loose_end = loose_end
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)

        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps

        self._ops = nn.ModuleList()
        for i in range(self._steps):
            for j in range(i + 2):
                stride = 2 if reduction and j < 2 else 1
                op = NASOp(C, stride, op_type)
                self._ops.append(op)

        self._concat = concat

    def forward(self, s0, s1, arch):
        """
        :param s0:
        :param s1:
        :param arch: a list, the element is (op_id, from_node, to_node), sorted by to_node (!!not check
                     the ordering for efficiency, but must be assured when generating!!)
                     from_node/to_node starts from 0, 0 is the prev_prev_node, 1 is prev_node
                     The mapping from (F, T) pair to edge_ID is (T-2)(T+1)/2+S,
        :return:
        """
        s0 = self.preprocess0.forward(s0)
        s1 = self.preprocess1.forward(s1)
        states = {0: s0, 1: s1}
        used_nodes = set()
        for op, f, t in arch:
            edge_id = int((t - 2) * (t + 1) / 2 + f)
            if t in states:
                states[t] = states[t] + self._ops[edge_id](states[f], op)
            else:
                states[t] = self._ops[edge_id](states[f], op)
            used_nodes.add(f)
        if self._concat is not None:
            state_list = []
            for i in range(2, self._steps + 2):
                if i in self._concat:
                    state_list.append(states[i])
                else:
                    state_list.append(states[i] * 0)
            return torch.cat(state_list, dim=1)
        else:
            if self.loose_end:
                state_list = []
                for i in range(2, self._steps + 2):
                    if i not in used_nodes:
                        state_list.append(states[i])
                    else:
                        state_list.append(states[i] * 0)
                return torch.cat(state_list, dim=1)
            else:
                return torch.cat([states[i] for i in range(2, self._steps + 2)], dim=1)


class ArchMaster(nn.Module):
    def __init__(self, n_ops, n_nodes, device, controller_hid=100, lstm_num_layers=2):
        super(ArchMaster, self).__init__()
        self.K = sum([x + 2 for x in range(n_nodes)])
        self.n_ops = n_ops
        self.n_nodes = n_nodes
        self.device = device

        self.controller_hid = controller_hid
        self.attention_hid = self.controller_hid
        self.lstm_num_layers = lstm_num_layers
        self.node_op_hidden = nn.Embedding(n_nodes + 1 + n_ops, self.controller_hid)
        self.emb_attn = nn.Linear(self.controller_hid, self.attention_hid, bias=False)
        self.hid_attn = nn.Linear(self.controller_hid, self.attention_hid, bias=False)
        self.v_attn = nn.Linear(self.controller_hid, 1, bias=False)
        self.w_soft = nn.Linear(self.controller_hid, self.n_ops)
        self.lstm = nn.LSTMCell(self.controller_hid, self.controller_hid)
        self.reset_parameters()
        self.static_init_hidden = utils.keydefaultdict(self.init_hidden)
        self.static_inputs = utils.keydefaultdict(self._get_default_hidden)
        self.tanh = nn.Tanh()
        self.prev_nodes, self.prev_ops = [], []
        self.query_index = torch.LongTensor(range(0, n_nodes + 1)).to(device)
        self.demo = False
        self.analyze = False

    def _get_default_hidden(self, key):
        return utils.get_variable(torch.zeros(key, self.controller_hid), self.device, requires_grad=False)
    
    def use_demo(self, value = True):
        self.demo = value

    # device
    def init_hidden(self, batch_size):
        zeros = torch.zeros(batch_size, self.controller_hid)
        return (
            utils.get_variable(zeros, self.device, requires_grad=False),
            utils.get_variable(zeros.clone(), self.device, requires_grad=False),
        )

    def reset_parameters(self):
        init_range = 0.1
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)
        self.w_soft.bias.data.fill_(0)

    def forward(self):
        flag = 0
        while not flag:
            self.prev_nodes, self.prev_ops = [], []
            batch_size = 1
            inputs = self.static_inputs[batch_size]  # batch_size x hidden_dim

            for node_idx in range(self.n_nodes):
                for i in range(2):  # index_1, index_2
                    if node_idx == 0 and i == 0:
                        embed = inputs
                    else:
                        embed = self.node_op_hidden(inputs)
                    # force uniform
                    probs = F.softmax(torch.zeros(node_idx + 2).type_as(embed), dim=-1)
                    action = probs.multinomial(num_samples=1)
                    self.prev_nodes.append(action)
                    inputs = utils.get_variable(action, self.device, requires_grad=False)
                for i in range(2):  # op_1, op_2
                    embed = self.node_op_hidden(inputs)
                    # force uniform
                    probs = F.softmax(torch.zeros(self.n_ops).type_as(embed), dim=-1)
                    if self.demo:
                        probs[-1] = probs[0] * 5
                        probs[1] = probs[0] * 2
                        probs[2] = probs[0] * 2
                    elif self.analyze:
                        probs[1] = probs[0] * 1
                        probs[2] = probs[0] * 1
                        probs[-1] = probs[0] * 1.3
                    action = probs.multinomial(num_samples=1)
                    self.prev_ops.append(action)
                    inputs = utils.get_variable(action + self.n_nodes + 1, self.device, requires_grad=False)
            arch = utils.convert_lstm_output(self.n_nodes, torch.cat(self.prev_nodes), torch.cat(self.prev_ops))

            if self.demo or self.analyze:
                import random

                for step, (op, f, t) in enumerate(arch):
                    if step % 2 == 0:
                        if op == arch[step + 1][0] and op == 8:
                            op == random.randint(0, 7)
                    pos = step
                    if pos >= len(arch) - 2:
                        break
                    flag = 0
                    for i in range(pos + 1, len(arch)):
                        (op1, f1, _) = arch[i]
                        if f1 == t and op1 != 8:
                            flag = 1
                    if not flag:
                        arch[pos + 2] = (random.randint(0, 7), t, arch[pos + 2][2])
            flag = utils.check_connectivity(arch)
            connectivity = utils.get_connectivity(arch)
            for i in range(2, 5):
                if not utils.check_connectivity_transform(connectivity, i):
                    flag = False
                    break

        return arch
