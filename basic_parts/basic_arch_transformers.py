import genotypes
from operations import *
import utils
from pygcn.layers import GraphConvolution
import torch
import utils
import genotypes
from operations import *
import utils
from pygcn.layers import GraphConvolution
from genotypes import LooseEnd_Transition_Dict, FullyConcat_Transition_Dict


class ArchTransformer(nn.Module):
    def __init__(self, steps, device, edge_hid, nfeat, gcn_hid, dropout, normalize=False, op_type="FULLY_CONCAT_PRIMITIVES"):
        """
        :param nfeat: feature dimension of each node in the graph
        :param nhid: hidden dimension
        :param dropout: dropout rate for GCN
        """
        super(ArchTransformer, self).__init__()
        self.steps = steps
        self.device = device
        self.normalize = normalize
        self.op_type = op_type
        num_ops = len(genotypes.TRANSFORM_PRIMITIVES)
        self.gc1 = GraphConvolution(nfeat, gcn_hid)
        self.gc2 = GraphConvolution(gcn_hid, gcn_hid)
        self.dropout = dropout
        self.fc = nn.Linear(gcn_hid, num_ops * 2)

        try:
            COMPACT_PRIMITIVES = eval("genotypes.%s" % op_type)
        except:
            assert False, "not supported op type %s" % (op_type)

        # the first two nodes
        self.node_hidden = nn.Embedding(2, 2 * edge_hid)
        self.op_hidden = nn.Embedding(len(COMPACT_PRIMITIVES), edge_hid)
        # [op0, op1]
        self.emb_attn = nn.Linear(2 * edge_hid, nfeat)

    def forward(self, arch):
        # initial the first two nodes
        op0_list = []
        op1_list = []
        for idx, (op, f, t) in enumerate(arch):
            if idx % 2 == 0:
                op0_list.append(op)
            else:
                op1_list.append(op)
        assert len(op0_list) == len(op1_list), "inconsistent size between op0_list and op1_list"
        node_list = utils.get_variable(list(range(0, 2, 1)), self.device, requires_grad=False)
        op0_list = utils.get_variable(op0_list, self.device, requires_grad=False)
        op1_list = utils.get_variable(op1_list, self.device, requires_grad=False)
        # first two nodes
        x_node_hidden = self.node_hidden(node_list)
        x_op0_hidden = self.op_hidden(op0_list)
        x_op1_hidden = self.op_hidden(op1_list)
        """
            node0
            node1
            op0, op1
        """
        x_op_hidden = torch.cat((x_op0_hidden, x_op1_hidden), dim=1)
        x_hidden = torch.cat((x_node_hidden, x_op_hidden), dim=0)
        # initialize x and adj
        x = self.emb_attn(x_hidden)
        adj = utils.parse_arch(arch, self.steps + 2).to(self.device)
        # normalize features and adj
        if self.normalize:
            x = utils.sum_normalize(x)
            adj = utils.sum_normalize(adj)
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        x = x[2:]
        logits = self.fc(x)
        logits = logits.view(self.steps * 2, -1)
        probs = F.softmax(logits, dim=-1)
        probs = probs + 1e-5
        log_probs = torch.log(probs)
        action = probs.multinomial(num_samples=1)
        selected_log_p = log_probs.gather(-1, action)
        log_p = selected_log_p.sum()
        entropy = -(log_probs * probs).sum()
        arch = utils.translate_arch(arch, action, self.op_type)
        return arch, log_p, entropy, probs


class ArchTransformerGates(nn.Module):
    def __init__(self, steps, device, edge_hid, nfeat, gcn_hid, dropout, normalize=False, op_type="FULLY_CONCAT_PRIMITIVES", transform_type_more=True):
        """

        :param nfeat: feature dimension of each node in the graph
        :param nhid: hidden dimension
        :param dropout: dropout rate for GCN
        """
        super(ArchTransformerGates, self).__init__()
        self.steps = steps
        self.device = device
        self.normalize = normalize
        self.op_type = op_type
        if transform_type_more:
            if op_type == "LOOSE_END_PRIMITIVES":
                num_ops = len(genotypes.LOOSE_END_PRIMITIVES)
            else:
                num_ops = len(genotypes.FULLY_CONCAT_PRIMITIVES)
        else:
            num_ops = len(genotypes.TRANSFORM_PRIMITIVES)
        self.gc1 = GraphConvolution(nfeat, gcn_hid)
        self.gc2 = GraphConvolution(gcn_hid, gcn_hid)
        self.dropout = dropout
        self.fc = nn.Linear(gcn_hid, num_ops * 2)

        try:
            COMPACT_PRIMITIVES = eval("genotypes.%s" % op_type)
        except:
            assert False, "not supported op type %s" % (op_type)

        # the first two nodes
        self.node_hidden = nn.Embedding(2, 2 * edge_hid)
        self.op_hidden = nn.Embedding(len(COMPACT_PRIMITIVES), edge_hid)
        # [op0, op1]
        self.emb_attn = nn.Linear(2 * edge_hid, nfeat)

    def forward(self, arch):
        # initial the first two nodes
        op0_list = []
        op1_list = []
        for idx, (op, f, t) in enumerate(arch):
            if idx % 2 == 0:
                op0_list.append(op)
            else:
                op1_list.append(op)
        assert len(op0_list) == len(op1_list), "inconsistent size between op0_list and op1_list"
        node_list = utils.get_variable(list(range(0, 2, 1)), self.device, requires_grad=False)
        op0_list = utils.get_variable(op0_list, self.device, requires_grad=False)
        op1_list = utils.get_variable(op1_list, self.device, requires_grad=False)
        # first two nodes
        x_node_hidden = self.node_hidden(node_list)
        x_op0_hidden = self.op_hidden(op0_list)
        x_op1_hidden = self.op_hidden(op1_list)
        """
            node0
            node1
            op0, op1
        """
        x_op_hidden = torch.cat((x_op0_hidden, x_op1_hidden), dim=1)
        x_hidden = torch.cat((x_node_hidden, x_op_hidden), dim=0)
        # initialize x and adj
        x = self.emb_attn(x_hidden)
        adj = utils.parse_arch(arch, self.steps + 2).to(self.device)
        # normalize features and adj
        if self.normalize:
            x = utils.sum_normalize(x)
            adj = utils.sum_normalize(adj)
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        x = x[2:]
        logits = self.fc(x)
        logits = logits.view(self.steps * 2, -1)
        entropy = 0
        log_p = 0
        arch_list = []
        try:
            COMPACT_PRIMITIVES = eval("genotypes.{}".format(self.op_type))
        except:
            assert False, "not supported op type %s" % (self.op_type)
        transition_dict = LooseEnd_Transition_Dict if self.op_type == "LOOSE_END_PRIMITIVES" else FullyConcat_Transition_Dict

        prob_mat = torch.zeros(len(COMPACT_PRIMITIVES), len(arch), device=self.device)

        for idx, (op, f, t) in enumerate(arch):
            select_op = transition_dict[COMPACT_PRIMITIVES[op]]
            selected_arch_index = [COMPACT_PRIMITIVES.index(i) for i in select_op]
            tmp = logits[idx]
            V = tmp.new_zeros(tmp.size(), requires_grad=False)
            V[selected_arch_index] = 1
            prob = utils.BinarySoftmax(tmp, V)
            prob_mat[:, idx] = prob
            log_prob = torch.log(torch.clamp(prob, min=1e-5, max=1 - 1e-5))
            entropy += -(log_prob * prob).sum()
            f_op = prob.multinomial(num_samples=1)
            selected_log_p = log_prob.gather(-1, f_op)
            log_p += selected_log_p.sum()
            arch_list.append((f_op, f, t))
        utils.check_transform(arch, arch_list, self.op_type)
        return arch_list, log_p, entropy, prob_mat.requires_grad_()
