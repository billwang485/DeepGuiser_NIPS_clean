import math
import numpy as np

from operations import *


class DenseGraphOpEdgeFlow(nn.Module):
    """
    For search space that has operation on the edge.
    """

    def __init__(
        self,
        in_features,
        out_features,
        op_emb_dim,
        has_attention=True,
        plus_I=False,
        share_self_op_emb=False,
        normalize=False,
        bias=False,
        residual_only=None,
        use_sum=False,
        concat=None,
        has_aggregate_op=False,
    ):
        super(DenseGraphOpEdgeFlow, self).__init__()

        self.plus_I = plus_I
        self.share_self_op_emb = share_self_op_emb
        self.residual_only = residual_only
        self.normalize = normalize
        self.in_features = in_features
        self.out_features = out_features
        self.op_emb_dim = op_emb_dim
        self.use_sum = use_sum
        # self.concat = concat
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.has_aggregate_op = has_aggregate_op
        if self.has_aggregate_op:
            self.aggregate_op = nn.Linear(out_features, out_features)
        if has_attention:
            self.op_attention = nn.Linear(op_emb_dim, out_features)
        else:
            assert self.op_emb_dim == self.out_features
            self.op_attention = nn.Identity()
        if self.plus_I and not self.share_self_op_emb:
            self.self_op_emb = nn.Parameter(torch.FloatTensor(op_emb_dim))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        # if self.concat is not None:
        #     assert isinstance(self.concat, int)
        #     self.concats
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs, adj, adj_op_inds_lst, op_emb, zero_index, self_op_emb=None):
        # support: (b, n_cg, V, h_i)
        support = torch.matmul(inputs, self.weight)
        op_emb_adj_lst = [F.embedding(adj_op_inds, op_emb) for adj_op_inds in adj_op_inds_lst]
        attn_mask_inds_lst = [(adj_op_inds == zero_index).unsqueeze(-1) for adj_op_inds in adj_op_inds_lst]
        if self.plus_I:
            eye_mask = support.new(np.eye(adj.shape[-1])).unsqueeze(-1).bool()
            # for i in range(len(adj_op_inds_lst)):
            #     op_emb_adj_lst[i] = torch.where(eye_mask, self.self_op_emb, op_emb_adj_lst[i])
            #     attn_mask_inds_lst[i] = attn_mask_inds_lst[i] & (~eye_mask.bool())
            self_op_emb = self_op_emb if self.share_self_op_emb else self.self_op_emb
            op_emb_adj_lst[0] = torch.where(eye_mask, self_op_emb, op_emb_adj_lst[0])
            attn_mask_inds_lst[0] = attn_mask_inds_lst[0] & (~eye_mask)

        # attn_mask_inds_stack: (n_d, b, n_cg, V, V, 1)
        attn_mask_inds_stack = torch.stack(attn_mask_inds_lst)
        # ob_emb_adj_stack: (n_d, b, n_cg, V, V, h_o)
        op_emb_adj_stack = torch.stack(op_emb_adj_lst)

        attn = torch.sigmoid(self.op_attention(op_emb_adj_stack))
        attn = torch.where(attn_mask_inds_stack, attn.new(1, 1, 1, 1, 1, attn.shape[-1]).zero_(), attn)
        # attn: (n_d, b, n_cg, V, V, h_o)

        # output = (adj_aug.unsqueeze(-1) * attn \
        #           * support.unsqueeze(2)).sum(-2) + support
        if self.residual_only is None:
            res_output = support
        else:
            res_output = torch.cat(
                (
                    support[:, :, : self.residual_only, :],
                    torch.zeros(
                        [
                            support.shape[0],
                            support.shape[1],
                            support.shape[2] - self.residual_only,
                            support.shape[3],
                        ],
                        device=support.device,
                    ),
                ),
                dim=2,
            )
        processed_info = (attn * support.unsqueeze(2)).sum(-2)
        processed_info = processed_info.sum(0) if self.use_sum else processed_info.mean(0)
        if self.has_aggregate_op:
            output = self.aggregate_op(processed_info) + res_output
        else:
            output = processed_info + res_output
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + " (" + str(self.in_features) + " -> " + str(self.out_features) + ")"


class GCNFlowArchEmbedder(nn.Module):
    NAME = "cellss-flow"

    def __init__(
        self,
        op_list,
        node_dim=32,
        op_dim=32,
        hidden_dim=32,
        gcn_out_dims=[64] * 4,
        other_node_zero=False,
        gcn_kwargs=None,
        dropout=0.0,
        normalize=False,
        use_bn=False,
        other_node_independent=False,
        share_self_op_emb=False,
        final_concat=False,
        schedule_cfg=None,
    ):
        super(GCNFlowArchEmbedder, self).__init__()

        # self.search_space = search_space

        # configs
        self.normalize = normalize
        self.node_dim = node_dim
        self.op_dim = op_dim
        self.hidden_dim = hidden_dim
        self.gcn_out_dims = gcn_out_dims
        self.dropout = dropout
        self.use_bn = use_bn
        self.other_node_independent = other_node_independent
        self.share_self_op_emb = share_self_op_emb
        # final concat only support the cell-ss that all nodes are concated
        # (loose-end is not supported)
        self.final_concat = final_concat

        self._num_init_nodes = 2
        self._num_node_inputs = 2
        self._num_steps = 4
        self._num_nodes = self._num_steps + self._num_init_nodes
        self._num_cg = 2  # normal reductin

        # different init node embedding for different cell groups
        # but share op embedding for different cell groups
        # maybe this should be separated? at least for stride-2 op and stride-1 op
        if self.other_node_independent:
            self.init_node_emb = nn.Parameter(torch.Tensor(self._num_cg, self._num_nodes, self.node_dim).normal_())
        else:
            # other nodes share init embedding
            self.init_node_emb = nn.Parameter(torch.Tensor(self._num_cg, self._num_init_nodes, self.node_dim).normal_())
            self.other_node_emb = nn.Parameter(
                torch.zeros(self._num_cg, 1, self.node_dim),
                requires_grad=not other_node_zero,
            )

        # op_list = op_list

        self.num_ops = len(op_list)
        try:
            self.none_index = op_list.index("none")
            self.add_none_index = False
            assert self.none_index == 0, "search space with none op should have none op as the first primitive"
        except ValueError:
            self.none_index = len(op_list)
            self.none_index = 0
            self.add_none_index = True
            self.num_ops += 1

        self.op_emb = []
        for idx in range(self.num_ops):
            if idx == self.none_index:
                emb = nn.Parameter(torch.zeros(self.op_dim), requires_grad=False)
            else:
                emb = nn.Parameter(torch.Tensor(self.op_dim).normal_())
            setattr(self, "op_embedding_{}".format(idx), emb)
            self.op_emb.append(emb)
        if self.share_self_op_emb:
            self.self_op_emb = nn.Parameter(torch.FloatTensor(self.op_dim).normal_())
        else:
            self.self_op_emb = None

        self.x_hidden = nn.Linear(self.node_dim, self.hidden_dim)

        # init graph convolutions
        self.gcns = []
        self.bns = []
        in_dim = self.hidden_dim
        for dim in self.gcn_out_dims:
            self.gcns.append(DenseGraphOpEdgeFlow(in_dim, dim, self.op_dim, **(gcn_kwargs or {})))
            in_dim = dim
            if self.use_bn:
                self.bns.append(nn.BatchNorm1d(self._num_nodes * self._num_cg))
        self.gcns = nn.ModuleList(self.gcns)
        if self.use_bn:
            self.bns = nn.ModuleList(self.bns)
        self.num_gcn_layers = len(self.gcns)
        if not self.final_concat:
            self.out_dim = self._num_cg * in_dim
        else:
            self.out_dim = self._num_cg * in_dim * self._num_steps

    def get_adj_dense(self, arch):
        return self._get_adj_dense(
            arch,
            self._num_init_nodes,
            self._num_node_inputs,
            self._num_nodes,
            self.none_index,
        )

    def _get_adj_dense(self, arch, num_init_nodes, num_node_inputs, num_nodes, none_index):  # pylint: disable=no-self-use
        """
        get dense adjecent matrix, could be batched
        """
        f_nodes = np.array(arch[:, 0, :])
        # n_d: input degree (num_node_inputs)
        # ops: (b_size * n_cg, n_steps * n_d)
        ops = np.array(arch[:, 1, :])
        if self.add_none_index:
            ops = ops + 1
        _ndim = f_nodes.ndim
        if _ndim == 1:
            f_nodes = np.expand_dims(f_nodes, 0)
            ops = np.expand_dims(ops, 0)
        else:
            assert _ndim == 2
        batch_size = f_nodes.shape[0]
        t_nodes = np.tile(
            np.repeat(np.array(range(num_init_nodes, num_nodes)), num_node_inputs)[None, :],
            [batch_size, 1],
        )
        batch_inds = np.tile(np.arange(batch_size)[:, None], [1, t_nodes.shape[1]])
        ori_indexes = np.stack((batch_inds, t_nodes, f_nodes))
        indexes = ori_indexes.reshape([3, -1])
        indexes, edge_counts = np.unique(indexes, return_counts=True, axis=1)
        adj = torch.zeros(batch_size, num_nodes, num_nodes)
        adj[indexes] += torch.tensor(edge_counts, dtype=torch.float32)
        adj_op_inds_lst = [torch.ones(batch_size, num_nodes, num_nodes, dtype=torch.long) * none_index for _ in range(num_node_inputs)]
        ori_indexes_lst = np.split(
            ori_indexes.reshape(
                3,
                ori_indexes.shape[1],
                ori_indexes.shape[-1] // num_node_inputs,
                num_node_inputs,
            ),
            range(1, num_node_inputs),
            axis=-1,
        )
        ops_lst = np.split(
            ops.reshape(ops.shape[0], ops.shape[1] // num_node_inputs, num_node_inputs),
            range(1, num_node_inputs),
            axis=-1,
        )
        for adj_op_inds, inds, op in zip(adj_op_inds_lst, ori_indexes_lst, ops_lst):
            adj_op_inds[inds] = torch.tensor(op)

        if _ndim == 1:
            adj = adj[0]
            adj_op_inds_lst = [adj_op_inds[0] for adj_op_inds in adj_op_inds_lst]
            # adj_op_inds = adj_op_inds[0]
        return adj, adj_op_inds_lst

    def embed_and_transform_arch(self, archs):
        if isinstance(archs, (np.ndarray, list, tuple)):
            archs = np.array(archs)
            if archs.ndim == 3:
                # one arch
                archs = np.expand_dims(archs, 0)
            else:
                if not archs.ndim == 4:
                    import ipdb

                    ipdb.set_trace()
                assert archs.ndim == 4

        # get adjacent matrix
        # sparse
        # archs[:, :, 0, :]: (batch_size, num_cell_groups, num_node_inputs * num_steps)
        b_size, n_cg, _, n_edge = archs.shape
        adjs, adj_op_inds_lst = self.get_adj_dense(archs.reshape(b_size * n_cg, 2, n_edge))
        adjs = adjs.reshape([b_size, n_cg, adjs.shape[1], adjs.shape[2]]).to(self.init_node_emb.device)
        adj_op_inds_lst = [adj_op_inds.reshape([b_size, n_cg, adj_op_inds.shape[1], adj_op_inds.shape[2]]).to(self.init_node_emb.device) for adj_op_inds in adj_op_inds_lst]
        # (batch_size, num_cell_groups, num_nodes, num_nodes)

        # embedding of init nodes
        # TODO: output op should have a embedding maybe? (especially for hierarchical purpose)
        if self.other_node_independent:
            node_embs = self.init_node_emb.unsqueeze(0).repeat(b_size, 1, 1, 1)
        else:
            node_embs = torch.cat(
                (
                    self.init_node_emb.unsqueeze(0).repeat(b_size, 1, 1, 1),
                    self.other_node_emb.unsqueeze(0).repeat(b_size, 1, self._num_steps, 1),
                ),
                dim=2,
            )
        # (batch_size, num_cell_groups, num_nodes, self.node_dim)

        x = self.x_hidden(node_embs)
        # (batch_size, num_cell_groups, num_nodes, op_hid)
        return adjs, adj_op_inds_lst, x

    def forward(self, archs):
        # (batch_size, num_cell_groups, f, op)
        # adjs: (batch_size, num_cell_groups, num_nodes, num_nodes)
        # adj_op_inds: (batch_size, num_cell_groups, num_nodes, num_nodes)
        # x: (batch_size, num_cell_groups, num_nodes, op_hid)
        adjs, adj_op_inds_lst, x = self.embed_and_transform_arch(archs)
        y = x
        for i_layer, gcn in enumerate(self.gcns):
            y = gcn(
                y,
                adjs,
                adj_op_inds_lst,
                torch.stack(self.op_emb),
                self.none_index,
                self_op_emb=self.self_op_emb,
            )
            if self.use_bn:
                shape_y = y.shape
                y = self.bns[i_layer](y.reshape(shape_y[0], -1, shape_y[-1])).reshape(shape_y)
            if i_layer != self.num_gcn_layers - 1:
                y = F.relu(y)
            y = F.dropout(y, self.dropout, training=self.training)
        # y: (batch_size, num_cell_groups, num_nodes, gcn_out_dims[-1])
        y = y[:, :, 2:, :]  # do not keep the init node embedding
        if self.normalize:
            y = F.normalize(y, 2, dim=-1)
        if not self.final_concat:
            y = torch.mean(y, dim=2)  # average across nodes (bs, nc, god)
        else:
            # concat across all internal nodes (bs, nc, num_steps * god)
            y = torch.reshape(y, [y.shape[0], y.shape[1], -1])
        if self.normalize:
            y = F.normalize(y, 2, dim=-1)
        y = torch.reshape(y, [y.shape[0], -1])  # concat across cell groups, just reshape here
        return y


# ---- END: GCNFlowArchEmbedder ----
