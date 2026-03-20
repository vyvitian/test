import torch as t
from torch import nn
import torch.nn.functional as F
from config.configurator import configs
from models.loss_utils import cal_bpr_loss, reg_params, cal_infonce_loss
from models.base_model import BaseModel
from models.model_utils import SpAdjEdgeDrop
import pdb  # For debugging purposes

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform
kaiming_init = nn.init.kaiming_uniform_

class simgcl_rgcn(BaseModel):
    def __init__(self, data_handler):
        super(simgcl_rgcn, self).__init__(data_handler)
        self.adj = data_handler.torch_adj
        self.split_adjs()  # 分解邻接矩阵为多个特定关系的邻接矩阵列表
        self.keep_rate = configs['model']['keep_rate']
        print('keep rate', self.keep_rate)
        self.embeds = nn.Parameter(init(t.empty(self.user_num + self.item_num + self.attr_num, self.embedding_size)))
        self.edge_dropper = SpAdjEdgeDrop()
        self.final_embeds = None
        self.is_training = False

        # hyper-parameter
        self.layer_num = self.hyper_config['layer_num']
        self.reg_weight = self.hyper_config['reg_weight']
        self.eps = self.hyper_config.get('eps', 0.1)
        self.temperature = self.hyper_config.get('temperature', 0.2)
        self.cl_weight = configs['model'].get('cl_weight', 0.1)

        self.relation_weights = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2*self.embedding_size, self.embedding_size, bias=False),
                nn.LeakyReLU(self.embedding_size),
                nn.Linear(self.embedding_size, 1, bias=False)
            ) for _ in range(len(self.adjs))
        ])
        

    def split_adjs(self):
        """
        将原始邻接矩阵分解为多个特定关系的邻接矩阵列表 self.adjs
        所有分解出的矩阵都保持 [N, N] 的形状，其中 N = user+item+attr
        """
        self.adjs = []
        total_nodes = self.user_num + self.item_num + self.attr_num
        
        # 获取稀疏矩阵的索引和值
        adj_indices = self.adj.coalesce().indices()
        adj_values = self.adj.coalesce().values()
        rows, cols = adj_indices[0, :], adj_indices[1, :]
        
        # 定义图中不同类型节点的索引边界
        user_end_idx = self.user_num
        item_end_idx = self.user_num + self.item_num
        attr_end_idx = self.user_num + self.item_num + self.attr_num
        
        # 关系1: user <-> item
        # mask_ui = (rows < item_end_idx) & (cols < item_end_idx)
        # ui_indices = adj_indices[:, mask_ui]
        # ui_values = adj_values[mask_ui]
        # self.adjs.append(t.sparse_coo_tensor(ui_indices, ui_values, (total_nodes, total_nodes)))
        # 关系1： all
        mask_ui = (rows < attr_end_idx) & (cols < attr_end_idx)
        ui_indices = adj_indices[:, mask_ui]
        ui_values = adj_values[mask_ui]
        self.adjs.append(t.sparse_coo_tensor(ui_indices, ui_values, (total_nodes, total_nodes)))

        if self.attr_num > 0:
            # 关系2: item <-> attr
            mask_ia = (rows >= user_end_idx) & (cols >= user_end_idx)
            ia_indices = adj_indices[:, mask_ia]
            ia_values = adj_values[mask_ia]
            self.adjs.append(t.sparse_coo_tensor(ia_indices, ia_values, (total_nodes, total_nodes)))

            # 关系3: user <-> attr
            mask_ua = ((rows < user_end_idx) & (cols >= item_end_idx)) | ((rows >= item_end_idx) & (cols < user_end_idx))
            ua_indices = adj_indices[:, mask_ua]
            ua_values = adj_values[mask_ua]
            self.adjs.append(t.sparse_coo_tensor(ua_indices, ua_values, (total_nodes, total_nodes)))
        
        # 检查 adjs中是否至少存在一条边
        assert any(adj._nnz() > 0 for adj in self.adjs), "No edges found in the adjacency matrices."
        # self.leakly_relu = nn.LeakyReLU(negative_slope=0.2)
        self.num_relations = len(self.adjs)

    def _propagate(self, adj, embeds):
        return t.spmm(adj, embeds)
    
    def _perturb_embedding(self, embeds):
        noise = (F.normalize(t.rand(embeds.shape).cuda(), p=2) * t.sign(embeds)) * self.eps
        return embeds + noise

    def split_embeds(self, embeds):
        user_embeds = embeds[:self.user_num]
        item_embeds = embeds[self.user_num:self.user_num + self.item_num]
        attr_embeds = embeds[self.user_num + self.item_num:] if self.attr_num > 0 else None
        return user_embeds, item_embeds, attr_embeds


    def _propagate(self, adj, embeds):
        return t.spmm(adj, embeds)
    
    def split_embeds(self, embeds):
        user_embeds = embeds[:self.user_num]
        item_embeds = embeds[self.user_num:self.user_num + self.item_num]
        attr_embeds = embeds[self.user_num + self.item_num:] if self.attr_num > 0 else None
        return user_embeds, item_embeds, attr_embeds


    def forward(self, adjs=None, keep_rate=1.0, perturb=False):
        if adjs is None:
            adjs = self.adjs
        if not self.is_training and self.final_embeds is not None:
            return self.final_embeds[:self.user_num], self.final_embeds[self.user_num:self.user_num + self.item_num]
        embeds_list = [self.embeds]
        if self.is_training:
            adjs = [self.edge_dropper(adj, keep_rate) for adj in adjs]
        for i in range(self.layer_num):
            current_layer_embeds = embeds_list[-1]
            next_layer_embeds = 0
            for j in range(self.num_relations):
                rel_embeds = self._propagate(adjs[j], current_layer_embeds)
    
                concat_embeds = t.concat([current_layer_embeds, rel_embeds], axis=1)
                rel_weights = self.relation_weights[j](concat_embeds)
                rel_weights = t.sigmoid(rel_weights) # Sigmoid activation for weights
                next_layer_embeds += rel_embeds * rel_weights
            if perturb:
                next_layer_embeds = self._perturb_embedding(next_layer_embeds)
            embeds_list.append(next_layer_embeds)
        embeds = sum(embeds_list)
        self.final_embeds = embeds
        return embeds[:self.user_num], embeds[self.user_num:self.user_num + self.item_num]
    
    def cal_loss(self, batch_data):
        self.is_training = True
        user_embeds, item_embeds = self.forward(self.adjs, self.keep_rate, perturb=False)
        user_embeds_1, item_embeds_1 = self.forward(self.adjs, self.keep_rate, perturb=True)
        user_embeds_2, item_embeds_2 = self.forward(self.adjs, self.keep_rate, perturb=True)
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]
        bpr_loss = cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds) / anc_embeds.shape[0]
        cl_loss = cal_infonce_loss(user_embeds_1[ancs], user_embeds_2[ancs], user_embeds_2, self.temperature) + cal_infonce_loss(item_embeds_1[poss], item_embeds_2[poss], item_embeds_2, self.temperature)
        cl_loss /= anc_embeds.shape[0]
        cl_loss *= self.cl_weight
        reg_loss = self.reg_weight * reg_params(self)
        loss = bpr_loss + reg_loss + cl_loss
        losses = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss, 'cl_loss': cl_loss}
        return loss, losses

    def full_predict(self, batch_data):
        user_embeds, item_embeds = self.forward(self.adjs, 1.0)
        self.is_training = False
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        pck_user_embeds = user_embeds[pck_users]
        full_preds = pck_user_embeds @ item_embeds.T
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds