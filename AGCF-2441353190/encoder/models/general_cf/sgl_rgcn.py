
import torch as t
from torch import nn
import torch.nn.functional as F
from config.configurator import configs
from models.loss_utils import cal_bpr_loss, reg_params, cal_infonce_loss
from models.base_model import BaseModel
from models.model_utils import SpAdjEdgeDrop, NodeDrop
import pdb  # For debugging purposes

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform
kaiming_init = nn.init.kaiming_uniform_

class sgl_rgcn(BaseModel):
    def __init__(self, data_handler):
        super(sgl_rgcn, self).__init__(data_handler)
        self.adj = data_handler.torch_adj
        self.split_adjs()  # 分解邻接矩阵为多个特定关系的邻接矩阵列表
        self.keep_rate = configs['model']['keep_rate']
        self.augmentation = configs['model']['augmentation']
        self.embeds = nn.Parameter(init(t.empty(self.user_num + self.item_num + self.attr_num, self.embedding_size)))
        self.node_dropper = NodeDrop()
        self.edge_dropper = SpAdjEdgeDrop()
        self.final_embeds = None
        self.is_training = False

        # hyper-parameter
        self.layer_num = self.hyper_config['layer_num']
        self.reg_weight = self.hyper_config['reg_weight']
        self.cl_weight = self.hyper_config['cl_weight']
        self.temperature = self.hyper_config['temperature']
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


    def forward(self, adjs=None, keep_rate=1.0):
        if adjs is None:
            adjs = self.adjs
        if not self.is_training and self.final_embeds is not None:
            return self.final_embeds[:self.user_num], self.final_embeds[self.user_num:self.user_num + self.item_num]
        embeds = self.embeds
        if self.augmentation == 'node_drop':
            embeds = self.node_dropper(embeds, keep_rate)
        embeds_list = [self.embeds]
        if self.augmentation == 'edge_drop':
            # adj = self.edge_dropper(adj, keep_rate)
            adjs = [self.edge_dropper(adj, keep_rate) for adj in adjs]
        for i in range(self.layer_num):
            random_walk = self.augmentation == 'random_walk'
            tem_adjs = adjs if not random_walk else [self.edge_dropper(adj, keep_rate) for adj in adjs]
            current_layer_embeds = embeds_list[-1]
            next_layer_embeds = 0
            for j in range(self.num_relations):
                rel_embeds = self._propagate(adjs[j], current_layer_embeds)
                concat_embeds = t.concat([current_layer_embeds, rel_embeds], axis=1)
                rel_weights = self.relation_weights[j](concat_embeds)
                rel_weights = F.sigmoid(rel_weights) # Sigmoid activation for weights
                
                next_layer_embeds += rel_embeds * rel_weights
     
            embeds_list.append(next_layer_embeds)
        embeds = sum(embeds_list)
        self.final_embeds = embeds
        return embeds[:self.user_num], embeds[self.user_num:self.user_num + self.item_num]
    
    
    def _pick_embeds(self, user_embeds, item_embeds, batch_data):
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]
        return anc_embeds, pos_embeds, neg_embeds
    
    def cal_loss(self, batch_data):
        self.is_training = True
        user_embeds1, item_embeds1 = self.forward(self.adjs, self.keep_rate)
        user_embeds2, item_embeds2 = self.forward(self.adjs, self.keep_rate)
        user_embeds3, item_embeds3 = self.forward(self.adjs, 1.0)
        anc_embeds1, pos_embeds1, neg_embeds1 = self._pick_embeds(user_embeds1, item_embeds1, batch_data)
        anc_embeds2, pos_embeds2, neg_embeds2 = self._pick_embeds(user_embeds2, item_embeds2, batch_data)
        anc_embeds3, pos_embeds3, neg_embeds3 = self._pick_embeds(user_embeds3, item_embeds3, batch_data)

        bpr_loss = cal_bpr_loss(anc_embeds3, pos_embeds3, neg_embeds3) / anc_embeds3.shape[0]
        cl_loss = cal_infonce_loss(anc_embeds1, anc_embeds2, user_embeds2, self.temperature) + cal_infonce_loss(pos_embeds1, pos_embeds2, item_embeds2, self.temperature) + cal_infonce_loss(neg_embeds1, neg_embeds2, item_embeds2, self.temperature)
        cl_loss /= anc_embeds1.shape[0]
        reg_loss = self.reg_weight * reg_params(self)
        cl_loss *= self.cl_weight
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