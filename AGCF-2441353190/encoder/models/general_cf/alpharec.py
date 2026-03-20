import pickle
import torch as t
import torch.nn as nn
from config.configurator import configs
from models.loss_utils import cal_bpr_loss, reg_params, cal_infonce_loss
from models.base_model import BaseModel
from models.model_utils import SpAdjEdgeDrop
import torch.nn.functional as F

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform  

class AlphaRec(BaseModel):
    def __init__(self, data_handler):
        super(AlphaRec, self).__init__(data_handler)
        self.adj = data_handler.torch_adj
        self.keep_rate = configs['model']['keep_rate']
        # self.embeds = nn.Parameter(init(t.empty(self.user_num + self.item_num, self.embedding_size)))
        self.edge_dropper = SpAdjEdgeDrop()
        self.final_embeds = None
        self.is_training = False

        self.tau = configs['model'].get('tau', 0.2)
        self.embedding_size = configs['model']['embedding_size']
        self.usrprf_embeds = (configs['usrprf_embeds']).float().cuda()
        self.itmprf_embeds = t.tensor(configs['itmprf_embeds']).float().cuda()
        self.init_embed_shape = self.usrprf_embeds.shape[1]
        self.layer_num = configs['model']['layer_num']
        self.reg_weight = configs['model']['reg_weight']

        
        multiplier_dict = {
            'bert': 8,
            'roberta': 8,
            'v2': 2,
            'v3': 1/2,
            'v3_shuffle': 1/2,
        } 
        if(configs['model']['lm_model'] in multiplier_dict):
            multiplier = multiplier_dict[configs['model']['lm_model']]
        else:
            multiplier = 24/32

        self.mlp = nn.Sequential(
            nn.Linear(self.init_embed_shape, int(multiplier * self.init_embed_shape)),
            nn.LeakyReLU(),
            nn.Linear(int(multiplier * self.init_embed_shape), self.embedding_size)
        )
        self._init_weight()

    def _init_weight(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                init(m.weight)

    def _propagate(self, adj, embeds):
        return t.spmm(adj, embeds)
    
    def compute(self):
        user_prf_embeds = self.mlp(self.usrprf_embeds)
        item_prf_embeds = self.mlp(self.itmprf_embeds)
        all_embeds = t.concat([user_prf_embeds, item_prf_embeds], axis=0)
        embs = [all_embeds]
        if self.is_training:
            adj = self.edge_dropper(self.adj, self.keep_rate)
        else:
            adj = self.adj
        for _ in range(self.layer_num):
            all_embeds = self._propagate(adj, embs[-1])
            embs.append(all_embeds)
        embs = sum(embs)
        users, items = t.split(embs, [self.user_num, self.item_num], dim=0)
        return users, items
    
    def forward(self, adj=None, keep_rate=1.0):
        if adj is None:
            adj = self.adj
        if not self.is_training and self.final_embeds is not None:
            return self.final_embeds[:self.user_num], self.final_embeds[self.user_num:self.user_num + self.item_num]
        return self.compute()
    
    def _pick_embeds(self, user_embeds, item_embeds, batch_data):
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]
        return anc_embeds, pos_embeds, neg_embeds
    
    def cal_loss(self, batch_data):
        self.is_training = True
        user_embeds, item_embeds = self.forward(self.adj, self.keep_rate)
        anc_embeds, pos_embeds, neg_embeds = self._pick_embeds(user_embeds, item_embeds, batch_data)
        if configs['model'].get('train_norm', False):
            anc_embeds = F.normalize(anc_embeds, dim = -1)
            pos_embeds = F.normalize(pos_embeds, dim = -1)
            neg_embeds = F.normalize(neg_embeds, dim = -1)
        bpr_loss = cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds) / anc_embeds.shape[0]

        pos_ratings = t.sum(anc_embeds * pos_embeds, dim=-1)
        neg_ratings = t.sum(anc_embeds.unsqueeze(1) * neg_embeds, dim=-1)  # b x n
        numerator = t.exp(pos_ratings / self.tau)
        denominator = numerator + t.sum(t.exp(neg_ratings / self.tau), dim=1)
        infonce_loss = -t.mean(t.log(numerator / denominator))

        reg_loss = self.reg_weight * reg_params(self)
        loss = infonce_loss + reg_loss
        losses = {'infonce_loss': infonce_loss, 'reg_loss': reg_loss, 'bpr_loss': bpr_loss}
        return loss, losses
    
    @t.no_grad()
    def full_predict(self, batch_data):
        user_embeds, item_embeds = self.forward(self.adj, 1.0)
        self.is_training = False
        pck_users, train_mask = batch_data
        full_preds = t.matmul(user_embeds[pck_users], t.transpose(item_embeds, 0, 1))
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds
        

