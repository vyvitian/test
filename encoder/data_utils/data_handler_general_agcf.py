import pickle
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix, bmat
import scipy.sparse as sp
from config.configurator import configs
from data_utils.datasets_general_cf import PairwiseTrnData, PairwiseWEpochFlagTrnData, AllRankTstData
import torch as t
import torch.utils.data as data
import json  # added for optional saving
import os

class DataHandlerGeneralAGCF:
    def __init__(self):
        if configs['data']['name'] == 'amazon':
            predir = './data/amazon/'
        elif configs['data']['name'] == 'yelp':
            predir = './data/yelp/'
        elif configs['data']['name'] == 'steam':
            predir = './data/steam/'
        else:
            # raise NotImplementedError
            predir = './data/' + configs['data']['name'] + '/'
        self.trn_file = predir + 'trn_mat.pkl'
        self.val_file = predir + 'val_mat.pkl'
        self.tst_file = predir + 'tst_mat.pkl'
        self.attr_file = predir + 'attr_edges.pkl'
        # placeholders for later statistics
        self.user_attribute_matrix = None
        self.item_attribute_matrix = None
        self.uai_path_stats = None


    def _load_one_mat(self, file):
        """Load one single adjacent matrix from file

        Args:
            file (string): path of the file to load

        Returns:
            scipy.sparse.coo_matrix: the loaded adjacent matrix
        """
        with open(file, 'rb') as fs:
            mat = (pickle.load(fs) != 0).astype(np.float32)
        if type(mat) != coo_matrix:
            mat = coo_matrix(mat)
        return mat
    
    def _normalize_adj(self, mat):
        """Laplacian normalization for mat in coo_matrix

        Args:
            mat (scipy.sparse.coo_matrix): the un-normalized adjacent matrix

        Returns:
            scipy.sparse.coo_matrix: normalized adjacent matrix
        """
        degree = np.array(mat.sum(axis=-1))
        d_inv_sqrt = np.reshape(np.power(degree, -0.5), [-1])
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_inv_sqrt_mat = sp.diags(d_inv_sqrt)
        return mat.dot(d_inv_sqrt_mat).transpose().dot(d_inv_sqrt_mat).tocoo()
    
    def _make_torch_adj(self, mat, self_loop=False):
        """Transform uni-directional adjacent matrix in coo_matrix into bi-directional adjacent matrix in torch.sparse.FloatTensor

        Args:
            mat (coo_matrix): the uni-directional adjacent matrix

        Returns:
            torch.sparse.FloatTensor: the bi-directional matrix
        """
        if not self_loop:
            a = csr_matrix((configs['data']['user_num'], configs['data']['user_num']))
            b = csr_matrix((configs['data']['item_num'], configs['data']['item_num']))
        else:
            data = np.ones(configs['data']['user_num'])
            row_indices = np.arange(configs['data']['user_num'])
            column_indices = np.arange(configs['data']['user_num'])
            a = csr_matrix((data, (row_indices, column_indices)), shape=(configs['data']['user_num'], configs['data']['user_num']))

            data = np.ones(configs['data']['item_num'])
            row_indices = np.arange(configs['data']['item_num'])
            column_indices = np.arange(configs['data']['item_num'])
            b = csr_matrix((data, (row_indices, column_indices)), shape=(configs['data']['item_num'], configs['data']['item_num']))

        mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
        mat = (mat != 0) * 1.0
        mat = self._normalize_adj(mat)

        # make torch tensor
        idxs = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = t.from_numpy(mat.data.astype(np.float32))
        shape = t.Size(mat.shape)
        return t.sparse.FloatTensor(idxs, vals, shape).to(configs['device'])
    
    def _make_uai_adj(self, trn_mat, attr_edges, self_loop=False):
        user_num, item_num = trn_mat.shape
        user_item_num, attr_num = attr_edges.shape
        trn_mat_csr = sp.csr_matrix(trn_mat) # num_user, num_item = trn_mat_csr.shape
        attr_csr = sp.csr_matrix(attr_edges)
        user_attribute_matrix = attr_csr[:user_num, :]
        item_attribute_matrix = attr_csr[user_num:, :]
        # 保存以便统计路径
        self.user_attribute_matrix = user_attribute_matrix
        self.item_attribute_matrix = item_attribute_matrix
        if not self_loop:
            a = csr_matrix((user_num, user_num))
            b = csr_matrix((item_num, item_num))
            c = csr_matrix((attr_num, attr_num))
        else:
            a = sp.eye(user_num, format='csr')
            b = sp.eye(item_num, format='csr')
            c = sp.eye(attr_num, format='csr')
        # pdb.set_trace()
        adj_mat = bmat([[a, trn_mat_csr, user_attribute_matrix],
                        [trn_mat_csr.T, b, item_attribute_matrix],# num_user+num_item+num_attr
                        [user_attribute_matrix.T, item_attribute_matrix.T, c]], format='coo')
        adj_mat = (adj_mat !=0 ) *1.0
        if configs.get('data', {}).get('pre_norm', True):
            adj_mat = self._normalize_adj(adj_mat)
        idxs = t.from_numpy(np.vstack([adj_mat.row, adj_mat.col]).astype(np.int64))
        vals = t.from_numpy(adj_mat.data.astype(np.float32))
        shape = t.Size(adj_mat.shape)
        return t.sparse.FloatTensor(idxs, vals, shape).to(configs['device'])
    
    def _make_ui_adj(self, trn_mat, attr_edges, self_loop=False):
        user_num, item_num = trn_mat.shape
        _, attr_num = attr_edges.shape
        trn_mat_csr = sp.csr_matrix(trn_mat) # num_user, num_item = trn_mat_csr.shape
        if not self_loop:
            a = csr_matrix((user_num, user_num))
            b = csr_matrix((item_num, item_num))
            c = csr_matrix((attr_num, attr_num))
            d = csr_matrix((user_num, attr_num))
            e = csr_matrix((item_num, attr_num))

        else:
            a = sp.eye(user_num, format='csr')
            b = sp.eye(item_num, format='csr')
            c = sp.eye(attr_num, format='csr')
            d = sp.eye(user_num, attr_num, format='csr')
            e = sp.eye(item_num, attr_num, format='csr')
        adj_mat = bmat([[a, trn_mat_csr, d],
                        [trn_mat_csr.T, b, e],# num_user+num_item+num_attr
                        [d.T, e.T, c]], format='coo')
        adj_mat = (adj_mat !=0 ) *1.0
        if configs.get('data', {}).get('pre_norm', True):
            adj_mat = self._normalize_adj(adj_mat)
        idxs = t.from_numpy(np.vstack([adj_mat.row, adj_mat.col]).astype(np.int64))
        vals = t.from_numpy(adj_mat.data.astype(np.float32))
        shape = t.Size(adj_mat.shape)
        return t.sparse.FloatTensor(idxs, vals, shape).to(configs['device'])
    
    def load_data(self):
        trn_mat = self._load_one_mat(self.trn_file)
        val_mat = self._load_one_mat(self.val_file)
        tst_mat = self._load_one_mat(self.tst_file)
        self.trn_mat = trn_mat
        self.tst_mat = tst_mat  # 保存测试集矩阵以便统计
        configs['data']['user_num'], configs['data']['item_num'] = trn_mat.shape 
        if os.path.exists(self.attr_file):
            atr_mat = self._load_one_mat(self.attr_file)
            configs['data']['user_item_num'], configs['data']['attr_num'] = atr_mat.shape
            assert configs['data']['user_item_num'] == (configs['data']['user_num'] + configs['data']['item_num'])
        
        if configs['data']['mat_type'] == 'uai':
            self.torch_adj = self._make_uai_adj(trn_mat, atr_mat, self_loop=True)
            self.ui_adj = self._make_ui_adj(trn_mat, atr_mat, self_loop=True)
        else:
            self.torch_adj = self._make_torch_adj(trn_mat, self_loop=True)
        
        if configs['model']['name'] == 'gccf':
            self.torch_adj = self._make_torch_adj(trn_mat, self_loop=True)

        if configs['train']['loss'] == 'pairwise':
            trn_data = PairwiseTrnData(trn_mat)
        elif configs['train']['loss'] == 'pairwise_with_epoch_flag':
            trn_data = PairwiseWEpochFlagTrnData(trn_mat)

        val_data = AllRankTstData(val_mat, trn_mat)
        tst_data = AllRankTstData(tst_mat, trn_mat)
        self.test_dataloader = data.DataLoader(tst_data, batch_size=configs['test']['batch_size'], shuffle=False, num_workers=0)
        self.valid_dataloader = data.DataLoader(val_data, batch_size=configs['test']['batch_size'], shuffle=False, num_workers=0)
        self.train_dataloader = data.DataLoader(trn_data, batch_size=configs['train']['batch_size'], shuffle=True, num_workers=0)

        # 统计 U-A-I 路径及与测试集的重合度（只有在有属性文件且构建了 user/item attribute 矩阵时）
        if self.user_attribute_matrix is not None and self.item_attribute_matrix is not None:
            self.uai_path_stats = self.compute_uai_path_stats(save_json=True)
            print('[UAI Path Stats]', json.dumps({k: v for k, v in self.uai_path_stats.items() if k != 'sample_counts'}, ensure_ascii=False, indent=2))

    # ================= 新增: 统计 U-A-I 路径与测试集重合度 ================= #
    def compute_uai_path_stats(self, sample_size=20, save_json=False):
        """统计 user-attribute-item (U-A-I) 路径数量与其与测试集交互(tst_mat)的重合情况。

        说明:
            若 user_attribute_matrix[u,a]=1 且 item_attribute_matrix[i,a]=1, 则存在一条 u -> a -> i 的长度2路径。
            对测试集中每个 (u,i) 交互统计其路径条数，并计算：
                1) 有至少一条路径的测试交互占比
                2) 平均每个测试交互的路径数
                3) 在存在路径的测试交互上的平均路径数
                4) 最大路径数
        参数:
            sample_size(int): 随机抽样若干测试交互返回其路径数量以便人工检查
            save_json(bool): 是否把结果存为 json 文件（与数据集同目录下）
        返回:
            dict 统计结果
        """
        if self.tst_mat is None or self.user_attribute_matrix is None or self.item_attribute_matrix is None:
            return {}
        user_attr = self.user_attribute_matrix.tocsr()
        item_attr = self.item_attribute_matrix.tocsr()
        tst = self.tst_mat.tocoo()
        total = tst.nnz
        test_with_path = 0
        total_path_counts = 0
        total_path_counts_positive = 0
        max_paths = 0
        path_counts_list = []  # 仅用于采样/可选分析

        # 预取每个用户/物品的属性索引，减少重复访问
        user_attr_indices_cache = {}
        item_attr_indices_cache = {}

        def intersection_size(a_idx, b_idx):
            # 两个已排序数组求交集大小（scipy 索引升序）
            cnt = 0
            i = j = 0
            la, lb = len(a_idx), len(b_idx)
            while i < la and j < lb:
                if a_idx[i] == b_idx[j]:
                    cnt += 1; i += 1; j += 1
                elif a_idx[i] < b_idx[j]:
                    i += 1
                else:
                    j += 1
            return cnt

        for u, i in zip(tst.row, tst.col):
            if u not in user_attr_indices_cache:
                user_attr_indices_cache[u] = user_attr[u].indices
            if i not in item_attr_indices_cache:
                item_attr_indices_cache[i] = item_attr[i].indices
            ua = user_attr_indices_cache[u]
            ia = item_attr_indices_cache[i]
            pc = intersection_size(ua, ia)
            path_counts_list.append(pc)
            total_path_counts += pc
            if pc > 0:
                test_with_path += 1
                total_path_counts_positive += pc
                if pc > max_paths:
                    max_paths = pc

        ratio = test_with_path / total if total > 0 else 0.0
        avg_all = total_path_counts / total if total > 0 else 0.0
        avg_pos = total_path_counts_positive / test_with_path if test_with_path > 0 else 0.0

        # 采样展示前 sample_size 个（或全部不足 sample_size）
        sample_counts = path_counts_list[:sample_size]

        stats = {
            'total_test_interactions': int(total),
            'test_with_uai_path': int(test_with_path),
            'ratio_test_with_uai_path': ratio,
            'avg_uai_paths_per_test_interaction': avg_all,
            'avg_uai_paths_per_positive_test_interaction': avg_pos,
            'max_uai_paths_for_test_interaction': int(max_paths),
            'sample_counts': sample_counts,
        }

        if save_json:
            out_dir = './encoder/log/'
            os.makedirs(out_dir, exist_ok=True)
            out_file = os.path.join(out_dir, f'uai_path_stats_{configs["data"]["name"]}.json')
            with open(out_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            stats['saved_to'] = out_file
        return stats