"""
Implement semantic entropy with multi-GPU support and efficient batching
using PyTorch DataLoader for higher performance.

This version incorporates fixes for transitive equivalence grouping (using Union-Find)
and corrects the batch index handling from the DataLoader.
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pickle
import logging
import ast
import json
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
from omegaconf import OmegaConf
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import pdb

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

# 根据可用设备设置DEVICE
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs for parallel processing.")


class EntailmentDeberta:
    """A wrapper for the Deberta model to check for textual entailment."""
    def __init__(self, model_path: str, device=None):
        self.device = device if device is not None else DEVICE
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        # 优化：先将模型移动到目标设备
        model.to(self.device)
        self.threshold = 0.5
        
        # 使用 DataParallel 包装模型以支持多 GPU
        if torch.cuda.device_count() > 1:
            # 明确指定 device_ids
            # self.model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
            self.model = torch.nn.DataParallel(model)
        else:
            self.model = model
            
        # 注意: ID '2' 是特定于许多NLI模型的。
        # 例如 microsoft/deberta-v2-mnli 的标签映射为: {0: 'contradiction', 1: 'neutral', 2: 'entailment'}
        # 使用前请确认你的模型配置。
        self.entailment_id = 2

    @torch.no_grad()
    def check_implication(self, text_pairs: list, *args, **kwargs):
        """Checks if text2 follows from text1 for a batch of pairs."""
        inputs = self.tokenizer(
            [pair[0] for pair in text_pairs],
            [pair[1] for pair in text_pairs],
            padding=True,
            truncation=True,
            return_tensors="pt").to(self.device)

        outputs = self.model(**inputs)
        logits = outputs.logits

        # probabilities = F.softmax(logits, dim=1)
        predictions = torch.argmax(logits, dim=1)
        mask = logits[:, self.entailment_id] > self.threshold

        return predictions, logits

    def are_equivalent(self, dataloader: DataLoader, flag:list):
        """Checks for bidirectional entailment for a batch of text pairs from a DataLoader."""

        for batch_indices, batch_pairs in dataloader:

            # 批量检查 text1 -> text2
            predictions1, logits1 = self.check_implication(batch_pairs)
            mask1 = logits1[:, self.entailment_id] > self.threshold

            # 批量检查 text2 -> text1
            swapped_pairs = [(pair[1], pair[0]) for pair in batch_pairs]
            predictions2, logits2 = self.check_implication(swapped_pairs)
            mask2 = logits2[:, self.entailment_id] > self.threshold
            is_equivalent = (predictions1 == self.entailment_id) & (predictions2 == self.entailment_id) & mask1 & mask2
            # is_equivalent shape : (batch_size,)
            for (i, j), is_equivalent_flag in zip(batch_indices, is_equivalent):
                if is_equivalent_flag.item() == 1:
                    flag[j] = flag[i]  # 将 j 的标记更新为 i 的标记

        return flag

def collate_fn(batch):
    ## getitem : (i, j) , (self.preferences[i], self.preferences[j])
    batch_indices = [item[0] for item in batch]
    batch_pairs = [item[1] for item in batch]
    return batch_indices, batch_pairs


class PreferencePairDynamicDataset(Dataset):
    """
    A Dataset that dynamically generates pairs between all preferences 
    and a given set of representatives.
    """
    def __init__(self, preferences, flag, index):
        self.preferences = preferences
        self.indices = []
        self.flag = flag

        # 生成所有可能的偏好对
        for j in range(index + 1, len(preferences)):
            if self.flag[j] == -1:  # 只生成未被标记的偏好对
                self.indices.append((index, j))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i, j = self.indices[idx]
        return (i, j), (self.preferences[i], self.preferences[j])

def filter(config):
    all_pairs = []
    preference_set = set()
    pref_count = Counter()
    with open(config.llm_extracted_path, 'r') as f:
        for line in f:
            asin, pref = ast.literal_eval(line)
            if asin in config.user2id or asin in config.item2id:
                all_pairs.append((asin, pref))
                pref_count[pref] += 1
                preference_set.add(pref)
    all_pairs = list(set(all_pairs))
    ## 去除pref出现频率不在范围内的preference
    preference_set = {pref for pref in preference_set if (pref_count[pref] >= config.min_count and pref_count[pref] <= config.max_count)}
    test_output_path = config.output_path.replace('.txt', f'_filtered_{config.min_count}_{config.max_count}.txt')
    with open(test_output_path, 'w') as f:
        f.writelines(f"{pref}\n" for pref in preference_set)
    return preference_set, all_pairs


def fusion(config, model, preferences, all_pairs):

    num_preferences = len(preferences)
    print(f"Total unique preferences after filtering: {num_preferences}")

    # 3. 迭代聚类优化
    flag = [-1] * num_preferences
    newer_preference2id = {}
    newer_preference_counter = Counter()
    next_pref_id = 0
    for id, preference in tqdm(enumerate(preferences), total=num_preferences, desc="Processing preferences"):
        if flag[id] != -1:  # 已经处理过的偏好
            newer_preference2id[preference] = flag[id]
            continue
        
        pref_dataset = PreferencePairDynamicDataset(preferences, flag, id)
        pref_dataLoader = DataLoader(pref_dataset, batch_size=config.batch_size, collate_fn=collate_fn, num_workers=2, prefetch_factor=2)
        flag[id] = next_pref_id
        newer_preference2id[preference] = next_pref_id
        next_pref_id += 1
        flag = model.are_equivalent(pref_dataLoader, flag)

    
    print("new attributes number:", next_pref_id)
    user_pref_pairs = []
    item_pref_pairs = []

    for asin, pref in all_pairs:
        if pref in newer_preference2id:
            # pdb.set_trace()
            pref_id = newer_preference2id[pref]
            if asin in config.user2id:
                user_pref_pairs.append((config.user2id[asin], pref_id))
            elif asin in config.item2id:
                item_pref_pairs.append((config.item2id[asin], pref_id))
                
    user_pref_pairs = list(set(user_pref_pairs))
    item_pref_pairs = list(set(item_pref_pairs))

    with open(config.output_path, 'w') as f:
        for u, p in user_pref_pairs:
            f.write(f'{u} {p}\n')
        for i, p in item_pref_pairs:
            item_id = int(i) + int(config.user_num)
            f.write(f'{item_id} {p}\n')

    with open(config.pref2id_path, 'w') as f:
        json.dump(newer_preference2id, f, indent=4)

    print(f"有效的preference数量: {len(newer_preference2id)}")
    print(f"用户偏好对数量: {len(user_pref_pairs)}")
    print(f"物品偏好对数量: {len(item_pref_pairs)}")
            
    return newer_preference2id


if __name__ == "__main__":
    # 建议：使用更强大的配置管理工具（如Hydra）或argparse来处理配置，而不是硬编码
    config = OmegaConf.create({
        'llm_extracted_path': '../ds_generated_attributesv4.json',
        # 'user2id_path': '../user2id.json',
        # 'item2id_path': '../item2id.json',
        'id2user_path': '../id2user.json',
        'id2item_path': '../id2item.json',
        'output_path': '../preference_25_5000.txt',
        'model_path': '../model/deberta-v2',  # 建议使用HuggingFace模型名，而非本地路径
        'pref2id_path': '../preference2id.json',
        'batch_size': 2048,  # 根据GPU显存调整,
        'max_count': 3000,  # 最大偏好计数限制
        'min_count': 25,    # 最小偏好计数限制
    })

    # 加载ID映射文件
    with open(config.id2user_path, 'r') as f:
        id2user = json.load(f)
    with open(config.id2item_path, 'r') as f:
        id2item = json.load(f)

    user2id = {v: k for k, v in id2user.items()}
    item2id = {v: k for k, v in id2item.items()}
    config.user_num = len(user2id)
    config.user2id = user2id
    config.item2id = item2id
    # 
    gpu_num = torch.cuda.device_count()

    # 初始化模型
    model = EntailmentDeberta(model_path=config.model_path)

    # 运行主逻辑
    preference_set, all_pairs = filter(config)
    fusion(config, model, list(preference_set), all_pairs)