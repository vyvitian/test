import os
import yaml
import torch
import pickle
import argparse
import numpy as np
import torch.nn as nn

def parse_configure(model=None, dataset=None):
    parser = argparse.ArgumentParser(description='RLMRec')
    parser.add_argument('--model', type=str, default='LightGCN', help='Model name')
    parser.add_argument('--dataset', type=str, default='amazon', help='Dataset name')
    parser.add_argument('--device', type=str, default='cuda', help='cpu or cuda')
    parser.add_argument('--seed', type=int, default=None, help='Device number')
    parser.add_argument('--cuda', type=str, default='0', help='Device number')
    parser.add_argument('--tqdm', type=bool, default=False, help='Use tqdm to show the training progress')
    parser.add_argument('--emb_size', type=int, default=None, help='Embedding size')
    parser.add_argument('--num_layers', type=int, default=None, help='Number of GNN layers')
    parser.add_argument('--cold_start_ratio', type=float, default=None, help='The ratio of cold-start items')
    args, _ = parser.parse_known_args()

    # cuda
    if args.device == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

    # model name
    if model is not None:
        model_name = model.lower()
    elif args.model is not None:
        model_name = args.model.lower()
    else:
        model_name = 'default'
        # print("Read the default (blank) configuration.")

    # dataset
    if dataset is not None:
        args.dataset = dataset



    # find yml file
    if not os.path.exists('./encoder/config/modelconf/{}.yml'.format(model_name)):
        raise Exception("Please create the yaml file for your model first.")

    # read yml file
    with open('./encoder/config/modelconf/{}.yml'.format(model_name), encoding='utf-8') as f:
        config_data = f.read()
        configs = yaml.safe_load(config_data)

         # training setting
        configs['tqdm'] = args.tqdm
        configs['model']['name'] = configs['model']['name'].lower()
        if 'tune' not in configs:
            configs['tune'] = {'enable': False}
        configs['device'] = args.device
        if args.dataset is not None:
            configs['data']['name'] = args.dataset
        if args.seed is not None:
            configs['train']['seed'] = args.seed
        if args.emb_size is not None:
            configs['model']['embedding_size'] = args.emb_size
        if args.cold_start_ratio is not None:
            configs['data']['cold_start_ratio'] = args.cold_start_ratio
        if args.num_layers is not None:
            configs['model'][configs['data']['name']]['layer_num'] = args.num_layers
        

        # semantic embeddings
        usrprf_embeds_path = "./data/{}/usr_emb_np.pkl".format(configs['data']['name'])
        itmprf_embeds_path = "./data/{}/itm_emb_np.pkl".format(configs['data']['name'])
        with open(usrprf_embeds_path, 'rb') as f:
            configs['usrprf_embeds'] = pickle.load(f)
        with open(itmprf_embeds_path, 'rb') as f:
            configs['itmprf_embeds'] = pickle.load(f)

        if configs['model']['name'] == 'alpharec':
            usrprf_embeds_path = "./data/{}/mean_user_prf_embeds.pkl".format(configs['data']['name'])
            configs['usrprf_embeds'] = pickle.load(open(usrprf_embeds_path, 'rb'))
            itmprf_embeds_path = "./data/{}/mean_item_prf_embeds.pkl".format(configs['data']['name'])
            configs['itmprf_embeds'] = pickle.load(open(itmprf_embeds_path, 'rb'))

        return configs

configs = parse_configure()
