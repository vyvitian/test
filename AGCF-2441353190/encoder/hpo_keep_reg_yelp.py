import argparse
import json
import math
import copy
from datetime import datetime

# Import will parse basic args (model, dataset, device, cuda) via configurator
from config.configurator import configs
from trainer.trainer import init_seed, Trainer
from trainer.logger import Logger
from data_utils.build_data_handler import build_data_handler
from models.bulid_model import build_model


def parse_float_list(s):
    return [float(x) for x in s.split(',') if x.strip()]

def main():
    parser = argparse.ArgumentParser(description='Grid Search keep_rate, reg_weight, temperature & eps on Yelp for simgcl_rgcn')
    parser.add_argument('--keep_rates', type=str, default='0.9,0.95,1.0', help='Comma separated keep_rate list')
    parser.add_argument('--reg_weights', type=str, default='1.0e-7,5.0e-6,1.0e-6, 1.0e-5', help='Comma separated reg_weight list')
    parser.add_argument('--temperatures', type=str, default='0.1,0.2,0.3,0.4', help='Comma separated temperature list for contrastive loss')
    parser.add_argument('--eps_list', type=str, default='0.1', help='Comma separated eps list for simgcl perturbation')
    parser.add_argument('--max_epoch', type=int, default=400, help='Override total epochs for search (smaller for speed)')
    parser.add_argument('--patience', type=int, default=5, help='Early stop patience')
    parser.add_argument('--seed_base', type=int, default=2023, help='Optional base seed; if set each run uses seed_base + run_idx')
    parser.add_argument('--log_configs_each_run', action='store_true', help='If set, log full configs for each run')
    parser.add_argument('--topk_index', type=int, default=-1, help='Which recall@k to optimize (-1 means last k in list)')
    args, _ = parser.parse_known_args()

    # Basic sanity
    # assert configs['data']['name'] == 'yelp', '请使用 --dataset yelp'
    # assert configs['model']['name'] == 'simgcl_rgcn', '请使用 --model simgcl_rgcn'

    # Enable tune mode for distinct checkpoint / log naming
    configs['tune']['enable'] = True

    # Override training epochs & patience for search
    configs['train']['epoch'] = args.max_epoch
    configs['train']['patience'] = args.patience

    keep_rates = parse_float_list(args.keep_rates)
    reg_weights = parse_float_list(args.reg_weights)
    temperatures = parse_float_list(args.temperatures)
    eps_values = parse_float_list(args.eps_list)

    optimize_k_index = args.topk_index
    if optimize_k_index < 0:
        optimize_k_index = len(configs['test']['k']) - 1  # 默认用最大的k (通常是@20)

    # Build data once (数据不依赖 keep_rate, reg_weight)
    init_seed()
    data_handler = build_data_handler()
    data_handler.load_data()

    results = []  # 存放每个组合的表现
    best_combo = None
    best_recall = -1e9

    total_runs = len(keep_rates) * len(reg_weights) * len(temperatures) * len(eps_values)
    run_idx = 0

    for kr in keep_rates:
        for rw in reg_weights:
            for temp in temperatures:
                for eps_val in eps_values:
                    run_idx += 1
                    run_tag = f'keep{kr}_reg{rw}_temp{temp}_eps{eps_val}'
                    print(f'==== 开始第 {run_idx}/{total_runs} 次: {run_tag} ====')

                    # 设置当前超参 (顶层 & 数据集特定 yelp 块, 防止被覆盖)
                    configs['model']['keep_rate'] = kr
                    configs['model']['reg_weight'] = rw
                    # 也记录温度与 eps
                    configs['model']['temperature'] = temp
                    configs['model']['eps'] = eps_val
                    data_name = configs['data']['name']
                    configs['model'][data_name]['keep_rate'] = kr
                    configs['model'][data_name]['reg_weight'] = rw
                    configs['model'][data_name]['temperature'] = temp
                    configs['model'][data_name]['eps'] = eps_val
                    # 用于保存文件名
                    configs['tune']['now_para_str'] = run_tag

                    # 可选：不同 seed
                    # if args.seed_base is not None:
                    #     configs['train']['seed'] = args.seed_base + run_idx

                    # # 重新设随机种子
                    # init_seed()

                    # 构建模型 & 训练器
                    model = build_model(data_handler).to(configs['device'])
                    # 日志：只在第一轮输出完整配置 (除非用户强制)
                    logger = Logger(log_configs=(run_idx == 1 or args.log_configs_each_run))
                    trainer = Trainer(data_handler, logger)
                    trainer.create_optimizer(model)

                    # 自定义训练主循环（复制 Trainer.train 逻辑以捕获 best_recall）
                    now_patience = 0
                    best_epoch = 0
                    local_best_recall = -1e9
                    best_state = None
                    train_conf = configs['train']
                    test_step = train_conf['test_step']

                    for epoch in range(train_conf['epoch']):
                        trainer.train_epoch(model, epoch)
                        if epoch % test_step == 0:
                            eval_res = trainer.evaluate(model, epoch)
                            cur_recall = eval_res['recall'][optimize_k_index]
                            if cur_recall > local_best_recall:
                                local_best_recall = cur_recall
                                best_epoch = epoch
                                best_state = copy.deepcopy(model.state_dict())
                                now_patience = 0
                            else:
                                now_patience += 1
                            if now_patience >= train_conf['patience']:
                                break

                    # 测试：载入最佳参数
                    best_model = build_model(data_handler).to(configs['device'])
                    best_model.load_state_dict(best_state)
                    test_res = trainer.test(best_model)
                    test_recall_target = test_res['recall'][optimize_k_index]

                    combo_record = {
                        'keep_rate': kr,
                        'reg_weight': rw,
                        'temperature': temp,
                        'eps': eps_val,
                        'best_epoch': best_epoch,
                        'val_recall_target': float(local_best_recall),
                        'test_recall_target': float(test_recall_target),
                        'val_recall_list': [float(x) for x in eval_res['recall']],
                        'test_recall_list': [float(x) for x in test_res['recall']],
                        'k': configs['test']['k'],
                    }
                    results.append(combo_record)

                    # 更新全局最优 (使用验证 recall@k)
                    if local_best_recall > best_recall:
                        best_recall = local_best_recall
                        best_combo = combo_record

                    print(f'完成 {run_tag}: 验证集目标 recall={local_best_recall:.6f}, 测试集目标 recall={test_recall_target:.6f}')

    # 保存结果
    out_path = f'./encoder/log/hpo_{configs["model"]["name"]}_{configs["data"]["name"]}.json'
    summary = {
        'datetime': datetime.now().isoformat(),
        'optimize_metric': f"recall@{configs['test']['k'][optimize_k_index]}",
        'best': best_combo,
        'all_results': results,
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print('\n==== 搜索完成 ====')
    metric_k = configs['test']['k'][optimize_k_index]
    print(f"最优组合: keep_rate={best_combo['keep_rate']}, reg_weight={best_combo['reg_weight']}, 验证 recall@{metric_k}={best_combo['val_recall_target']:.6f}, 测试 recall@{metric_k}={best_combo['test_recall_target']:.6f}")
    print(f'全部结果已保存到: {out_path}')


if __name__ == '__main__':
    main()
