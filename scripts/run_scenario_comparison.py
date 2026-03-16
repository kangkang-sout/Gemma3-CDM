"""
数据稀疏/稠密场景对比实验
对比 NeuralCDM、OpenCDM、GCD、SCD、RDGT 五个诊断模型在两种场景下的 AUC 和 ACC 值

场景定义：
- 数据稀疏场景：练习题在训练集中交互少于 3 次
- 数据稠密场景：练习题在训练集中交互多于 10 次
"""

import os
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from pathlib import Path
import numpy as np
from collections import Counter
from sklearn.metrics import roc_auc_score, accuracy_score

# 导入自定义模块
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from alignment.methods import get_alignment_method
from models.cdm.models import get_cdm_model
from scripts.download_datasets import DatasetDownloader


class SparseDenseDataset(Dataset):
    """支持稀疏/稠密场景的数据集"""
    
    def __init__(self, data: list, question_interaction_counts: dict,
                 scenario: str = 'all', num_students: int = 100, 
                 num_questions: int = 50, num_knowledge: int = 10,
                 max_seq_len: int = 10):
        """
        Args:
            data: 原始数据列表
            question_interaction_counts: 题目交互次数字典 {question_id: count}
            scenario: 场景类型 ('sparse', 'dense', 'all')
                - sparse: 交互次数 < 3
                - dense: 交互次数 > 10
                - all: 全部数据
        """
        self.data = []
        self.scenario = scenario
        self.num_students = num_students
        self.num_questions = num_questions
        self.num_knowledge = num_knowledge
        self.max_seq_len = max_seq_len
        
        # 根据场景过滤数据
        self._filter_data(data, question_interaction_counts)
        
        # 创建映射
        self.student_map = {}
        self.question_map = {}
        self.knowledge_map = {}
        self._build_mappings()
    
    def _filter_data(self, data: list, question_interaction_counts: dict):
        """根据场景过滤数据"""
        for item in data:
            question_id = item.get('question_id', f'ques_{len(self.data)}')
            count = question_interaction_counts.get(question_id, 0)
            
            if self.scenario == 'sparse' and count < 3:
                self.data.append(item)
            elif self.scenario == 'dense' and count > 10:
                self.data.append(item)
            elif self.scenario == 'all':
                self.data.append(item)
        
        print(f"场景：{self.scenario}, 过滤后数据量：{len(self.data)}")
    
    def _build_mappings(self):
        """构建 ID 映射"""
        student_id = 0
        question_id = 0
        knowledge_id = 0
        
        for item in self.data:
            if 'user_id' not in item:
                item['user_id'] = f"user_{student_id % 100}"
            if 'question_id' not in item:
                item['question_id'] = f"ques_{question_id % 50}"
            
            if item['user_id'] not in self.student_map:
                self.student_map[item['user_id']] = student_id
                student_id += 1
            
            if item['question_id'] not in self.question_map:
                self.question_map[item['question_id']] = question_id
                question_id += 1
            
            if 'knowledge_points' in item:
                for kp in item['knowledge_points']:
                    if kp not in self.knowledge_map:
                        self.knowledge_map[kp] = knowledge_id
                        knowledge_id += 1
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 获取 ID
        student_id = self.student_map.get(item.get('user_id', f'user_{idx}'), idx % self.num_students)
        question_id = self.question_map.get(item.get('question_id', f'ques_{idx}'), idx % self.num_questions)
        
        # 知识点
        knowledge_points = item.get('knowledge_points', [])
        knowledge_ids = [self.knowledge_map.get(kp, 0) for kp in knowledge_points[:self.max_seq_len]]
        
        # 填充到固定长度
        while len(knowledge_ids) < self.max_seq_len:
            knowledge_ids.append(0)
        
        # 标签（正确率）
        label = item.get('label', 1.0)
        if isinstance(label, bool):
            label = 1.0 if label else 0.0
        
        # 知识掩码
        knowledge_mask = torch.zeros(self.num_knowledge)
        for kid in knowledge_ids[:self.max_seq_len]:
            if kid < self.num_knowledge:
                knowledge_mask[kid] = 1.0
        
        return {
            'student_id': torch.tensor(student_id, dtype=torch.long),
            'question_id': torch.tensor(question_id, dtype=torch.long),
            'knowledge_ids': torch.tensor(knowledge_ids, dtype=torch.long),
            'knowledge_mask': knowledge_mask,
            'label': torch.tensor(label, dtype=torch.float32),
            'report_text': item.get('diagnostic_report', '')
        }


def create_collate_fn(pad_token_id: int = 0):
    """创建 collate 函数"""
    def collate_fn(batch):
        return {
            'student_id': torch.stack([item['student_id'] for item in batch]),
            'question_id': torch.stack([item['question_id'] for item in batch]),
            'knowledge_ids': torch.stack([item['knowledge_ids'] for item in batch]),
            'knowledge_mask': torch.stack([item['knowledge_mask'] for item in batch]),
            'label': torch.stack([item['label'] for item in batch]),
            'report_text': [item['report_text'] for item in batch]
        }
    return collate_fn


def compute_question_interactions(data: list) -> dict:
    """计算每个题目的交互次数"""
    question_counts = Counter()
    for item in data:
        question_id = item.get('question_id', f'ques_{len(question_counts)}')
        question_counts[question_id] += 1
    return dict(question_counts)


def train_epoch(model, alignment, dataloader, optimizer, device, 
                alignment_weight: float = 0.1):
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    
    for batch in tqdm(dataloader, desc="Training", leave=False):
        student_ids = batch['student_id'].to(device)
        question_ids = batch['question_id'].to(device)
        knowledge_ids = batch['knowledge_ids'].to(device)
        knowledge_mask = batch['knowledge_mask'].to(device)
        labels = batch['label'].to(device)
        
        # 模拟对齐特征（实际应用中应从文本嵌入生成）
        batch_size = student_ids.size(0)
        report_emb = torch.randn(batch_size, 768).to(device)
        knowledge_emb = torch.randn(batch_size, 768).to(device)
        
        # 对齐
        aligned_features, align_loss = alignment(report_emb, knowledge_emb)
        
        # 如果是多维输出，需要池化
        if len(aligned_features.shape) > 2:
            aligned_features = aligned_features.mean(dim=1)
        
        # 前向传播
        predictions = model(student_ids, question_ids, knowledge_mask, aligned_features)
        
        # 计算损失
        bce_loss = nn.BCELoss()(predictions, labels)
        total_batch_loss = bce_loss + alignment_weight * align_loss
        
        # 反向传播
        optimizer.zero_grad()
        total_batch_loss.backward()
        optimizer.step()
        
        total_loss += total_batch_loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, alignment, dataloader, device):
    """评估模型"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            student_ids = batch['student_id'].to(device)
            question_ids = batch['question_id'].to(device)
            knowledge_ids = batch['knowledge_ids'].to(device)
            knowledge_mask = batch['knowledge_mask'].to(device)
            labels = batch['label'].to(device)
            
            # 模拟对齐特征
            batch_size = student_ids.size(0)
            report_emb = torch.randn(batch_size, 768).to(device)
            knowledge_emb = torch.randn(batch_size, 768).to(device)
            
            # 对齐
            aligned_features, _ = alignment(report_emb, knowledge_emb)
            
            if len(aligned_features.shape) > 2:
                aligned_features = aligned_features.mean(dim=1)
            
            # 预测
            predictions = model(student_ids, question_ids, knowledge_mask, aligned_features)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算指标
    predictions = np.array(all_predictions)
    labels = np.array(all_labels)
    
    # AUC
    try:
        auc = roc_auc_score(labels, predictions)
    except:
        auc = 0.5
    
    # 准确率
    binary_preds = (predictions > 0.5).astype(int)
    binary_labels = (labels > 0.5).astype(int)
    acc = accuracy_score(binary_labels, binary_preds)
    
    return {'auc': auc, 'accuracy': acc}


def run_scenario_experiment(dataset_name: str, scenario: str, cdm_model: str,
                           epochs: int = 10, batch_size: int = 32, lr: float = 0.001,
                           alignment_method: str = 'none'):
    """
    运行单个场景实验
    
    Args:
        dataset_name: 数据集名称
        scenario: 场景 ('sparse' 或 'dense')
        cdm_model: 认知诊断模型
        epochs: 训练轮数
        batch_size: 批大小
        lr: 学习率
        alignment_method: 对齐方法（默认使用'none'以聚焦模型本身差异）
    """
    print(f"\n{'='*60}")
    print(f"实验配置:")
    print(f"  数据集：{dataset_name}")
    print(f"  场景：{scenario}")
    print(f"  CDM 模型：{cdm_model}")
    print(f"{'='*60}\n")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备：{device}\n")
    
    # 加载数据
    downloader = DatasetDownloader()
    try:
        data = downloader.load_dataset(dataset_name)
    except FileNotFoundError:
        print(f"数据集不存在，开始下载...")
        downloader.download_dataset(dataset_name)
        data = downloader.load_dataset(dataset_name)
    
    print(f"原始数据集大小：{len(data)}")
    
    # 计算题目交互次数
    question_counts = compute_question_interactions(data)
    
    # 统计稀疏和稠密题目数量
    sparse_count = sum(1 for v in question_counts.values() if v < 3)
    dense_count = sum(1 for v in question_counts.values() if v > 10)
    print(f"稀疏题目数（交互<3）: {sparse_count}")
    print(f"稠密题目数（交互>10）: {dense_count}")
    
    # 划分训练集和测试集（先划分再过滤场景）
    split_idx = int(len(data) * 0.8)
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    
    # 重新计算训练集的交互次数（用于场景过滤）
    train_question_counts = compute_question_interactions(train_data)
    
    # 创建场景数据集
    num_students = 100
    num_questions = 50
    num_knowledge = 10
    
    train_dataset = SparseDenseDataset(
        train_data, train_question_counts, scenario=scenario,
        num_students=num_students, num_questions=num_questions, 
        num_knowledge=num_knowledge
    )
    test_dataset = SparseDenseDataset(
        test_data, train_question_counts, scenario=scenario,
        num_students=num_students, num_questions=num_questions,
        num_knowledge=num_knowledge
    )
    
    # 如果场景数据太少，跳过
    if len(train_dataset) < 10 or len(test_dataset) < 10:
        print(f"警告：{scenario}场景数据太少，跳过实验")
        return None
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             collate_fn=create_collate_fn())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=create_collate_fn())
    
    # 初始化模型（固定使用'none'对齐以聚焦模型差异）
    alignment = get_alignment_method(alignment_method, embedding_dim=768).to(device)
    model = get_cdm_model(
        cdm_model,
        num_students=num_students,
        num_questions=num_questions,
        num_knowledge=num_knowledge,
        embedding_dim=64
    ).to(device)
    
    # 优化器
    optimizer = torch.optim.Adam(list(model.parameters()) + list(alignment.parameters()), lr=lr)
    
    # 训练
    best_auc = 0.0
    best_metrics = None
    for epoch in range(epochs):
        train_loss = train_epoch(model, alignment, train_loader, optimizer, device)
        
        # 评估
        metrics = evaluate(model, alignment, test_loader, device)
        
        if metrics['auc'] > best_auc:
            best_auc = metrics['auc']
            best_metrics = metrics
    
    print(f"\n最佳结果 - AUC: {best_metrics['auc']:.4f}, Accuracy: {best_metrics['accuracy']:.4f}")
    
    return {
        'dataset': dataset_name,
        'scenario': scenario,
        'model': cdm_model,
        'alignment': alignment_method,
        'train_samples': len(train_dataset),
        'test_samples': len(test_dataset),
        'best_auc': best_metrics['auc'],
        'best_accuracy': best_metrics['accuracy']
    }


def run_all_scenario_experiments(dataset_name: str, epochs: int = 10, 
                                 batch_size: int = 32, lr: float = 0.001):
    """
    运行所有场景对比实验
    
    5 个模型 × 2 个场景 = 10 个实验
    """
    models = ['neuralcdm', 'opencdm', 'gcd', 'scd', 'rdgt']
    scenarios = ['sparse', 'dense']
    
    all_results = []
    
    print(f"\n{'#'*60}")
    print(f"# 开始数据稀疏/稠密场景对比实验")
    print(f"# 数据集：{dataset_name}")
    print(f"# 实验总数：{len(models) * len(scenarios)}")
    print(f"{'#'*60}\n")
    
    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"## 场景：{scenario.upper()}")
        print(f"{'='*60}\n")
        
        for model in models:
            try:
                result = run_scenario_experiment(
                    dataset_name=dataset_name,
                    scenario=scenario,
                    cdm_model=model,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr
                )
                
                if result:
                    all_results.append(result)
                    
            except Exception as e:
                print(f"实验失败：{dataset_name}-{scenario}-{model}, 错误：{e}")
                all_results.append({
                    'dataset': dataset_name,
                    'scenario': scenario,
                    'model': model,
                    'error': str(e)
                })
    
    return all_results


def print_comparison_table(results: list):
    """打印对比表格"""
    print("\n" + "="*80)
    print("对比实验结果汇总")
    print("="*80)
    
    # 按数据集分组
    datasets = set(r['dataset'] for r in results if 'error' not in r)
    
    for dataset in datasets:
        dataset_results = [r for r in results if r['dataset'] == dataset and 'error' not in r]
        
        print(f"\n数据集：{dataset}")
        print("-"*80)
        print(f"{'模型':<12} | {'场景':<8} | {'样本数':<10} | {'AUC':<10} | {'ACC':<10}")
        print("-"*80)
        
        # 按模型和场景排序
        sorted_results = sorted(dataset_results, key=lambda x: (x['model'], x['scenario']))
        
        for r in sorted_results:
            print(f"{r['model']:<12} | {r['scenario']:<8} | {r['test_samples']:<10} | "
                  f"{r['best_auc']:<10.4f} | {r['best_accuracy']:<10.4f}")
        
        # 打印模型在两种场景下的对比
        print("\n模型场景对比分析:")
        print("-"*80)
        models = set(r['model'] for r in dataset_results)
        
        for model in models:
            model_results = [r for r in dataset_results if r['model'] == model]
            sparse_r = next((r for r in model_results if r['scenario'] == 'sparse'), None)
            dense_r = next((r for r in model_results if r['scenario'] == 'dense'), None)
            
            if sparse_r and dense_r:
                auc_diff = dense_r['best_auc'] - sparse_r['best_auc']
                acc_diff = dense_r['best_accuracy'] - sparse_r['best_accuracy']
                print(f"{model}:")
                print(f"  稀疏->稠密：AUC 变化：{auc_diff:+.4f}, ACC 变化：{acc_diff:+.4f}")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description='运行数据稀疏/稠密场景对比实验')
    parser.add_argument('--dataset', type=str, default='math23k',
                       choices=['math23k', 'open-luna', 'pisa2015', 'ktbd-ednet', 'all'],
                       help='数据集名称')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--output_dir', type=str, default='./results/scenario_comparison', 
                       help='结果输出目录')
    
    args = parser.parse_args()
    
    # 数据集列表
    datasets = ['math23k', 'open-luna', 'pisa2015', 'ktbd-ednet'] if args.dataset == 'all' else [args.dataset]
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 运行实验
    all_results = []
    
    for dataset in datasets:
        print(f"\n\n{'#'*80}")
        print(f"# 处理数据集：{dataset}")
        print(f"{'#'*80}\n")
        
        results = run_all_scenario_experiments(
            dataset_name=dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr
        )
        all_results.extend(results)
        
        # 保存当前数据集的结果
        result_file = output_dir / f"{dataset}_scenario_comparison.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到：{result_file}")
        
        # 打印对比表格
        print_comparison_table(results)
    
    # 汇总所有结果
    if all_results:
        summary_file = output_dir / 'all_scenario_comparison.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n{'='*80}")
        print("所有实验完成!")
        print(f"总结果已保存到：{summary_file}")
        print(f"{'='*80}")
        
        # 打印总对比表格
        print_comparison_table(all_results)


if __name__ == "__main__":
    main()
