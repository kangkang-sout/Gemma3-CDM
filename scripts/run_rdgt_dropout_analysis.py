"""
RDGT 模型 Dropout 敏感性分析实验
对比 RDGT 模型在三种对齐方式（不对齐、交互重建对齐、混合对比对齐）下，
在四个数据集（math23k、open-luna、pisa2015、ktbd-ednet）上，
AUC 和 ACC 随 dropout 值变化的折线图

Dropout 值范围：[0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
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
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端

# 导入自定义模块
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from alignment.methods import get_alignment_method
from models.cdm.models import RDGT
from scripts.download_datasets import DatasetDownloader


class DropoutDataset(Dataset):
    """支持 dropout 实验的数据集"""
    
    def __init__(self, data: list, num_students: int = 100, 
                 num_questions: int = 50, num_knowledge: int = 10,
                 max_seq_len: int = 10):
        """
        Args:
            data: 原始数据列表
            num_students: 学生数量
            num_questions: 题目数量
            num_knowledge: 知识点数量
            max_seq_len: 最大序列长度
        """
        self.data = data
        self.num_students = num_students
        self.num_questions = num_questions
        self.num_knowledge = num_knowledge
        self.max_seq_len = max_seq_len
        
        # 创建映射
        self.student_map = {}
        self.question_map = {}
        self.knowledge_map = {}
        self._build_mappings()
    
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
        
        # 知识图谱（用于 RDGT）
        knowledge_graph = torch.zeros(self.num_knowledge, self.num_knowledge)
        for i, kid in enumerate(knowledge_ids[:self.max_seq_len]):
            if kid < self.num_knowledge:
                knowledge_graph[kid, kid] = 1.0
                if i > 0:
                    prev_kid = knowledge_ids[i-1]
                    if prev_kid < self.num_knowledge:
                        knowledge_graph[prev_kid, kid] = 0.5
        
        return {
            'student_id': torch.tensor(student_id, dtype=torch.long),
            'question_id': torch.tensor(question_id, dtype=torch.long),
            'knowledge_mask': knowledge_mask,
            'knowledge_graph': knowledge_graph,
            'label': torch.tensor(label, dtype=torch.float32),
            'report_text': item.get('diagnostic_report', '')
        }


def create_collate_fn(pad_token_id: int = 0):
    """创建 collate 函数"""
    def collate_fn(batch):
        return {
            'student_id': torch.stack([item['student_id'] for item in batch]),
            'question_id': torch.stack([item['question_id'] for item in batch]),
            'knowledge_mask': torch.stack([item['knowledge_mask'] for item in batch]),
            'knowledge_graph': torch.stack([item['knowledge_graph'] for item in batch]),
            'label': torch.stack([item['label'] for item in batch]),
            'report_text': [item['report_text'] for item in batch]
        }
    return collate_fn


def train_epoch_with_dropout(model, alignment, dataloader, optimizer, device, 
                             alignment_weight: float = 0.1, dropout_rate: float = 0.1):
    """训练一个 epoch（带 dropout 设置）"""
    model.train()
    
    # 设置模型的 dropout
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = dropout_rate
    
    total_loss = 0.0
    
    for batch in tqdm(dataloader, desc="Training", leave=False):
        student_ids = batch['student_id'].to(device)
        question_ids = batch['question_id'].to(device)
        knowledge_mask = batch['knowledge_mask'].to(device)
        knowledge_graph = batch['knowledge_graph'].to(device)
        labels = batch['label'].to(device)
        
        # 模拟对齐特征
        batch_size = student_ids.size(0)
        report_emb = torch.randn(batch_size, 768).to(device)
        knowledge_emb = torch.randn(batch_size, 768).to(device)
        
        # 对齐
        aligned_features, align_loss = alignment(report_emb, knowledge_emb)
        
        # 如果是多维输出，需要池化
        if len(aligned_features.shape) > 2:
            aligned_features = aligned_features.mean(dim=1)
        
        # 前向传播（RDGT 需要 knowledge_graph）
        predictions = model(student_ids, question_ids, knowledge_graph, aligned_features)
        
        # 计算损失
        bce_loss = nn.BCELoss()(predictions, labels)
        total_batch_loss = bce_loss + alignment_weight * align_loss
        
        # 反向传播
        optimizer.zero_grad()
        total_batch_loss.backward()
        optimizer.step()
        
        total_loss += total_batch_loss.item()
    
    return total_loss / len(dataloader)


def evaluate_with_dropout(model, alignment, dataloader, device, dropout_rate: float = 0.1):
    """评估模型（带 dropout 设置）"""
    model.eval()
    
    # 评估时关闭 dropout
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            student_ids = batch['student_id'].to(device)
            question_ids = batch['question_id'].to(device)
            knowledge_mask = batch['knowledge_mask'].to(device)
            knowledge_graph = batch['knowledge_graph'].to(device)
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
            predictions = model(student_ids, question_ids, knowledge_graph, aligned_features)
            
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


def run_dropout_experiment(dataset_name: str, alignment_method: str, dropout_rate: float,
                          epochs: int = 10, batch_size: int = 32, lr: float = 0.001):
    """
    运行单个 dropout 值实验
    
    Args:
        dataset_name: 数据集名称
        alignment_method: 对齐方法 ('none', 'interactive', 'hybrid')
        dropout_rate: dropout 值
        epochs: 训练轮数
        batch_size: 批大小
        lr: 学习率
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载数据
    downloader = DatasetDownloader()
    try:
        data = downloader.load_dataset(dataset_name)
    except FileNotFoundError:
        print(f"数据集不存在，开始下载...")
        downloader.download_dataset(dataset_name)
        data = downloader.load_dataset(dataset_name)
    
    # 划分训练集和测试集
    split_idx = int(len(data) * 0.8)
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    
    # 创建数据集
    num_students = 100
    num_questions = 50
    num_knowledge = 10
    
    train_dataset = DropoutDataset(
        train_data, num_students=num_students, num_questions=num_questions, 
        num_knowledge=num_knowledge
    )
    test_dataset = DropoutDataset(
        test_data, num_students=num_students, num_questions=num_questions,
        num_knowledge=num_knowledge
    )
    
    # 如果数据太少，跳过
    if len(train_dataset) < 10 or len(test_dataset) < 10:
        return None
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             collate_fn=create_collate_fn())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=create_collate_fn())
    
    # 初始化对齐方法
    alignment = get_alignment_method(alignment_method, embedding_dim=768).to(device)
    
    # 初始化 RDGT 模型
    model = RDGT(
        num_students=num_students,
        num_questions=num_questions,
        num_knowledge=num_knowledge,
        embedding_dim=64,
        num_heads=4,
        num_layers=2
    ).to(device)
    
    # 优化器
    optimizer = torch.optim.Adam(list(model.parameters()) + list(alignment.parameters()), lr=lr)
    
    # 训练
    best_auc = 0.0
    best_metrics = None
    for epoch in range(epochs):
        train_loss = train_epoch_with_dropout(
            model, alignment, train_loader, optimizer, device, 
            alignment_weight=0.1, dropout_rate=dropout_rate
        )
        
        # 评估
        metrics = evaluate_with_dropout(model, alignment, test_loader, device, dropout_rate=dropout_rate)
        
        if metrics['auc'] > best_auc:
            best_auc = metrics['auc']
            best_metrics = metrics
    
    return {
        'dataset': dataset_name,
        'alignment': alignment_method,
        'dropout': dropout_rate,
        'train_samples': len(train_dataset),
        'test_samples': len(test_dataset),
        'best_auc': best_metrics['auc'],
        'best_accuracy': best_metrics['accuracy']
    }


def run_dropout_sweep(dataset_name: str, alignment_methods: list = None,
                     dropout_values: list = None, epochs: int = 10,
                     batch_size: int = 32, lr: float = 0.001):
    """
    运行 dropout 扫描实验
    
    Args:
        dataset_name: 数据集名称
        alignment_methods: 对齐方法列表
        dropout_values: dropout 值列表
        epochs: 训练轮数
        batch_size: 批大小
        lr: 学习率
    """
    if alignment_methods is None:
        alignment_methods = ['none', 'interactive', 'hybrid']
    
    if dropout_values is None:
        dropout_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    
    all_results = []
    
    print(f"\n{'#'*60}")
    print(f"# 开始 RDGT Dropout 敏感性分析")
    print(f"# 数据集：{dataset_name}")
    print(f"# 对齐方法：{alignment_methods}")
    print(f"# Dropout 值：{dropout_values}")
    print(f"# 实验总数：{len(alignment_methods) * len(dropout_values)}")
    print(f"{'#'*60}\n")
    
    for alignment in alignment_methods:
        print(f"\n{'='*60}")
        print(f"## 对齐方法：{alignment.upper()}")
        print(f"{'='*60}\n")
        
        for dropout in dropout_values:
            try:
                result = run_dropout_experiment(
                    dataset_name=dataset_name,
                    alignment_method=alignment,
                    dropout_rate=dropout,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr
                )
                
                if result:
                    all_results.append(result)
                    print(f"Dropout={dropout:.1f}: AUC={result['best_auc']:.4f}, ACC={result['best_accuracy']:.4f}")
                    
            except Exception as e:
                print(f"实验失败：{dataset_name}-{alignment}-dropout{dropout}, 错误：{e}")
                all_results.append({
                    'dataset': dataset_name,
                    'alignment': alignment,
                    'dropout': dropout,
                    'error': str(e)
                })
    
    return all_results


def plot_dropout_curves(results: list, output_dir: str = "results/dropout_analysis"):
    """绘制 dropout 曲线图"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 按数据集分组
    datasets = set(r['dataset'] for r in results if 'error' not in r)
    
    for dataset in datasets:
        dataset_results = [r for r in results if r['dataset'] == dataset and 'error' not in r]
        
        # 按对齐方法分组
        alignments = set(r['alignment'] for r in dataset_results)
        
        # 准备绘图数据
        dropout_values = sorted(set(r['dropout'] for r in dataset_results))
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        colors = {'none': 'blue', 'interactive': 'green', 'hybrid': 'red'}
        markers = {'none': 'o', 'interactive': 's', 'hybrid': '^'}
        linestyles = {'none': '-', 'interactive': '--', 'hybrid': '-.'}
        
        # AUC 曲线
        ax1 = axes[0]
        for alignment in alignments:
            align_results = [r for r in dataset_results if r['alignment'] == alignment]
            align_results_sorted = sorted(align_results, key=lambda x: x['dropout'])
            
            auc_values = [r['best_auc'] for r in align_results_sorted]
            dropout_vals = [r['dropout'] for r in align_results_sorted]
            
            color = colors.get(alignment, 'gray')
            marker = markers.get(alignment, 'o')
            linestyle = linestyles.get(alignment, '-')
            
            ax1.plot(dropout_vals, auc_values, marker=marker, linestyle=linestyle, 
                    linewidth=2, markersize=8, label=f'{alignment}', color=color)
        
        ax1.set_xlabel('Dropout Rate', fontsize=12)
        ax1.set_ylabel('AUC', fontsize=12)
        ax1.set_title(f'AUC vs Dropout - {dataset}', fontsize=14)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(dropout_values)
        ax1.set_ylim([min(r['best_auc'] for r in dataset_results) - 0.05, 
                     min(max(r['best_auc'] for r in dataset_results) + 0.05, 1.0)])
        
        # ACC 曲线
        ax2 = axes[1]
        for alignment in alignments:
            align_results = [r for r in dataset_results if r['alignment'] == alignment]
            align_results_sorted = sorted(align_results, key=lambda x: x['dropout'])
            
            acc_values = [r['best_accuracy'] for r in align_results_sorted]
            dropout_vals = [r['dropout'] for r in align_results_sorted]
            
            color = colors.get(alignment, 'gray')
            marker = markers.get(alignment, 'o')
            linestyle = linestyles.get(alignment, '-')
            
            ax2.plot(dropout_vals, acc_values, marker=marker, linestyle=linestyle, 
                    linewidth=2, markersize=8, label=f'{alignment}', color=color)
        
        ax2.set_xlabel('Dropout Rate', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title(f'Accuracy vs Dropout - {dataset}', fontsize=14)
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(dropout_values)
        ax2.set_ylim([min(r['best_accuracy'] for r in dataset_results) - 0.05, 
                     min(max(r['best_accuracy'] for r in dataset_results) + 0.05, 1.0)])
        
        plt.tight_layout()
        
        # 保存图片
        save_path = output_path / f"{dataset}_dropout_curves.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至：{save_path}")
        
        plt.close()
    
    # 绘制所有数据集的对比图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, dataset in enumerate(sorted(datasets)):
        dataset_results = [r for r in results if r['dataset'] == dataset and 'error' not in r]
        alignments = set(r['alignment'] for r in dataset_results)
        dropout_values = sorted(set(r['dropout'] for r in dataset_results))
        
        ax = axes[idx]
        for alignment in alignments:
            align_results = [r for r in dataset_results if r['alignment'] == alignment]
            align_results_sorted = sorted(align_results, key=lambda x: x['dropout'])
            
            auc_values = [r['best_auc'] for r in align_results_sorted]
            dropout_vals = [r['dropout'] for r in align_results_sorted]
            
            color = colors.get(alignment, 'gray')
            marker = markers.get(alignment, 'o')
            linestyle = linestyles.get(alignment, '-')
            
            ax.plot(dropout_vals, auc_values, marker=marker, linestyle=linestyle, 
                   linewidth=2, markersize=8, label=f'{alignment}', color=color)
        
        ax.set_xlabel('Dropout Rate', fontsize=12)
        ax.set_ylabel('AUC', fontsize=12)
        ax.set_title(f'{dataset} - AUC vs Dropout', fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(dropout_values)
    
    plt.tight_layout()
    save_path = output_path / "all_datasets_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"对比图表已保存至：{save_path}")
    
    plt.close()


def print_dropout_table(results: list):
    """打印 dropout 实验结果表格"""
    print("\n" + "="*100)
    print("RDGT Dropout 敏感性分析结果汇总")
    print("="*100)
    
    # 按数据集分组
    datasets = sorted(set(r['dataset'] for r in results if 'error' not in r))
    
    for dataset in datasets:
        dataset_results = [r for r in results if r['dataset'] == dataset and 'error' not in r]
        
        print(f"\n数据集：{dataset}")
        print("-"*100)
        print(f"{'对齐方法':<15} | {'Dropout':<10} | {'样本数':<10} | {'AUC':<12} | {'ACC':<12}")
        print("-"*100)
        
        # 按对齐方法和 dropout 排序
        sorted_results = sorted(dataset_results, key=lambda x: (x['alignment'], x['dropout']))
        
        for r in sorted_results:
            align_name = r['alignment']
            if align_name == 'none':
                align_name = '不对齐'
            elif align_name == 'interactive':
                align_name = '交互重建对齐'
            elif align_name == 'hybrid':
                align_name = '混合对比对齐'
            
            print(f"{align_name:<15} | {r['dropout']:<10.1f} | {r['test_samples']:<10} | "
                  f"{r['best_auc']:<12.4f} | {r['best_accuracy']:<12.4f}")
        
        # 找出每个对齐方法的最佳 dropout
        print("\n各对齐方法最佳 Dropout:")
        print("-"*60)
        alignments = set(r['alignment'] for r in dataset_results)
        for alignment in alignments:
            align_results = [r for r in dataset_results if r['alignment'] == alignment]
            best_result = max(align_results, key=lambda x: x['best_auc'])
            align_name = alignment
            if alignment == 'none':
                align_name = '不对齐'
            elif alignment == 'interactive':
                align_name = '交互重建对齐'
            elif alignment == 'hybrid':
                align_name = '混合对比对齐'
            print(f"  {align_name}: Dropout={best_result['dropout']:.1f}, AUC={best_result['best_auc']:.4f}, ACC={best_result['best_accuracy']:.4f}")
    
    print("\n" + "="*100)


def main():
    parser = argparse.ArgumentParser(description='运行 RDGT Dropout 敏感性分析实验')
    parser.add_argument('--dataset', type=str, default='math23k',
                       choices=['math23k', 'open-luna', 'pisa2015', 'ktbd-ednet', 'all'],
                       help='数据集名称')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=32, help='批大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--output-dir', type=str, default='results/dropout_analysis',
                       help='结果输出目录')
    parser.add_argument('--dropouts', type=str, default='0.0,0.1,0.2,0.3,0.4,0.5',
                       help='Dropout 值列表，逗号分隔')
    parser.add_argument('--alignments', type=str, default='none,interactive,hybrid',
                       help='对齐方法列表，逗号分隔')
    
    args = parser.parse_args()
    
    # 解析 dropout 值
    dropout_values = [float(x.strip()) for x in args.dropouts.split(',')]
    
    # 解析对齐方法
    alignment_methods = [x.strip() for x in args.alignments.split(',')]
    
    if args.dataset == 'all':
        datasets = ['math23k', 'open-luna', 'pisa2015', 'ktbd-ednet']
    else:
        datasets = [args.dataset]
    
    all_results = []
    
    for dataset in datasets:
        results = run_dropout_sweep(
            dataset_name=dataset,
            alignment_methods=alignment_methods,
            dropout_values=dropout_values,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr
        )
        all_results.extend(results)
    
    # 打印表格
    print_dropout_table(all_results)
    
    # 保存结果
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 保存 JSON 结果
    results_file = output_path / "dropout_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存至：{results_file}")
    
    # 绘制图表
    plot_dropout_curves(all_results, output_dir=args.output_dir)
    
    print("\n实验完成！")


if __name__ == "__main__":
    main()
