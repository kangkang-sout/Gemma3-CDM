"""
RDGT 模型 Dropout 敏感性分析实验（轻量级测试版）
使用模拟数据快速验证代码逻辑
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# 导入自定义模块
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from alignment.methods import get_alignment_method
from models.cdm.models import RDGT


class MockDataset(Dataset):
    """模拟数据集用于快速测试"""
    
    def __init__(self, num_samples: int = 100, num_students: int = 50, 
                 num_questions: int = 30, num_knowledge: int = 10):
        self.num_samples = num_samples
        self.num_students = num_students
        self.num_questions = num_questions
        self.num_knowledge = num_knowledge
        
        # 生成随机数据
        self.student_ids = torch.randint(0, num_students, (num_samples,))
        self.question_ids = torch.randint(0, num_questions, (num_samples,))
        self.labels = torch.rand(num_samples)
        
        # 生成知识图谱
        self.knowledge_graphs = []
        self.knowledge_masks = []
        for _ in range(num_samples):
            kg = torch.zeros(num_knowledge, num_knowledge)
            km = torch.zeros(num_knowledge)
            
            # 随机选择几个知识点
            num_selected = np.random.randint(2, 5)
            selected = np.random.choice(num_knowledge, num_selected, replace=False)
            
            for kid in selected:
                km[kid] = 1.0
                kg[kid, kid] = 1.0
            
            self.knowledge_graphs.append(kg)
            self.knowledge_masks.append(km)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return {
            'student_id': self.student_ids[idx],
            'question_id': self.question_ids[idx],
            'knowledge_mask': self.knowledge_masks[idx],
            'knowledge_graph': self.knowledge_graphs[idx],
            'label': self.labels[idx]
        }


def create_collate_fn():
    def collate_fn(batch):
        return {
            'student_id': torch.stack([item['student_id'] for item in batch]),
            'question_id': torch.stack([item['question_id'] for item in batch]),
            'knowledge_mask': torch.stack([item['knowledge_mask'] for item in batch]),
            'knowledge_graph': torch.stack([item['knowledge_graph'] for item in batch]),
            'label': torch.stack([item['label'] for item in batch])
        }
    return collate_fn


def train_and_evaluate(dropout_rate: float, alignment_method: str, 
                       epochs: int = 3, lr: float = 0.001):
    """训练和评估单个配置"""
    device = torch.device('cpu')
    
    # 创建数据
    train_dataset = MockDataset(num_samples=80)
    test_dataset = MockDataset(num_samples=20)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, 
                             collate_fn=create_collate_fn())
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False,
                            collate_fn=create_collate_fn())
    
    # 初始化对齐方法
    alignment = get_alignment_method(alignment_method, embedding_dim=768).to(device)
    
    # 初始化 RDGT 模型
    model = RDGT(
        num_students=50,
        num_questions=30,
        num_knowledge=10,
        embedding_dim=64,
        num_heads=4,
        num_layers=2
    ).to(device)
    
    # 设置 dropout
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = dropout_rate
    
    optimizer = torch.optim.Adam(list(model.parameters()) + list(alignment.parameters()), lr=lr)
    
    # 训练
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            student_ids = batch['student_id'].to(device)
            question_ids = batch['question_id'].to(device)
            knowledge_graph = batch['knowledge_graph'].to(device)
            labels = batch['label'].to(device)
            
            batch_size = student_ids.size(0)
            report_emb = torch.randn(batch_size, 768).to(device)
            knowledge_emb = torch.randn(batch_size, 768).to(device)
            
            aligned_features, align_loss = alignment(report_emb, knowledge_emb)
            if len(aligned_features.shape) > 2:
                aligned_features = aligned_features.mean(dim=1)
            
            predictions = model(student_ids, question_ids, knowledge_graph, aligned_features)
            
            bce_loss = nn.BCELoss()(predictions, labels)
            total_loss = bce_loss + 0.1 * align_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
    
    # 评估
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            student_ids = batch['student_id'].to(device)
            question_ids = batch['question_id'].to(device)
            knowledge_graph = batch['knowledge_graph'].to(device)
            labels = batch['label'].to(device)
            
            batch_size = student_ids.size(0)
            report_emb = torch.randn(batch_size, 768).to(device)
            knowledge_emb = torch.randn(batch_size, 768).to(device)
            
            aligned_features, _ = alignment(report_emb, knowledge_emb)
            if len(aligned_features.shape) > 2:
                aligned_features = aligned_features.mean(dim=1)
            
            predictions = model(student_ids, question_ids, knowledge_graph, aligned_features)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    predictions = np.array(all_predictions)
    labels = np.array(all_labels)
    
    try:
        auc = roc_auc_score(labels, predictions)
    except:
        auc = 0.5
    
    binary_preds = (predictions > 0.5).astype(int)
    binary_labels = (labels > 0.5).astype(int)
    acc = accuracy_score(binary_labels, binary_preds)
    
    return {'auc': auc, 'accuracy': acc}


def run_test():
    """运行完整的 dropout 扫描测试"""
    alignment_methods = ['none', 'interactive', 'hybrid']
    dropout_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    
    all_results = []
    
    print("\n" + "="*60)
    print("RDGT Dropout 敏感性分析（轻量级测试）")
    print("="*60 + "\n")
    
    for alignment in alignment_methods:
        print(f"\n对齐方法：{alignment}")
        print("-"*40)
        
        for dropout in dropout_values:
            result = train_and_evaluate(
                dropout_rate=dropout,
                alignment_method=alignment,
                epochs=3
            )
            
            all_results.append({
                'alignment': alignment,
                'dropout': dropout,
                'auc': result['auc'],
                'accuracy': result['accuracy']
            })
            
            print(f"  Dropout={dropout:.1f}: AUC={result['auc']:.4f}, ACC={result['accuracy']:.4f}")
    
    # 绘制图表
    plot_results(all_results)
    
    # 打印表格
    print_table(all_results)
    
    return all_results


def plot_results(results):
    """绘制结果图表"""
    output_dir = Path("results/dropout_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    colors = {'none': 'blue', 'interactive': 'green', 'hybrid': 'red'}
    markers = {'none': 'o', 'interactive': 's', 'hybrid': '^'}
    linestyles = {'none': '-', 'interactive': '--', 'hybrid': '-.'}
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # AUC 曲线
    ax1 = axes[0]
    for alignment in ['none', 'interactive', 'hybrid']:
        align_results = [r for r in results if r['alignment'] == alignment]
        align_results_sorted = sorted(align_results, key=lambda x: x['dropout'])
        
        auc_values = [r['auc'] for r in align_results_sorted]
        dropout_vals = [r['dropout'] for r in align_results_sorted]
        
        color = colors.get(alignment, 'gray')
        marker = markers.get(alignment, 'o')
        linestyle = linestyles.get(alignment, '-')
        
        ax1.plot(dropout_vals, auc_values, marker=marker, linestyle=linestyle, 
                linewidth=2, markersize=8, label=f'{alignment}', color=color)
    
    ax1.set_xlabel('Dropout Rate', fontsize=12)
    ax1.set_ylabel('AUC', fontsize=12)
    ax1.set_title('AUC vs Dropout (Mock Data)', fontsize=14)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    
    # ACC 曲线
    ax2 = axes[1]
    for alignment in ['none', 'interactive', 'hybrid']:
        align_results = [r for r in results if r['alignment'] == alignment]
        align_results_sorted = sorted(align_results, key=lambda x: x['dropout'])
        
        acc_values = [r['accuracy'] for r in align_results_sorted]
        dropout_vals = [r['dropout'] for r in align_results_sorted]
        
        color = colors.get(alignment, 'gray')
        marker = markers.get(alignment, 'o')
        linestyle = linestyles.get(alignment, '-')
        
        ax2.plot(dropout_vals, acc_values, marker=marker, linestyle=linestyle, 
                linewidth=2, markersize=8, label=f'{alignment}', color=color)
    
    ax2.set_xlabel('Dropout Rate', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Accuracy vs Dropout (Mock Data)', fontsize=14)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    
    plt.tight_layout()
    
    save_path = output_dir / "mock_dropout_curves.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n图表已保存至：{save_path}")
    
    plt.close()


def print_table(results):
    """打印结果表格"""
    print("\n" + "="*80)
    print("RDGT Dropout 敏感性分析结果汇总（模拟数据）")
    print("="*80)
    print(f"{'对齐方法':<15} | {'Dropout':<10} | {'AUC':<12} | {'ACC':<12}")
    print("-"*80)
    
    sorted_results = sorted(results, key=lambda x: (x['alignment'], x['dropout']))
    
    for r in sorted_results:
        align_name = r['alignment']
        if align_name == 'none':
            align_name = '不对齐'
        elif align_name == 'interactive':
            align_name = '交互重建对齐'
        elif align_name == 'hybrid':
            align_name = '混合对比对齐'
        
        print(f"{align_name:<15} | {r['dropout']:<10.1f} | {r['auc']:<12.4f} | {r['accuracy']:<12.4f}")
    
    print("="*80)


if __name__ == "__main__":
    run_test()
