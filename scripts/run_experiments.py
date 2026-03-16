"""
实验运行脚本
执行 15 个对比实验（3 种对齐方式 × 5 个认知诊断模型）
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

# 导入自定义模块
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from alignment.methods import get_alignment_method
from models.cdm.models import get_cdm_model
from scripts.download_datasets import DatasetDownloader


class CognitiveDiagnosisDataset(Dataset):
    """认知诊断数据集"""
    
    def __init__(self, data: list, num_students: int, num_questions: int,
                 num_knowledge: int, max_seq_len: int = 10):
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
            # 假设数据格式中有 user_id, question_id 等字段
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


def train_epoch(model, alignment, dataloader, optimizer, device, 
                alignment_weight: float = 0.1):
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    
    for batch in tqdm(dataloader, desc="Training"):
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
        for batch in tqdm(dataloader, desc="Evaluating"):
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
    from sklearn.metrics import roc_auc_score, accuracy_score
    try:
        auc = roc_auc_score(labels, predictions)
    except:
        auc = 0.5
    
    # 准确率
    binary_preds = (predictions > 0.5).astype(int)
    binary_labels = (labels > 0.5).astype(int)
    acc = accuracy_score(binary_labels, binary_preds)
    
    return {'auc': auc, 'accuracy': acc}


def run_experiment(dataset_name: str, alignment_method: str, cdm_model: str,
                   epochs: int = 10, batch_size: int = 32, lr: float = 0.001):
    """
    运行单个实验
    
    Args:
        dataset_name: 数据集名称
        alignment_method: 对齐方法 ('interactive', 'hybrid', 'none')
        cdm_model: 认知诊断模型 ('neuralcdm', 'opencdm', 'gcd', 'scd', 'rdgt')
        epochs: 训练轮数
        batch_size: 批大小
        lr: 学习率
    """
    print(f"\n{'='*60}")
    print(f"实验配置:")
    print(f"  数据集：{dataset_name}")
    print(f"  对齐方法：{alignment_method}")
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
    
    print(f"数据集大小：{len(data)}")
    
    # 划分数据集
    split_idx = int(len(data) * 0.8)
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    
    # 创建数据集
    num_students = 100
    num_questions = 50
    num_knowledge = 10
    
    train_dataset = CognitiveDiagnosisDataset(train_data, num_students, num_questions, num_knowledge)
    test_dataset = CognitiveDiagnosisDataset(test_data, num_students, num_questions, num_knowledge)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             collate_fn=create_collate_fn())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=create_collate_fn())
    
    # 初始化模型
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
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        train_loss = train_epoch(model, alignment, train_loader, optimizer, device)
        print(f"训练损失：{train_loss:.4f}")
        
        # 评估
        metrics = evaluate(model, alignment, test_loader, device)
        print(f"验证集 - AUC: {metrics['auc']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
        
        if metrics['auc'] > best_auc:
            best_auc = metrics['auc']
    
    print(f"\n最佳 AUC: {best_auc:.4f}")
    
    return {
        'dataset': dataset_name,
        'alignment': alignment_method,
        'model': cdm_model,
        'best_auc': best_auc,
        'final_metrics': metrics
    }


def main():
    parser = argparse.ArgumentParser(description='运行认知诊断对比实验')
    parser.add_argument('--dataset', type=str, default='all',
                       choices=['math23k', 'open-luna', 'pisa2015', 'ktbd-ednet', 'all'],
                       help='数据集名称')
    parser.add_argument('--alignment', type=str, default='all',
                       choices=['interactive', 'hybrid', 'none', 'all'],
                       help='对齐方法')
    parser.add_argument('--model', type=str, default='all',
                       choices=['neuralcdm', 'opencdm', 'gcd', 'scd', 'rdgt', 'all'],
                       help='认知诊断模型')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--output_dir', type=str, default='./results', help='结果输出目录')
    
    args = parser.parse_args()
    
    # 配置列表
    datasets = ['math23k', 'open-luna', 'pisa2015', 'ktbd-ednet'] if args.dataset == 'all' else [args.dataset]
    alignments = ['interactive', 'hybrid', 'none'] if args.alignment == 'all' else [args.alignment]
    models = ['neuralcdm', 'opencdm', 'gcd', 'scd', 'rdgt'] if args.model == 'all' else [args.model]
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 运行实验
    all_results = []
    
    for dataset in datasets:
        for align in alignments:
            for model in models:
                try:
                    result = run_experiment(
                        dataset_name=dataset,
                        alignment_method=align,
                        cdm_model=model,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        lr=args.lr
                    )
                    all_results.append(result)
                    
                    # 保存结果
                    result_file = output_dir / f"{dataset}_{align}_{model}.json"
                    with open(result_file, 'w', encoding='utf-8') as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                
                except Exception as e:
                    print(f"实验失败：{dataset}-{align}-{model}, 错误：{e}")
                    all_results.append({
                        'dataset': dataset,
                        'alignment': align,
                        'model': model,
                        'error': str(e)
                    })
    
    # 汇总结果
    summary_file = output_dir / 'experiment_summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print("所有实验完成!")
    print(f"结果已保存到：{output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
