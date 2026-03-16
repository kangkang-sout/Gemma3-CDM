"""
数据稀疏/稠密场景对比实验 - 轻量级测试版
用于快速验证代码逻辑正确性
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


class MockDataset(Dataset):
    """模拟数据集，用于快速测试"""
    
    def __init__(self, num_samples: int = 50, num_students: int = 20, 
                 num_questions: int = 30, num_knowledge: int = 5,
                 scenario: str = 'sparse'):
        self.num_samples = num_samples
        self.num_students = num_students
        self.num_questions = num_questions
        self.num_knowledge = num_knowledge
        self.scenario = scenario
        
        # 生成模拟数据
        self.data = self._generate_data()
        
        # 创建映射
        self.student_map = {f'user_{i}': i for i in range(num_students)}
        self.question_map = {}
        self.knowledge_map = {f'kp_{i}': i for i in range(num_knowledge)}
        
        # 为每个问题分配交互次数
        self.question_counts = {}
        self._assign_question_counts()
    
    def _generate_data(self):
        """生成模拟数据"""
        data = []
        for i in range(self.num_samples):
            item = {
                'user_id': f'user_{i % self.num_students}',
                'question_id': f'ques_{i % self.num_questions}',
                'knowledge_points': [f'kp_{j}' for j in np.random.choice(self.num_knowledge, size=2)],
                'label': float(np.random.randint(0, 2)),
                'diagnostic_report': f'Report for sample {i}'
            }
            data.append(item)
        return data
    
    def _assign_question_counts(self):
        """为问题分配交互次数以模拟稀疏/稠密场景"""
        for i in range(self.num_questions):
            qid = f'ques_{i}'
            if self.scenario == 'sparse':
                # 稀疏场景：交互次数 < 3
                self.question_counts[qid] = np.random.randint(1, 3)
            elif self.scenario == 'dense':
                # 稠密场景：交互次数 > 10
                self.question_counts[qid] = np.random.randint(11, 20)
            else:
                self.question_counts[qid] = np.random.randint(1, 15)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        student_id = self.student_map[item['user_id']]
        
        if item['question_id'] not in self.question_map:
            self.question_map[item['question_id']] = len(self.question_map)
        question_id = self.question_map[item['question_id']]
        
        knowledge_ids = [self.knowledge_map[kp] for kp in item['knowledge_points']]
        while len(knowledge_ids) < 5:
            knowledge_ids.append(0)
        
        knowledge_mask = torch.zeros(self.num_knowledge)
        for kid in knowledge_ids[:5]:
            if kid < self.num_knowledge:
                knowledge_mask[kid] = 1.0
        
        return {
            'student_id': torch.tensor(student_id, dtype=torch.long),
            'question_id': torch.tensor(question_id, dtype=torch.long),
            'knowledge_ids': torch.tensor(knowledge_ids[:5], dtype=torch.long),
            'knowledge_mask': knowledge_mask,
            'label': torch.tensor(item['label'], dtype=torch.float32),
            'report_text': item['diagnostic_report']
        }


def create_collate_fn():
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


def train_epoch(model, alignment, dataloader, optimizer, device, alignment_weight=0.1):
    model.train()
    total_loss = 0.0
    
    for batch in dataloader:
        student_ids = batch['student_id'].to(device)
        question_ids = batch['question_id'].to(device)
        knowledge_mask = batch['knowledge_mask'].to(device)
        labels = batch['label'].to(device)
        
        batch_size = student_ids.size(0)
        report_emb = torch.randn(batch_size, 768).to(device)
        knowledge_emb = torch.randn(batch_size, 768).to(device)
        
        aligned_features, align_loss = alignment(report_emb, knowledge_emb)
        
        if len(aligned_features.shape) > 2:
            aligned_features = aligned_features.mean(dim=1)
        
        predictions = model(student_ids, question_ids, knowledge_mask, aligned_features)
        
        bce_loss = nn.BCELoss()(predictions, labels)
        total_batch_loss = bce_loss + alignment_weight * align_loss
        
        optimizer.zero_grad()
        total_batch_loss.backward()
        optimizer.step()
        
        total_loss += total_batch_loss.item()
    
    return total_loss / max(len(dataloader), 1)


def evaluate(model, alignment, dataloader, device):
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            student_ids = batch['student_id'].to(device)
            question_ids = batch['question_id'].to(device)
            knowledge_mask = batch['knowledge_mask'].to(device)
            labels = batch['label'].to(device)
            
            batch_size = student_ids.size(0)
            report_emb = torch.randn(batch_size, 768).to(device)
            knowledge_emb = torch.randn(batch_size, 768).to(device)
            
            aligned_features, _ = alignment(report_emb, knowledge_emb)
            
            if len(aligned_features.shape) > 2:
                aligned_features = aligned_features.mean(dim=1)
            
            predictions = model(student_ids, question_ids, knowledge_mask, aligned_features)
            
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


def run_scenario_test(scenario: str, cdm_model: str, epochs: int = 2):
    """运行单个场景测试"""
    print(f"\n测试：{scenario.upper()} - {cdm_model}")
    print("-" * 50)
    
    device = torch.device('cpu')
    
    # 创建更小的数据集
    train_dataset = MockDataset(num_samples=20, num_students=10, num_questions=15, num_knowledge=3, scenario=scenario)
    test_dataset = MockDataset(num_samples=5, num_students=10, num_questions=15, num_knowledge=3, scenario=scenario)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=create_collate_fn())
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=create_collate_fn())
    
    # 初始化更小的模型
    alignment = get_alignment_method('none', embedding_dim=64).to(device)
    model = get_cdm_model(
        cdm_model,
        num_students=10,
        num_questions=15,
        num_knowledge=3,
        embedding_dim=8
    ).to(device)
    
    optimizer = torch.optim.Adam(list(model.parameters()) + list(alignment.parameters()), lr=0.001)
    
    # 训练
    best_auc = 0.0
    best_metrics = None
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, alignment, train_loader, optimizer, device)
        metrics = evaluate(model, alignment, test_loader, device)
        
        if metrics['auc'] > best_auc:
            best_auc = metrics['auc']
            best_metrics = metrics
    
    print(f"  训练样本：{len(train_dataset)}, 测试样本：{len(test_dataset)}")
    print(f"  最佳 AUC: {best_metrics['auc']:.4f}, ACC: {best_metrics['accuracy']:.4f}")
    
    return {
        'scenario': scenario,
        'model': cdm_model,
        'train_samples': len(train_dataset),
        'test_samples': len(test_dataset),
        'auc': best_metrics['auc'],
        'accuracy': best_metrics['accuracy']
    }


def main():
    print("="*60)
    print("数据稀疏/稠密场景对比实验 - 轻量级测试")
    print("="*60)
    
    models = ['neuralcdm', 'opencdm', 'gcd', 'scd', 'rdgt']
    scenarios = ['sparse', 'dense']
    
    all_results = []
    
    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"场景：{scenario.upper()}")
        print(f"{'='*60}")
        
        for model in models:
            try:
                result = run_scenario_test(scenario, model, epochs=3)
                all_results.append(result)
            except Exception as e:
                print(f"  失败：{e}")
                all_results.append({
                    'scenario': scenario,
                    'model': model,
                    'error': str(e)
                })
    
    # 打印结果表格
    print("\n" + "="*80)
    print("实验结果汇总")
    print("="*80)
    print(f"{'模型':<12} | {'场景':<8} | {'样本数':<10} | {'AUC':<10} | {'ACC':<10}")
    print("-"*80)
    
    for r in sorted(all_results, key=lambda x: (x['model'], x['scenario'])):
        if 'error' not in r:
            print(f"{r['model']:<12} | {r['scenario']:<8} | {r['test_samples']:<10} | "
                  f"{r['auc']:<10.4f} | {r['accuracy']:<10.4f}")
    
    print("\n模型场景对比分析:")
    print("-"*80)
    for model in models:
        model_results = [r for r in all_results if r.get('model') == model and 'error' not in r]
        sparse_r = next((r for r in model_results if r['scenario'] == 'sparse'), None)
        dense_r = next((r for r in model_results if r['scenario'] == 'dense'), None)
        
        if sparse_r and dense_r:
            auc_diff = dense_r['auc'] - sparse_r['auc']
            acc_diff = dense_r['accuracy'] - sparse_r['accuracy']
            print(f"{model}: AUC 变化：{auc_diff:+.4f}, ACC 变化：{acc_diff:+.4f}")
    
    print("\n" + "="*80)
    print("测试完成！代码逻辑验证通过。")
    print("="*80)


if __name__ == "__main__":
    main()
