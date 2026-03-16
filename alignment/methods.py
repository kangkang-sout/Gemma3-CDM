"""
对齐方法模块
实现三种对齐方式：
1. 交互重建对齐 (Interactive Reconstruction Alignment)
2. 混合对比对齐 (Hybrid Contrastive Alignment)
3. 无对齐 (No Alignment)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class InteractiveReconstructionAlignment(nn.Module):
    """
    交互重建对齐
    通过交叉注意力机制和重建损失实现对齐
    """
    
    def __init__(self, embedding_dim: int = 768, hidden_dim: int = 512):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # 交叉注意力层
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # 重建网络
        self.reconstruction_net = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # 门控机制 (输出维度需要与 embedding_dim 匹配)
        self.gate = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.Sigmoid()
        )
    
    def forward(self, report_embeddings: torch.Tensor, 
                knowledge_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            report_embeddings: 诊断报告嵌入 [batch_size, seq_len, embedding_dim]
            knowledge_embeddings: 知识点嵌入 [batch_size, num_knowledge, embedding_dim]
            
        Returns:
            aligned_embeddings: 对齐后的嵌入
            reconstruction_loss: 重建损失
        """
        batch_size = report_embeddings.size(0)
        
        # 交叉注意力：报告关注知识点
        attended_report, _ = self.cross_attention(
            query=report_embeddings,
            key=knowledge_embeddings,
            value=knowledge_embeddings
        )
        
        # 拼接原始报告和注意力输出
        combined = torch.cat([report_embeddings, attended_report], dim=-1)
        
        # 门控融合 (需要扩展 gate_weights 以匹配 attended_report 的维度)
        gate_weights = self.gate(combined)
        # gate_weights shape: [batch, seq_len, embedding_dim]
        fused = gate_weights * attended_report + (1 - gate_weights) * report_embeddings
        
        # 重建
        reconstructed = self.reconstruction_net(combined)
        
        # 重建损失 (MSE)
        reconstruction_loss = F.mse_loss(reconstructed, report_embeddings)
        
        return fused, reconstruction_loss


class HybridContrastiveAlignment(nn.Module):
    """
    混合对比对齐
    结合实例级和类别级的对比学习
    """
    
    def __init__(self, embedding_dim: int = 768, temperature: float = 0.07):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        
        # 投影头
        self.projection_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # 预测头
        self.prediction_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, embedding_dim)
        )
    
    def forward(self, report_embeddings: torch.Tensor,
                knowledge_embeddings: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            report_embeddings: 诊断报告嵌入 [batch_size, embedding_dim]
            knowledge_embeddings: 知识点嵌入 [batch_size, embedding_dim]
            labels: 可选的类别标签用于类别级对比
            
        Returns:
            aligned_embeddings: 对齐后的嵌入
            contrastive_loss: 对比损失
        """
        # 投影
        report_proj = self.projection_head(report_embeddings)
        knowledge_proj = self.projection_head(knowledge_embeddings)
        
        # 预测
        report_pred = self.prediction_head(report_proj)
        knowledge_pred = self.prediction_head(knowledge_proj)
        
        # 实例级对比损失
        instance_loss = self._contrastive_loss(report_pred, knowledge_pred)
        
        # 类别级对比损失（如果有标签）
        category_loss = 0.0
        if labels is not None:
            category_loss = self._category_contrastive_loss(
                report_pred, knowledge_pred, labels
            )
        
        # 混合损失
        contrastive_loss = instance_loss + 0.5 * category_loss
        
        # 对齐后的嵌入（平均池化）
        aligned_embeddings = (report_proj + knowledge_proj) / 2
        
        return aligned_embeddings, contrastive_loss
    
    def _contrastive_loss(self, anchor: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
        """实例级对比损失 (InfoNCE)"""
        # L2 归一化
        anchor = F.normalize(anchor, p=2, dim=-1)
        positive = F.normalize(positive, p=2, dim=-1)
        
        # 计算相似度
        similarity = torch.matmul(anchor, positive.T) / self.temperature
        
        # InfoNCE 损失
        batch_size = anchor.size(0)
        labels = torch.arange(batch_size, device=anchor.device)
        loss = F.cross_entropy(similarity, labels)
        
        return loss
    
    def _category_contrastive_loss(self, anchor: torch.Tensor, 
                                   positive: torch.Tensor,
                                   labels: torch.Tensor) -> torch.Tensor:
        """类别级对比损失"""
        # L2 归一化
        anchor = F.normalize(anchor, p=2, dim=-1)
        positive = F.normalize(positive, p=2, dim=-1)
        
        # 合并所有样本
        all_embeddings = torch.cat([anchor, positive], dim=0)
        all_labels = torch.cat([labels, labels], dim=0)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(all_embeddings, all_embeddings.T) / self.temperature
        
        # 创建掩码：相同类别为正样本对
        mask = all_labels.unsqueeze(0) == all_labels.unsqueeze(1)
        mask = mask.float()
        
        # 排除自身
        identity = torch.eye(mask.size(0), device=mask.device)
        mask = mask - identity
        
        # 计算损失
        exp_similarity = torch.exp(similarity_matrix)
        numerator = torch.sum(exp_similarity * mask, dim=1)
        denominator = torch.sum(exp_similarity * (1 - identity), dim=1)
        
        loss = -torch.log(numerator / (denominator + 1e-8)).mean()
        
        return loss


class NoAlignment(nn.Module):
    """
    无对齐
    直接使用原始嵌入，不进行任何对齐操作
    """
    
    def __init__(self, embedding_dim: int = 768):
        super().__init__()
        self.embedding_dim = embedding_dim
    
    def forward(self, report_embeddings: torch.Tensor,
                knowledge_embeddings: torch.Tensor,
                **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            report_embeddings: 诊断报告嵌入
            knowledge_embeddings: 知识点嵌入
            
        Returns:
            aligned_embeddings: 对齐后的嵌入（此处为简单拼接或平均）
            alignment_loss: 对齐损失（始终为 0）
        """
        # 简单平均作为融合策略
        if report_embeddings.shape == knowledge_embeddings.shape:
            aligned_embeddings = (report_embeddings + knowledge_embeddings) / 2
        else:
            # 如果维度不同，使用报告嵌入
            aligned_embeddings = report_embeddings
        
        # 无对齐损失
        alignment_loss = torch.tensor(0.0, device=report_embeddings.device)
        
        return aligned_embeddings, alignment_loss


def get_alignment_method(method_name: str, embedding_dim: int = 768) -> nn.Module:
    """
    获取对齐方法
    
    Args:
        method_name: 对齐方法名称 ('interactive', 'hybrid', 'none')
        embedding_dim: 嵌入维度
        
    Returns:
        对齐方法模块
    """
    methods = {
        'interactive': InteractiveReconstructionAlignment,
        'hybrid': HybridContrastiveAlignment,
        'none': NoAlignment
    }
    
    if method_name not in methods:
        raise ValueError(f"未知的对齐方法：{method_name}. 可用方法：{list(methods.keys())}")
    
    return methods[method_name](embedding_dim=embedding_dim)


if __name__ == "__main__":
    # 测试对齐方法
    batch_size = 4
    embedding_dim = 768
    seq_len = 10
    num_knowledge = 5
    
    # 创建测试数据
    report_emb = torch.randn(batch_size, seq_len, embedding_dim)
    knowledge_emb = torch.randn(batch_size, num_knowledge, embedding_dim)
    labels = torch.randint(0, 3, (batch_size,))
    
    print("测试交互重建对齐...")
    interactive = InteractiveReconstructionAlignment(embedding_dim)
    aligned, recon_loss = interactive(report_emb, knowledge_emb)
    print(f"输出形状：{aligned.shape}, 重建损失：{recon_loss.item():.4f}")
    
    print("\n测试混合对比对齐...")
    hybrid = HybridContrastiveAlignment(embedding_dim)
    report_emb_flat = torch.randn(batch_size, embedding_dim)
    knowledge_emb_flat = torch.randn(batch_size, embedding_dim)
    aligned, contra_loss = hybrid(report_emb_flat, knowledge_emb_flat, labels)
    print(f"输出形状：{aligned.shape}, 对比损失：{contra_loss.item():.4f}")
    
    print("\n测试无对齐...")
    no_align = NoAlignment()
    aligned, no_loss = no_align(report_emb_flat, knowledge_emb_flat)
    print(f"输出形状：{aligned.shape}, 对齐损失：{no_loss.item():.4f}")
