"""
认知诊断模型模块
实现五种认知诊断模型：
1. NeuralCDM - 神经认知诊断模型
2. OpenCDM - 开放认知诊断模型
3. GCD - 图认知诊断
4. SCD - 序列认知诊断
5. RDGT - 关系感知诊断图 Transformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class NeuralCDM(nn.Module):
    """
    NeuralCDM: 神经认知诊断模型
    基于 IRT 和 MIRT 的神经网络扩展
    """
    
    def __init__(self, num_students: int, num_questions: int, 
                 num_knowledge: int, embedding_dim: int = 64):
        super().__init__()
        self.num_students = num_students
        self.num_questions = num_questions
        self.num_knowledge = num_knowledge
        
        # 学生能力嵌入
        self.student_emb = nn.Embedding(num_students, embedding_dim)
        
        # 题目特征嵌入
        self.question_emb = nn.Embedding(num_questions, embedding_dim)
        
        # 知识点嵌入
        self.knowledge_emb = nn.Embedding(num_knowledge, embedding_dim)
        
        # Q 矩阵映射（题目到知识点）
        self.q_matrix = nn.Linear(embedding_dim, num_knowledge)
        
        # 诊断网络
        self.diagnosis_net = nn.Sequential(
            nn.Linear(embedding_dim * 3 + num_knowledge, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, student_ids: torch.Tensor, question_ids: torch.Tensor,
                knowledge_ids: torch.Tensor, aligned_features: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            student_ids: 学生 ID [batch_size]
            question_ids: 题目 ID [batch_size]
            knowledge_ids: 知识点 ID [batch_size, num_knowledge]
            aligned_features: 对齐后的特征 [batch_size, feature_dim]
            
        Returns:
            预测的作答正确率
        """
        # 获取嵌入
        student_feat = self.student_emb(student_ids)
        question_feat = self.question_emb(question_ids)
        knowledge_feat = self.knowledge_emb(knowledge_ids).mean(dim=1)
        
        # Q 矩阵映射
        q_feat = torch.sigmoid(self.q_matrix(question_feat))
        
        # 拼接特征
        combined = torch.cat([student_feat, question_feat, knowledge_feat, aligned_features], dim=-1)
        
        # 诊断预测
        prediction = torch.sigmoid(self.diagnosis_net(combined))
        
        return prediction.squeeze(-1)


class OpenCDM(nn.Module):
    """
    OpenCDM: 开放认知诊断模型
    支持开放式题目的认知诊断
    """
    
    def __init__(self, num_students: int, num_questions: int,
                 num_knowledge: int, embedding_dim: int = 64):
        super().__init__()
        self.num_students = num_students
        self.num_questions = num_questions
        self.num_knowledge = num_knowledge
        
        # 学生多维能力
        self.student_alpha = nn.Embedding(num_students, num_knowledge)
        
        # 题目参数
        self.question_difficulty = nn.Embedding(num_questions, 1)
        self.question_discrimination = nn.Embedding(num_questions, num_knowledge)
        
        # 知识点关联
        self.knowledge_emb = nn.Embedding(num_knowledge, embedding_dim)
        
        # 文本特征融合
        self.text_fusion = nn.Linear(embedding_dim, num_knowledge)
        
        # 评分网络（用于开放式题目）
        self.scoring_net = nn.Sequential(
            nn.Linear(num_knowledge * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.1)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
    
    def forward(self, student_ids: torch.Tensor, question_ids: torch.Tensor,
                knowledge_mask: torch.Tensor, aligned_features: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            student_ids: 学生 ID [batch_size]
            question_ids: 题目 ID [batch_size]
            knowledge_mask: 知识点掩码 [batch_size, num_knowledge]
            aligned_features: 对齐后的文本特征 [batch_size, embedding_dim]
            
        Returns:
            预测得分
        """
        # 学生能力
        alpha = torch.sigmoid(self.student_alpha(student_ids))
        
        # 题目参数
        difficulty = torch.sigmoid(self.question_difficulty(question_ids))
        discrimination = torch.sigmoid(self.question_discrimination(question_ids))
        
        # 文本特征融合
        text_feat = torch.sigmoid(self.text_fusion(aligned_features))
        
        # 应用知识点掩码
        alpha_masked = alpha * knowledge_mask
        disc_masked = discrimination * knowledge_mask
        
        # 计算掌握度（点积）
        mastery = (alpha_masked * disc_masked).sum(dim=-1, keepdim=True)
        
        # 结合文本特征
        combined = torch.cat([mastery, text_feat], dim=-1)
        
        # 最终得分
        score = torch.sigmoid(self.scoring_net(combined) - difficulty)
        
        return score.squeeze(-1)


class GCD(nn.Module):
    """
    GCD: Graph Cognitive Diagnosis
    基于图神经网络的认知诊断
    """
    
    def __init__(self, num_students: int, num_questions: int,
                 num_knowledge: int, embedding_dim: int = 64,
                 num_gcn_layers: int = 2):
        super().__init__()
        self.num_students = num_students
        self.num_questions = num_questions
        self.num_knowledge = num_knowledge
        
        # 节点嵌入
        self.student_emb = nn.Embedding(num_students, embedding_dim)
        self.question_emb = nn.Embedding(num_questions, embedding_dim)
        self.knowledge_emb = nn.Embedding(num_knowledge, embedding_dim)
        
        # 图卷积层
        self.gcn_layers = nn.ModuleList()
        for i in range(num_gcn_layers):
            self.gcn_layers.append(
                nn.Linear(embedding_dim, embedding_dim)
            )
        
        # 边权重学习
        self.edge_attention = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.Tanh(),
            nn.Linear(embedding_dim, 1)
        )
        
        # 预测头
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
    
    def forward(self, student_ids: torch.Tensor, question_ids: torch.Tensor,
                knowledge_adj: torch.Tensor, aligned_features: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            student_ids: 学生 ID [batch_size]
            question_ids: 题目 ID [batch_size]
            knowledge_adj: 知识点邻接矩阵 [num_knowledge, num_knowledge]
            aligned_features: 对齐后的特征 [batch_size, embedding_dim]
            
        Returns:
            预测正确率
        """
        batch_size = student_ids.size(0)
        
        # 获取节点嵌入
        student_feat = self.student_emb(student_ids)
        question_feat = self.question_emb(question_ids)
        knowledge_feat = self.knowledge_emb.weight
        
        # 图卷积传播
        for gcn in self.gcn_layers:
            knowledge_feat = torch.relu(gcn(knowledge_feat))
            # 邻接矩阵传播
            knowledge_feat = torch.matmul(knowledge_adj, knowledge_feat)
        
        # 计算学生 - 知识点注意力
        student_knowledge_edge = torch.cat([
            student_feat.unsqueeze(1).expand(-1, self.num_knowledge, -1),
            knowledge_feat.unsqueeze(0).expand(batch_size, -1, -1)
        ], dim=-1)
        
        attention_weights = torch.softmax(
            self.edge_attention(student_knowledge_edge).squeeze(-1),
            dim=-1
        )
        
        # 聚合知识点信息
        student_knowledge = (attention_weights.unsqueeze(-1) * knowledge_feat).sum(dim=1)
        
        # 拼接特征
        combined = torch.cat([student_feat, question_feat, student_knowledge, aligned_features], dim=-1)
        
        # 预测
        prediction = torch.sigmoid(self.predictor(combined))
        
        return prediction.squeeze(-1)


class SCD(nn.Module):
    """
    SCD: Sequential Cognitive Diagnosis
    序列认知诊断，考虑时间依赖性
    """
    
    def __init__(self, num_students: int, num_questions: int,
                 num_knowledge: int, embedding_dim: int = 64,
                 hidden_dim: int = 128):
        super().__init__()
        self.num_students = num_students
        self.num_questions = num_questions
        self.num_knowledge = num_knowledge
        
        # 学生状态嵌入
        self.student_emb = nn.Embedding(num_students, embedding_dim)
        
        # 题目嵌入
        self.question_emb = nn.Embedding(num_questions, embedding_dim)
        
        # LSTM 用于序列建模
        self.lstm = nn.LSTM(
            input_size=embedding_dim * 2,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 诊断网络
        self.diagnosis_net = nn.Sequential(
            nn.Linear(hidden_dim + embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
    
    def forward(self, student_ids: torch.Tensor, question_seqs: torch.Tensor,
                response_seqs: torch.Tensor, aligned_features: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            student_ids: 学生 ID [batch_size]
            question_seqs: 题目序列 [batch_size, seq_len]
            response_seqs: 作答序列 [batch_size, seq_len]
            aligned_features: 对齐后的特征 [batch_size, embedding_dim]
            
        Returns:
            预测正确率
        """
        batch_size = student_ids.size(0)
        
        # 获取学生嵌入
        student_feat = self.student_emb(student_ids)
        
        # 获取序列嵌入
        question_emb = self.question_emb(question_seqs)
        response_emb = response_seqs.unsqueeze(-1).float()
        
        # 拼接输入
        lstm_input = torch.cat([question_emb, response_emb], dim=-1)
        
        # LSTM 编码
        lstm_out, (h_n, c_n) = self.lstm(lstm_input)
        
        # 注意力加权
        attention_scores = torch.softmax(self.attention(lstm_out), dim=1)
        context = (attention_scores * lstm_out).sum(dim=1)
        
        # 拼接学生特征
        combined = torch.cat([context, student_feat, aligned_features], dim=-1)
        
        # 预测
        prediction = torch.sigmoid(self.diagnosis_net(combined))
        
        return prediction.squeeze(-1)


class RDGT(nn.Module):
    """
    RDGT: Relation-aware Diagnostic Graph Transformer
    关系感知诊断图 Transformer
    """
    
    def __init__(self, num_students: int, num_questions: int,
                 num_knowledge: int, embedding_dim: int = 64,
                 num_heads: int = 4, num_layers: int = 2):
        super().__init__()
        self.num_students = num_students
        self.num_questions = num_questions
        self.num_knowledge = num_knowledge
        
        # 嵌入层
        self.student_emb = nn.Embedding(num_students, embedding_dim)
        self.question_emb = nn.Embedding(num_questions, embedding_dim)
        self.knowledge_emb = nn.Embedding(num_knowledge, embedding_dim)
        
        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(1, 100, embedding_dim) * 0.1)
        
        # Transformer 编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 关系注意力
        self.relation_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # 预测头
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
    
    def forward(self, student_ids: torch.Tensor, question_ids: torch.Tensor,
                knowledge_graph: torch.Tensor, aligned_features: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            student_ids: 学生 ID [batch_size]
            question_ids: 题目 ID [batch_size]
            knowledge_graph: 知识图谱 [batch_size, num_knowledge, num_knowledge]
            aligned_features: 对齐后的特征 [batch_size, embedding_dim]
            
        Returns:
            预测正确率
        """
        batch_size = student_ids.size(0)
        
        # 获取嵌入
        student_feat = self.student_emb(student_ids).unsqueeze(1)
        question_feat = self.question_emb(question_ids).unsqueeze(1)
        knowledge_feat = self.knowledge_emb.weight.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 添加位置编码
        knowledge_feat = knowledge_feat + self.pos_encoding[:, :knowledge_feat.size(1), :]
        
        # Transformer 编码
        knowledge_encoded = self.transformer_encoder(knowledge_feat)
        
        # 关系感知注意力
        student_query = student_feat
        knowledge_attended, _ = self.relation_attention(
            query=student_query,
            key=knowledge_encoded,
            value=knowledge_encoded,
            attn_mask=None
        )
        
        # 聚合
        knowledge_agg = knowledge_attended.mean(dim=1)
        
        # 拼接特征
        combined = torch.cat([student_feat.squeeze(1), question_feat.squeeze(1), 
                             knowledge_agg, aligned_features], dim=-1)
        
        # 预测
        prediction = torch.sigmoid(self.predictor(combined))
        
        return prediction.squeeze(-1)


def get_cdm_model(model_name: str, **kwargs) -> nn.Module:
    """
    获取认知诊断模型
    
    Args:
        model_name: 模型名称 ('neuralcdm', 'opencdm', 'gcd', 'scd', 'rdgt')
        **kwargs: 模型参数
        
    Returns:
        认知诊断模型
    """
    models = {
        'neuralcdm': NeuralCDM,
        'opencdm': OpenCDM,
        'gcd': GCD,
        'scd': SCD,
        'rdgt': RDGT
    }
    
    if model_name not in models:
        raise ValueError(f"未知的模型：{model_name}. 可用模型：{list(models.keys())}")
    
    return models[model_name](**kwargs)


if __name__ == "__main__":
    # 测试模型
    batch_size = 4
    num_students = 100
    num_questions = 50
    num_knowledge = 10
    embedding_dim = 64
    
    print("测试 NeuralCDM...")
    neuralcdm = NeuralCDM(num_students, num_questions, num_knowledge, embedding_dim)
    student_ids = torch.randint(0, num_students, (batch_size,))
    question_ids = torch.randint(0, num_questions, (batch_size,))
    knowledge_ids = torch.randint(0, num_knowledge, (batch_size, num_knowledge))
    features = torch.randn(batch_size, embedding_dim)
    output = neuralcdm(student_ids, question_ids, knowledge_ids, features)
    print(f"输出形状：{output.shape}")
    
    print("\n测试 OpenCDM...")
    opencdm = OpenCDM(num_students, num_questions, num_knowledge, embedding_dim)
    knowledge_mask = torch.ones(batch_size, num_knowledge)
    output = opencdm(student_ids, question_ids, knowledge_mask, features)
    print(f"输出形状：{output.shape}")
    
    print("\n测试 GCD...")
    gcd = GCD(num_students, num_questions, num_knowledge, embedding_dim)
    knowledge_adj = torch.eye(num_knowledge)
    output = gcd(student_ids, question_ids, knowledge_adj, features)
    print(f"输出形状：{output.shape}")
    
    print("\n测试 SCD...")
    scd = SCD(num_students, num_questions, num_knowledge, embedding_dim)
    question_seqs = torch.randint(0, num_questions, (batch_size, 5))
    response_seqs = torch.randint(0, 2, (batch_size, 5))
    output = scd(student_ids, question_seqs, response_seqs, features)
    print(f"输出形状：{output.shape}")
    
    print("\n测试 RDGT...")
    rdgt = RDGT(num_students, num_questions, num_knowledge, embedding_dim)
    knowledge_graph = torch.eye(num_knowledge).unsqueeze(0).expand(batch_size, -1, -1)
    output = rdgt(student_ids, question_ids, knowledge_graph, features)
    print(f"输出形状：{output.shape}")
