# Gemma3-CDM: 基于大模型的认知诊断框架

## 项目概述
本项目实现了一个基于本地大模型（gemma3:4b）的认知诊断系统，支持四个数据集上的实验，包含三种对齐方式和五种认知诊断模型的对比实验。

## 功能特性

### 1. 数据集支持
- **math23k**: 数学题目数据集
- **open-luna**: 开放学习数据集
- **pisa2015**: PISA 2015 评估数据
- **ktbd-ednet**: 知识追踪基准数据集

### 2. 大模型处理
- 调用本地 ollama 服务 (gemma3:4b)
- 生成文本诊断报告

### 3. 嵌入与对齐
- 嵌入模型生成向量表示
- 三种对齐方式：
  - 交互重建对齐 (Interactive Reconstruction Alignment)
  - 混合对比对齐 (Hybrid Contrastive Alignment)
  - 无对齐 (No Alignment)

### 4. 认知诊断模型
- NeuralCDM
- OpenCDM
- GCD (Graph Cognitive Diagnosis)
- SCD (Sequential Cognitive Diagnosis)
- RDGT (Relation-aware Diagnostic Graph Transformer)

## 实验配置
共 15 个对比实验 (3 种对齐方式 × 5 个模型)

## 目录结构
```
/workspace
├── data/                    # 数据存储
├── models/                  # 模型定义
├── alignment/               # 对齐方法
├── utils/                   # 工具函数
├── configs/                 # 配置文件
├── scripts/                 # 运行脚本
└── results/                 # 实验结果
```

## 安装依赖
```bash
pip install -r requirements.txt
```

## 使用方法
```bash
# 下载数据集
python scripts/download_datasets.py

# 生成诊断报告
python scripts/generate_reports.py

# 运行对比实验
python scripts/run_experiments.py --alignment interactive --model neuralcdm
```

## 许可证
MIT License
