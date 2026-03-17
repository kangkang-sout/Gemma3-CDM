# RDGT 模型 Dropout 敏感性分析实验指南

## 实验目的

对比 RDGT（Relation-aware Diagnostic Graph Transformer）模型在三种对齐方式下，
在四个数据集上 AUC 和 ACC 随 dropout 值变化的趋势。

## 实验设计

### 1. 实验变量

**自变量：**
- Dropout 值：[0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
- 对齐方法：不对齐 (none)、交互重建对齐 (interactive)、混合对比对齐 (hybrid)
- 数据集：math23k、open-luna、pisa2015、ktbd-ednet

**因变量：**
- AUC（ROC 曲线下面积）
- ACC（准确率）

**控制变量：**
- 模型架构：RDGT（固定参数）
- 训练轮数：10 epochs
- 批大小：32
- 学习率：0.001

### 2. 实验总数

3 种对齐方式 × 6 个 dropout 值 × 4 个数据集 = **72 个实验**

## 文件结构

```
/workspace/
├── scripts/
│   ├── run_rdgt_dropout_analysis.py    # 完整实验脚本（使用真实数据集）
│   └── test_rdgt_dropout.py            # 轻量级测试脚本（使用模拟数据）
├── alignment/
│   └── methods.py                       # 对齐方法实现
├── models/cdm/
│   └── models.py                        # RDGT 模型实现
└── results/dropout_analysis/
    ├── math23k_dropout_curves.png       # math23k 数据集结果图
    ├── open-luna_dropout_curves.png     # open-luna 数据集结果图
    ├── pisa2015_dropout_curves.png      # pisa2015 数据集结果图
    ├── ktbd-ednet_dropout_curves.png    # ktbd-ednet 数据集结果图
    ├── all_datasets_comparison.png      # 所有数据集对比图
    └── dropout_results.json             # 原始结果数据
```

## 使用方法

### 方法一：运行完整实验（需要真实数据集）

```bash
# 在单个数据集上运行
python scripts/run_rdgt_dropout_analysis.py --dataset math23k --epochs 10

# 在所有四个数据集上运行
python scripts/run_rdgt_dropout_analysis.py --dataset all --epochs 10

# 自定义 dropout 值
python scripts/run_rdgt_dropout_analysis.py --dataset math23k --dropouts "0.0,0.1,0.2,0.3"

# 自定义对齐方法
python scripts/run_rdgt_dropout_analysis.py --dataset math23k --alignments "none,interactive"
```

### 方法二：运行轻量级测试（使用模拟数据）

```bash
# 快速验证代码逻辑（约 2-5 分钟）
python scripts/test_rdgt_dropout.py
```

## 输出说明

### 1. 控制台输出

实验运行时会在控制台显示进度和结果表格：

```
============================================================
RDGT Dropout 敏感性分析
============================================================

对齐方法：none
----------------------------------------
  Dropout=0.0: AUC=0.xxxx, ACC=0.xxxx
  Dropout=0.1: AUC=0.xxxx, ACC=0.xxxx
  ...

============================================================
RDGT Dropout 敏感性分析结果汇总
============================================================
对齐方法         | Dropout    | 样本数        | AUC          | ACC         
--------------------------------------------------------------------------------
不对齐           | 0.0        | xxx        | 0.xxxx       | 0.xxxx      
不对齐           | 0.1        | xxx        | 0.xxxx       | 0.xxxx      
...
```

### 2. 可视化图表

生成两类图表：

**单数据集双指标图：**
- `math23k_dropout_curves.png`：包含 AUC 和 ACC 两个子图
- X 轴：Dropout Rate (0.0-0.5)
- Y 轴：AUC 或 Accuracy
- 三条曲线分别代表三种对齐方式

**多数据集对比图：**
- `all_datasets_comparison.png`：2×2 网格布局
- 每个子图展示一个数据集的 AUC 曲线对比

### 3. JSON 结果文件

`dropout_results.json` 包含所有原始数据：

```json
[
  {
    "dataset": "math23k",
    "alignment": "none",
    "dropout": 0.1,
    "train_samples": 800,
    "test_samples": 200,
    "best_auc": 0.7523,
    "best_accuracy": 0.7150
  },
  ...
]
```

## 核心代码说明

### 1. RDGT 模型

位于 `models/cdm/models.py`，关键特性：
- Transformer 编码器层处理知识图谱
- 关系感知注意力机制
- 可配置的 dropout 层

### 2. 对齐方法

位于 `alignment/methods.py`：

**不对齐 (NoAlignment)：**
```python
aligned_embeddings = (report_embeddings + knowledge_embeddings) / 2
```

**交互重建对齐 (InteractiveReconstructionAlignment)：**
- 交叉注意力机制
- 门控融合
- 重建损失

**混合对比对齐 (HybridContrastiveAlignment)：**
- 实例级对比损失 (InfoNCE)
- 类别级对比损失
- 投影头和预测头

### 3. Dropout 控制

训练时动态设置 dropout：
```python
for module in model.modules():
    if isinstance(module, nn.Dropout):
        module.p = dropout_rate  # 训练时设置
```

评估时关闭 dropout：
```python
for module in model.modules():
    if isinstance(module, nn.Dropout):
        module.p = 0.0  # 评估时关闭
```

## 预期结果分析

### 可能的趋势

1. **Dropout 与性能的关系：**
   - Dropout=0.0：可能过拟合，测试集性能较低
   - Dropout=0.1-0.3：通常最佳范围，平衡欠拟合和过拟合
   - Dropout>0.4：可能欠拟合，性能下降

2. **对齐方法的影响：**
   - 不对齐：基线性能
   - 交互重建对齐：可能在数据量大的数据集上表现更好
   - 混合对比对齐：可能在稀疏数据上更有优势

3. **数据集差异：**
   - math23k：数学题目，知识点结构清晰
   - open-luna：开放题目，可能需要更强的正则化
   - pisa2015：国际评估数据，质量较高
   - ktbd-ednet：在线学习数据，可能较稀疏

## 故障排除

### 问题 1：CUDA 内存不足

```bash
# 减小批大小
python scripts/run_rdgt_dropout_analysis.py --batch-size 16

# 减少训练轮数
python scripts/run_rdgt_dropout_analysis.py --epochs 5
```

### 问题 2：数据集不存在

```bash
# 先下载数据集
python scripts/download_datasets.py
```

### 问题 3：导入错误

确保已安装依赖：
```bash
pip install -r requirements.txt
```

## 扩展实验

### 添加更多 Dropout 值

```bash
python scripts/run_rdgt_dropout_analysis.py \
  --dropouts "0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5"
```

### 仅测试特定对齐方法

```bash
python scripts/run_rdgt_dropout_analysis.py \
  --alignments "interactive,hybrid"
```

### 保存中间结果

修改 `run_dropout_sweep` 函数，在每次实验后保存：
```python
import json
with open('intermediate_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)
```

## 引用

如果此代码对您的研究有帮助，请引用相关论文：
- RDGT 模型原始论文
- 认知诊断相关研究
- 对比学习对齐方法
