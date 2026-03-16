# 数据稀疏/稠密场景对比实验

## 实验设计

### 场景定义
- **数据稀疏场景**：练习题在训练集中交互少于 3 次
- **数据稠密场景**：练习题在训练集中交互多于 10 次

### 对比模型
1. **NeuralCDM** - 神经认知诊断模型
2. **OpenCDM** - 开放认知诊断模型
3. **GCD** - 图认知诊断
4. **SCD** - 序列认知诊断
5. **RDGT** - 关系感知诊断图 Transformer

### 评估指标
- **AUC** (Area Under Curve) - ROC 曲线下面积
- **ACC** (Accuracy) - 准确率

### 实验数量
5 个模型 × 2 个场景 = **10 个对比实验**

## 使用方法

### 运行完整实验
```bash
# 在单个数据集上运行
python scripts/run_scenario_comparison.py --dataset math23k --epochs 10 --batch_size 32

# 在所有四个数据集上运行
python scripts/run_scenario_comparison.py --dataset all --epochs 10 --batch_size 32
```

### 运行轻量级测试
```bash
# 快速验证代码逻辑（使用模拟数据）
python scripts/test_scenario_comparison.py
```

## 输出结果

实验完成后，结果将保存在 `./results/scenario_comparison/` 目录下：
- `{dataset}_scenario_comparison.json` - 单个数据集的详细结果
- `all_scenario_comparison.json` - 所有数据集的汇总结果

## 核心功能

### 1. 场景数据集类 (`SparseDenseDataset`)
根据题目交互次数过滤数据，支持稀疏/稠密两种场景。

### 2. 交互次数计算
自动统计每个题目在训练集中的交互次数。

### 3. 对比分析
自动生成模型在两种场景下的性能对比表格和差异分析。

## 预期发现

基于认知诊断理论，预期：
1. 所有模型在稠密场景下的 AUC 和 ACC 应显著高于稀疏场景
2. RDGT 等复杂模型可能在稠密场景下优势更明显
3. 稀疏场景下简单模型可能更稳健

## 文件列表

- `scripts/run_scenario_comparison.py` - 完整实验脚本
- `scripts/test_scenario_comparison.py` - 轻量级测试脚本
- `alignment/methods.py` - 对齐方法模块（已修复 NoAlignment 参数问题）
- `models/cdm/models.py` - 认知诊断模型模块（已修复 NeuralCDM 输入问题）
