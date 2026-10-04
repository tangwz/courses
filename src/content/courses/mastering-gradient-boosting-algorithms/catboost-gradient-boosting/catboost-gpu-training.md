---
course: "mastering-gradient-boosting-algorithms"
chapter: "catboost-gradient-boosting"
lesson: "catboost-gpu-training"
sourceId: 2017
sourceUrl: "https://apxml.com/zh/courses/mastering-gradient-boosting-algorithms/chapter-6-catboost-gradient-boosting/catboost-gpu-training"
title: "GPU训练加速"
description: "CatBoost高效GPU实现的概述。"
order: 6
plots: []
sourceHash: "007f5564674fd88242deab48d46742b9f145b11ba986ac7aa011ebf399842aed"
sourceCorrections: []
---

训练复杂的梯度提升模型，尤其是在大型数据集上，或在生成大量特征组合时（如CatBoost所能做到），通常会变得计算密集。顺序构建数千棵树，每棵树都需要评估跨特征和样本的潜在分裂点，这要求很高的处理能力。虽然基于CPU的并行化提供了一些加速，但图形处理单元（GPU）提供了一种不同的架构，一种特别适合提升过程部分固有的海量并行计算的架构。

CatBoost包含了一个高度优化的实现在支持CUDA的NVIDIA GPU上进行训练。与基于CPU的训练相比，这能实现大幅加速，通常将训练时间从数小时缩短到数分钟，特别是对于具有数十万或数百万个样本和众多特征的数据集。

### 运用GPU架构

GPU擅长单指令多数据（SIMD）操作。与CPU的少量复杂核心相比，它们包含数千个更简单的核心。这种架构非常适合需要同时对许多不同数据点执行相同操作的任务。

这如何应用于CatBoost？

1. **直方图构建：** 类似于LightGBM和XGBoost的近似算法，CatBoost可以运用基于直方图的方法来查找分裂点，尤其是在GPU上。计算直方图（特征值分布、梯度和、Hessian和）可以在GPU上有效地在数据点之间并行进行。
2. **对称树（Oblivious Trees）：** 对称树的结构对GPU计算尤其有利。由于同一深度层的所有节点都使用相同的分裂标准（特征和阈值），因此可以对该层的所有节点同时进行高度并行化的样本左右分支判断。这与传统的按层或按叶增长的树形成对比，在那些树中，同一深度的不同节点可能测试不同的特征，导致计算不那么统一。对称树的可预测结构可以非常高效地映射到GPU的并行处理能力上。

> CatBoost中的对称树在每个深度层都使用相同的特征和阈值进行分裂。这种一致性使得GPU能够高效地并行处理给定层的所有节点。

3. **类别特征处理：** 尽管有序TS的核心逻辑包含可能与并行化看似不符的顺序依赖，但当对大量样本或特征进行操作时，其底层计算（如统计数据计算和置换应用）仍可受益于GPU加速。组合生成和对称树结构进一步促进了并行执行。

### 在CatBoost中启用GPU训练

在CatBoost中使用GPU加速很简单。这需要安装支持GPU的CatBoost库版本，并拥有兼容的NVIDIA硬件以及已安装的CUDA驱动程序。代码中的主要修改是在初始化模型时，将`task_type`参数 (parameter)设置为`'GPU'`。

```python
import catboost as cb
import pandas as pd
from sklearn.model_selection import train_test_split

# 假设X（特征）和y（目标）已加载为Pandas DataFrame/Series
# 如果需要，请确保正确识别类别特征索引

# 识别类别特征（示例）
categorical_features_indices = [i for i, col in enumerate(X.columns) if X[col].dtype == 'object' or X[col].dtype.name == 'category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化CatBoostClassifier进行GPU训练
model_gpu = cb.CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3,
    loss_function='Logloss',
    eval_metric='AUC',
    task_type='GPU',  # 指定GPU训练
    devices='0',      # 可选：指定GPU设备ID
    random_seed=42,
    verbose=100,      # 每100次迭代打印一次进度
    early_stopping_rounds=50
)

# 在GPU上训练模型
model_gpu.fit(
    X_train, y_train,
    cat_features=categorical_features_indices,
    eval_set=(X_test, y_test),
    plot=False # 将plot设置为True可在交互式环境中查看学习曲线
)

# 后续可进行预测和评估
# preds_gpu = model_gpu.predict_proba(X_test)[:, 1]
```

GPU训练的重要参数：

- `task_type='GPU'`：这是在GPU上启用训练的必要参数。
- `devices`：一个字符串，指定要使用的GPU设备ID（例如，`'0'`、`'0:1'`、`'1'`）。如果省略，CatBoost通常使用默认GPU（设备0）。

### GPU使用注意事项

- **硬件：** 您需要一块具有足够显存（VRAM）和计算能力的NVIDIA GPU。大型数据集和复杂模型需要更多VRAM。
- **数据传输：** 在主系统内存（RAM）和GPU显存（VRAM）之间传输数据存在开销。对于非常小的数据集，这种开销可能会抵消计算加速带来的好处。GPU加速对于中大型数据集效益最大，因为在这种情况下，计算时间远超数据传输时间。
- **参数 (parameter)调优：** 某些超参数 (hyperparameter)与GPU训练的交互方式可能与CPU训练不同，尽管核心算法保持不变。通常建议使用与最终模型相同的`task_type`进行超参数调优。
- **数值精度：** CPU和GPU之间浮点运算的微小差异有时可能导致结果或收敛路径略有不同，尽管CatBoost力求一致性。

通过运用GPU加速，CatBoost大大缩短了训练复杂模型所需的时间，特别是对于涉及大量类别特征或大型数据集的模型，这使得迭代开发和超参数调优变得更加可行。

## 参考资料

- [GPU support](https://catboost.ai/docs/concepts/gpu-training.html) — CatBoost Developers ()
  Publisher: Yandex
  CatBoost GPU训练的官方指南，涵盖参数设置和优化考量。
- [CatBoost: Unbiased Boosting with Categorical Features](https://proceedings.neurips.cc/paper/2018/file/6f4922f45568161a8cdf4ad2299f6d23-Paper.pdf) — Liudmila Prokhorenkova, Gleb Gusev, Aleksandr Vorobev, Anna Veronika Dorogush, Andrey Gulin (2018)
  Journal: Advances in Neural Information Processing Systems; Publisher: NIPS Foundation; Volume: 31
  介绍了CatBoost算法，包括其处理分类特征和Oblivious Trees的方式，这些特性非常适合GPU并行化。
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html) — NVIDIA (2023)
  Publisher: NVIDIA
  了解CUDA编程模型和NVIDIA GPU架构的基本指南，对理解GPU如何加速计算至关重要。
- [LightGBM: A Highly Efficient Gradient Boosting Decision Tree](http://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf) — Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, Tie-Yan Liu (2017)
  Journal: Advances in Neural Information Processing Systems; Publisher: Curran Associates, Inc.; Volume: 30; Pages: 3146-3154
  描述了基于直方图的梯度提升，该技术在正文内容中被提及为有利于GPU加速，也是最早支持GPU的高效GBDT实现之一。
