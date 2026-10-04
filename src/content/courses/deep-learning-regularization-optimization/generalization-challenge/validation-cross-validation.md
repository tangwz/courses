---
course: "deep-learning-regularization-optimization"
chapter: "generalization-challenge"
lesson: "validation-cross-validation"
sourceId: 4881
sourceUrl: "https://apxml.com/zh/courses/deep-learning-regularization-optimization/chapter-1-generalization-challenge/validation-cross-validation"
title: "验证与交叉验证策略"
description: "涵盖模型验证的策略，包括交叉验证技术。"
order: 5
plots: []
sourceHash: "5263fa6724e4c7295f0c9f3ac7eb82e4bf4544f76413d7989bae35a3db23674a"
sourceCorrections: []
---

正如我们在学习曲线中看到的那样，仅仅最小化训练数据上的损失并不能保证在未见过的数据上表现良好。我们需要可靠的途径来估算模型在部署*前*的泛化能力。这需要仔细划分数据，以模拟遇到新示例的情形。

### 留出验证集

深度学习 (deep learning)中，尤其是在大型数据集上，最常见的策略是将可用数据分为三个不同的数据集：

1. **训练集：** 这是数据中最大的一部分，专门用于通过反向传播 (backpropagation)训练模型参数 (parameter)（权重 (weight)和偏置 (bias)）。模型从这些数据中学习其潜在的规律。
2. **验证集（或开发集）：** 该数据集在训练和开发阶段使用，用于评估模型在未直接训练过的数据上的表现。其主要目的是指导关于超参数 (hyperparameter)（如学习率、正则化 (regularization)强度、网络结构选择）和模型选择的决定。你可以训练几种模型变体或调整超参数，而验证集会帮助你选择表现最好的模型，而*不*触及最终的测试集。监测验证集与训练集上的表现（如学习曲线所示）对于诊断过拟合 (overfitting)或欠拟合 (underfitting)很重要。
3. **测试集：** 该数据集被预留并*仅使用一次*，在所有训练和超参数调整完成后。它提供了模型在真正未见过数据上泛化表现的最终、无偏估计。在模型开发过程中避免将测试集用于任何决策很重要；否则，你可能会不经意地对测试集过拟合，导致对性能的估计过于乐观。

典型的划分比例可能是训练集占70%，验证集占15%，测试集占15%，但这些百分比会根据数据集的总大小而显著不同。对于非常大型的数据集（数百万个示例），验证集和测试集所占的比例可能小得多（例如，各占1%），但其绝对数量仍足以提供可靠的评估。

留出法的主要优点是其简单性和计算效率。然而，在验证集上获得的性能估计可能对特定的随机划分敏感，特别是当数据集不是很大时。“幸运”或“不幸运”的划分可能会对模型真实能力产生误导性印象。

### 交叉验证（CV）

K折交叉验证是一种有用的方法，尤其在数据稀缺或需要可靠泛化性能评估时。它通过使用多次数据划分来减少评估结果的方差，从而提供模型性能的更稳定估计。

其工作原理如下：

1. **划分：** *原始训练数据*（不包括最终的测试集，测试集保持独立）被随机划分为 *k* 个大小相等的折（子集）。*k* 的常见取值是 5 或 10。
2. **迭代：** 模型训练 *k* 次。在每次迭代 *i* 中（从 1 到 k）：
   - 折 *i* 被作为该次迭代的验证集。
   - 其余的 *k-1* 个折被组合起来作为训练集。
3. **评估：** 每次训练运行后，模型在该次迭代中使用的验证折上的性能（例如，准确率、损失）被计算。
4. **平均：** 最终的性能评估是所有 *k* 次迭代中获得的性能指标的平均值。

> K折交叉验证图示。训练数据被分成 k 个折。在每次迭代中，一个折作为验证集（红色），而其他折用于训练（蓝色）。每次迭代的性能指标被平均。

**K折交叉验证的优点：**

- **更可靠的评估：** 通过对 *k* 次运行求平均，性能评估对单一随机划分的依赖性较小。
- **数据使用效率高：** 每个数据点都恰好被用于验证一次，并被用于训练 *k-1* 次。

**K折交叉验证的缺点：**

- **计算成本：** 训练模型 *k* 次比训练一次要昂贵得多，对于需要数天或数周训练的非常大型的深度学习 (deep learning)模型来说，这可能难以承受。

**分层K折：** 在处理分类问题时，特别是当类别不平衡时，重要的是每个折都能保留与完整数据集近似的每个类别的样本百分比。**分层K折交叉验证**确保这种分层，从而得到更可靠的评估。

尽管由于成本原因，K折交叉验证在深度学习中较少用于*最终模型训练*，但它是一种非常有用的方法，可用于：

- 在较小数据集上评估不同的模型结构或特征集。
- 在研究中比较不同方法时，获得可靠的性能评估。
- 超参数 (parameter) (hyperparameter)调整，尽管在 *k* 个折上调整会进一步增加计算成本。

### 实际数据划分

无论是使用简单的留出划分还是准备交叉验证，都需要仔细实现。像 Scikit-learn 这样的库提供了方便的功能。

```python
import torch
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np

# 假设 X_data 包含你的特征，y_data 包含你的标签（作为 NumPy 数组或张量）
# 示例模拟数据
X_data = np.random.rand(1000, 20) # 1000 samples, 20 features
y_data = np.random.randint(0, 2, 1000) # 1000 binary labels

# --- 留出划分 ---
# 首先，划分为训练+验证集和测试集
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_data, y_data, test_size=0.15, random_state=42, stratify=y_data # 分类问题使用 stratify 参数进行分层
)

# 然后，将训练+验证集划分为实际的训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.1765, # 大约是原始数据的 15% (0.15 / (1-0.15))
    random_state=42, stratify=y_train_val # 保持分层
)

print(f"留出划分大小：训练集={len(X_train)}, 验证集={len(X_val)}, 测试集={len(X_test)}")
# 输出：留出划分大小：训练集=700, 验证集=150, 测试集=150 (近似比例)

# --- K折交叉验证设置（使用分层K折）---
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

print(f"\n{n_splits}折交叉验证索引（使用原始训练+验证数据）：")
# 注意：这里我们使用 X_train_val，因为测试集总是保持独立。
fold_num = 1
for train_index, val_index in skf.split(X_train_val, y_train_val):
    print(f"折 {fold_num}:")
    print(f"  训练样本: {len(train_index)}, 验证样本: {len(val_index)}")
    # 在真实的交叉验证循环中，你会使用这些索引来选择数据：
    # X_train_fold, X_val_fold = X_train_val[train_index], X_train_val[val_index]
    # y_train_fold, y_val_fold = y_train_val[train_index], y_train_val[val_index]
    # ... 然后在这些折特定的数据集上训练和评估模型
    fold_num += 1

# 如果训练需要，转换为 PyTorch 张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long) # 假设是分类标签
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)
```

> 使用 Scikit-learn 创建留出划分并生成 K折交叉验证的索引。`random_state` 确保可复现性，`stratify` 保持类别比例。

请记住：

- 在划分数据之前，务必打乱数据（除非是时间序列数据，其中时间顺序很重要）。大多数划分函数在 `shuffle=True` 时默认会这样做。
- 对于分类任务使用分层，以确保类别分布在不同划分中得到保留。
- 测试集在项目结束之前始终不被触及。所有模型开发、调整和选择都使用训练集和验证集（或在训练数据上进行交叉验证）进行。

选择正确的验证策略很重要。它使我们能够可靠地评估泛化性能，诊断过拟合 (overfitting)等问题，并在模型开发过程中做出明智的决策，最终产生在实际遇到的数据上表现良好的模型。

## 参考资料

- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本全面涵盖深度学习的书籍，详细解释了泛化、验证策略和过拟合。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems, 3rd Edition](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125974/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  提供使用Python库实现机器学习技术的实践指导，包括数据划分和交叉验证。
- [Cross-validation: evaluating estimator performance](https://scikit-learn.org/stable/modules/cross_validation.html) — scikit-learn developers (2024)
  Publisher: scikit-learn
  官方文档，解释了scikit-learn库中各种交叉验证技术和数据划分工具。
- [Structuring Machine Learning Projects (Deep Learning Specialization, Course 3)](https://www.coursera.org/learn/machine-learning-project-strategy) — Andrew Ng (2017)
  Publisher: Coursera / deeplearning.ai
  该课程讨论了构建机器学习项目的实践策略，包括有效使用训练集、验证集和测试集。
