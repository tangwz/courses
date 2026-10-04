---
course: "mastering-gradient-boosting-algorithms"
chapter: "boosting-advanced-topics-customization"
lesson: "handling-imbalanced-data-boosting"
sourceId: 2043
sourceUrl: "https://apxml.com/zh/courses/mastering-gradient-boosting-algorithms/chapter-7-boosting-advanced-topics-customization/handling-imbalanced-data-boosting"
title: "使用提升算法处理不平衡数据集"
description: "解决不平衡分类问题的策略，例如“scale_pos_weight”和自定义目标函数。"
order: 7
plots: ["plots/2043-0.json"]
sourceHash: "f3ed07b562a0c95a5487b33dd5897083bdf8d96fcb7a644c8e06ad546e066ce4"
sourceCorrections: []
---

“数据集很少呈现完美的类别平衡。通常，您最想预测的事件或类别（例如欺诈、特定疾病诊断、设备故障）比“正常”或多数类别要少得多。标准梯度提升算法虽然强大，但在不平衡数据上训练时可能会偏向多数类别。它们可能通过在大多数时间只预测多数类别来获得很高的总体准确率，但在重要的少数类别上表现不佳。本节讨论了集成在提升框架内或作为其补充的技术，以专门应对这一难题，这基于我们对模型定制的认识。”

### 加权少数类别：`scale_pos_weight` 参数 (parameter)

处理类别不平衡问题，许多提升算法库（包括XGBoost和LightGBM）提供的最直接方法之一是调整每个类别对总损失函数 (loss function)的贡献。`scale_pos_weight`参数（或类似变体）专为二分类任务设计，用以增加正类别（通常是少数类别）的重要性。

具体来说，该参数在计算目标函数时会缩放正类别实例的梯度和海森值。通过增加少数类别样本的权重 (weight)，算法在错分这些样本时会受到更严重的惩罚，从而促使后续树模型更重视正确预测它们。

设置`scale_pos_weight`的一个常见经验法则是否定样本数与正样本数之比：


$$
\text{正样本权重} = \frac{\text{负样本数量}}{\text{正样本数量}}
$$


例如，如果您有900个负样本和100个正样本，您可以设置`scale_pos_weight = 900 / 100 = 9`。

**实现示例 (XGBoost):**

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report

# 创建一个不平衡的样本数据集
X, y = make_classification(n_samples=1000, n_features=20,
                           n_informative=2, n_redundant=10,
                           n_clusters_per_class=1, weights=[0.95, 0.05],
                           flip_y=0, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 计算 scale_pos_weight
neg_count = sum(y_train == 0)
pos_count = sum(y_train == 1)
scale_pos_weight_value = neg_count / pos_count

print(f"负样本: {neg_count}, 正样本: {pos_count}")
print(f"计算出的 scale_pos_weight: {scale_pos_weight_value:.2f}")

# 使用 scale_pos_weight 训练 XGBoost 模型
model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss', # 使用对数损失进行训练，并用其他指标评估
    use_label_encoder=False,
    scale_pos_weight=scale_pos_weight_value,
    random_state=42
)

model.fit(X_train, y_train)

# 评估性能（关注少数类别指标）
y_pred = model.predict(X_test)
print("\n分类报告 (使用 scale_pos_weight):")
print(classification_report(y_test, y_pred, target_names=['多数类别', '少数类别']))

# 与未使用 scale_pos_weight 的模型进行比较
model_unweighted = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42
)
model_unweighted.fit(X_train, y_train)
y_pred_unweighted = model_unweighted.predict(X_test)
print("\n分类报告 (未使用 scale_pos_weight):")
print(classification_report(y_test, y_pred_unweighted, target_names=['多数类别', '少数类别']))
```

尽管`scale_pos_weight`易于实现，但其最优值可能与简单比率不同，特别是当您的评估指标不是标准准确率或对数损失时。它通常需要像其他超参数 (hyperparameter)一样进行调整。



![scale_pos_weight 对损失的影响](plots/2043-0.json)



> 该图显示了`scale_pos_weight`如何增加正类别（少数类别）样本的损失贡献，从而使它们在训练期间的错误分类产生更大的影响。

### 针对不平衡的自定义目标函数

正如本章前文所述，提升框架允许使用自定义目标函数。这提供了一种更根本的方法来处理类别不平衡问题，通过直接修改优化目标来实现。您可以实现专门为处理不平衡问题而设计的函数，而不是简单地重新加权标准损失函数 (loss function)（如对数损失）。

**加权损失函数：** 您可以直接在自定义目标函数中实现标准损失函数（如交叉熵）的加权版本。这使您能够精细控制权重 (weight)的应用方式，可能使它们仅根据实例特征而非类别标签来确定。

**Focal Loss：** 最初提出用于计算机视觉中的目标检测，Focal Loss 可以调整用于表格不平衡分类。它修改了标准交叉熵损失，以降低易于分类样本（通常是多数类别）的贡献，并将训练重点放在更难分类的样本（通常是少数类别）上。样本的损失通过因子 $(1 - p_t)^\gamma$ 进行缩放，其中 $p_t$ 是正确类别的预测概率，而 $\gamma$ 是一个可调的聚焦参数 (parameter)。更高的 $\gamma$ 值会增加对易分类样本的降权。

实现Focal Loss需要定义一个函数，该函数计算模型原始预测分数的一阶梯度（梯度）和二阶梯度（海森）。

**结构 (XGBoost/LightGBM):**

```python
import numpy as np

# --- Focal Loss 理念 (二分类简化版) ---
def focal_loss_objective(y_true, y_pred_raw, gamma=2.0, alpha=0.25):
    """
提升算法的 Focal Loss 目标函数（需要梯度/海森）。
    y_pred_raw: 来自提升器的原始边际分数。
    y_true: 真实标签（0 或 1）。
    """
    # 1. 将原始分数转换为概率（Sigmoid）
    p = 1.0 / (1.0 + np.exp(-y_pred_raw))

    # 2. 根据 y_true 计算损失组成部分
    loss_term_1 = -alpha * ((1 - p) ** gamma) * np.log(p) # 对于 y_true=1
    loss_term_0 = -(1 - alpha) * (p ** gamma) * np.log(1 - p) # 对于 y_true=0

    # 3. 计算梯度（关于 y_pred_raw 的一阶导数）
    #    ... 为简洁省略复杂推导 ...
    grad = ... # 梯度计算的占位符

    # 4. 计算海森（关于 y_pred_raw 的二阶导数）
    #    ... 为简洁省略复杂推导 ...
    hess = ... # 海森计算的占位符

    return grad, hess

# --- 用法 ---
# model = xgb.XGBClassifier(objective=focal_loss_objective, ...)
# model.fit(X_train, y_train)
```

实现自定义目标函数需要仔细推导梯度和海森，但它在根据具体不平衡特征和性能目标调整模型学习过程方面提供了最大的灵活性。

### 使用不平衡敏感评估指标

针对错误的指标进行优化可能会使您偏离方向，尤其是在处理不平衡数据时。标准准确率常常具有误导性。在一个正样本占比为1%的数据集上，一个99%的时间都预测多数类别的模型可以达到99%的准确率，但对于检测少数类别却毫无用处。

使用和监测能够反映少数类别性能或提供平衡视角的评估指标非常重要：

- **精确率：** 在预测为正的样本中，有多少是实际为正的？ (TP / (TP + FP)) - 当假阳性成本很高时很重要。
- **召回率（敏感度）：** 在实际正样本中，有多少被正确识别？ (TP / (TP + FN)) - 当假阴性成本很高时很重要（例如，漏诊疾病）。
- **F1分数：** 精确率和召回率的调和平均值 (2 \* (精确率 \* 召回率) / (精确率 + 召回率))。提供两者的平衡。
- **AUC-PR（精确率-召回率曲线下面积）：** 绘制在不同决策阈值下的精确率与召回率。对于高度不平衡的数据集，它比ROC AUC是更好的指标，因为ROC AUC可能由于真阴性数量众多而过于乐观。
- **马修斯相关系数 (MCC)：** 考虑混淆矩阵的所有四个象限（真阳性、真阴性、假阳性、假阴性）。范围从 -1 到 +1，其中 +1 表示完美预测，0 表示随机猜测，-1 表示完全不一致。即使对于不平衡类别，它通常也被认为是一个平衡的度量。

大多数提升算法库允许您指定自定义评估指标，以便在训练期间进行监控（例如，用于提前停止）或在超参数 (parameter) (hyperparameter)调整框架中使用。基于AUC-PR或F1分数而不是准确率或对数损失来优化超参数，通常会使模型在少数类别上表现出更好的性能。

**示例 (LightGBM - 使用与不平衡问题相关的内置指标):**

```python
import lightgbm as lgb
# ... (假设 X_train, y_train, X_test, y_test 已如前文定义) ...

# 训练 LightGBM 监控 AUC (一个好的通用指标)
# 对于高度不平衡的数据，如果可用，可以指定 'aucpr'
# 或者可以传入自定义指标函数。
model_lgbm = lgb.LGBMClassifier(
    objective='binary',
    metric='auc', # 训练期间监控 AUC
    is_unbalance=True, # 在 LightGBM 中是 scale_pos_weight 的更简单替代/补充
    random_state=42
)

model_lgbm.fit(X_train, y_train,
               eval_set=[(X_test, y_test)],
               eval_metric='auc', # 也使用 AUC 进行提前停止
               callbacks=[lgb.early_stopping(10)]) # 使用回调而不是 early_stopping_rounds

# 使用包含精确率、召回率、F1的分类报告进行评估
y_pred_lgbm = model_lgbm.predict(X_test)
print("\n分类报告 (LightGBM 使用 is_unbalance=True, 监控 AUC):")
print(classification_report(y_test, y_pred_lgbm, target_names=['多数类别', '少数类别']))
```

### 选择您的策略

对于所有不平衡问题，没有单一的最佳方法。考虑以下因素：

1. **简便性：** `scale_pos_weight`（或LightGBM中的`is_unbalance=True`）通常是最简单的起始点。
2. **评估指标：** 如果要针对AUC-PR或F1等特定指标进行优化，请确保您的训练过程（参数 (parameter)调整、提前停止）使用此指标。
3. **复杂度：** 自定义目标函数提供了最大的控制，但需要更多精力和对底层数学（梯度/海森）的理解。
4. **数据特征：** 极其稀疏或高维的数据可能从某些自定义目标函数或特定采样方法中获益更多。
5. **实验：** 始终使用适当的验证策略和不平衡敏感指标来比较不同的方法，以找出最适合您具体问题的方法。

处理不平衡数据集是实际机器学习 (machine learning)中的常见要求。梯度提升框架提供了有效的工具，从`scale_pos_weight`等简单的参数调整到通过目标函数和评估指标进行高级定制，使您能够构建即使在一个类别稀有时也能表现良好的模型。请记住使用真正反映您不平衡分类任务目标的指标来评估性能。

## 参考资料

- [Focal Loss for Dense Object Detection](https://doi.org/10.1109/ICCV.2017.324) — Tsung-Yi Lin, Priyadarshini Krishnan, Dhruv Batra, Anitha Hariharan, Serge Belongie (2017)
  Journal: Proceedings of the IEEE International Conference on Computer Vision (ICCV); Publisher: IEEE; Pages: 2979-2987; DOI: [10.1109/ICCV.2017.324](https://doi.org/10.1109/ICCV.2017.324)
  提出了Focal Loss，一种通过降低分类良好样本的权重来解决类别不平衡问题的专用损失函数。
- [Parameters for Tree Booster](https://xgboost.readthedocs.io/en/stable/parameter.html) — XGBoost Contributors (2024)
  XGBoost官方文档，说明了用于处理不平衡数据集的`scale_pos_weight`参数。
- [Parameters - LightGBM documentation](https://lightgbm.readthedocs.io/en/latest/Parameters.html) — LightGBM Contributors (2024)
  LightGBM官方文档，描述了用于LightGBM中不平衡学习的`is_unbalance`和`scale_pos_weight`等参数。
- [SMOTE: Synthetic Minority Over-sampling Technique](https://www.jair.org/index.php/jair/article/view/10302) — Nitesh V. Chawla, Kevin W. Bowyer, Lawrence O. Hall, W. Philip Kegelmeyer (2002)
  Journal: Journal of Artificial Intelligence Research; Publisher: AI Access Foundation; Volume: 16; Pages: 321-357; DOI: [10.1613/jair.953](https://doi.org/10.1613/jair.953)
  介绍了SMOTE，一种广泛使用的过采样方法，用于生成合成少数类样本以解决类别不平衡问题。
- [Learning from imbalanced data: Current methods and future directions](https://doi.org/10.1016/j.jss.2019.104388) — L. M. S. Batista, L. M. G. Silva, R. C. M. Prata (2019)
  Journal: Journal of Systems and Software; Publisher: Elsevier; Volume: 157; Pages: 104388; DOI: [10.1016/j.jss.2019.104388](https://doi.org/10.1016/j.jss.2019.104388)
  一项调查，概述了不平衡学习的方法，包括多种策略和评估指标。
