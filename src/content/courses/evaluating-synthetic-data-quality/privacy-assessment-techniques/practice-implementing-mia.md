---
course: "evaluating-synthetic-data-quality"
chapter: "privacy-assessment-techniques"
lesson: "practice-implementing-mia"
sourceId: 5697
sourceUrl: "https://apxml.com/zh/courses/evaluating-synthetic-data-quality/chapter-4-privacy-assessment-techniques/practice-implementing-mia"
title: "实践操作：实现一个基本的成员推断攻击"
description: "代码示例演示基本成员推断攻击。"
order: 6
plots: ["plots/5697-0.json"]
sourceHash: "37a2d01b23131803b6c942715f8f638360cf09d369ccee15d9e4f6fce3ec4070"
sourceCorrections: []
---

成员推断攻击（MIA）的实际实现涉及构建并评估一个简单的分类器，该分类器旨在区分用于生成合成数据的原始训练数据集中的记录与非该数据集中的记录。成功区分这些记录表明可能存在隐私泄露。

### 攻击设置

MIA 的核心在于将问题视为二元分类任务。我们需要一个模型来预测给定数据记录是否存在于生成模型的训练集中。

为了构建这个攻击模型，我们需要三个数据集：

1. `real_data`: 用于训练合成数据生成器的原始数据集。这些是我们的“成员”。
2. `synthetic_data`: 模型生成的数据集，我们希望对其隐私风险进行评估。
3. `real_holdout_data`: 原始数据分布中*未*用于训练生成模型的一部分。这些作为实际的“非成员”。

核心思路是专门使用*真实*数据范围中的标记 (token)样本来训练分类器：将`real_data`标记为成员（例如，类别1），将`real_holdout_data`标记为非成员（例如，类别0）。

### 构建攻击训练数据集

我们将`real_data`和`real_holdout_data`组合起来，形成我们攻击分类器的训练集。

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 假设 real_data, real_holdout_data 是 Pandas DataFrames
# 确保它们具有相同的列

# 为 real_data 分配成员标签 (1)
real_data_labeled = real_data.copy()
real_data_labeled['is_member'] = 1

# 为 real_holdout_data 分配非成员标签 (0)
real_holdout_labeled = real_holdout_data.copy()
real_holdout_labeled['is_member'] = 0

# 组合它们以创建用于 MIA 分类器的数据集
mia_data = pd.concat([real_data_labeled, real_holdout_labeled], ignore_index=True)

# 分离特征 (X) 和目标标签 (y)
X_mia = mia_data.drop(columns=['is_member'])
y_mia = mia_data['is_member']

# 最好将这些数据用于训练和测试*攻击模型本身*
# 这种分割有助于评估攻击*可能*表现如何
X_mia_train, X_mia_test, y_mia_train, y_mia_test = train_test_split(
    X_mia, y_mia, test_size=0.3, stratify=y_mia, random_state=42
)

print(f"MIA 训练集形状: {X_mia_train.shape}")
print(f"MIA 测试集形状: {X_mia_test.shape}")
# 预期输出 (示例形状):
# MIA 训练集形状: (700, 10)  <- 假设有 10 个特征，总共 1000 条真实记录
# MIA 测试集形状: (300, 10)
```

### 训练攻击分类器

我们可以为此任务使用各种标准分类器。简单的逻辑回归或随机森林通常能提供一个合理的基准。这里我们使用逻辑回归，并加入特征缩放，这通常是可取的。

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 创建用于缩放和分类的管道
# 使用管道确保缩放一致应用
attack_pipeline = Pipeline([
    ('scaler', StandardScaler()), # 应用缩放，假设特征是数值型
    ('classifier', LogisticRegression(solver='liblinear', random_state=42))
])

# 在指定的 MIA 训练分割上训练攻击模型
attack_pipeline.fit(X_mia_train, y_mia_train)

print("攻击模型已训练。")
# 预期输出:
# 攻击模型已训练。
```

*注意：确保在所有涉及的数据集（`X_mia_train`、`X_mia_test`以及之后的`synthetic_data`）上一致应用适当的预处理（例如，数值特征的缩放，分类特征的独热编码）。Scikit-learn 的`Pipeline`对象在管理这些步骤时非常有效。*

### 评估攻击模型的表现

在将攻击模型应用于合成数据之前，我们必须使用保留数据分割（`X_mia_test`、`y_mia_test`）来评估它区分成员和非成员的能力。这项评估为通过此特定攻击设置可识别的潜在隐私风险提供了上限。接收者操作特征曲线下面积（AUC）是此评估的标准指标。AUC 为 0.5 表示性能与随机猜测相当（无区分能力），而 AUC 为 1.0 则表示完美分离。

```python
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve

# 在 MIA 测试集 (真实数据保留部分) 上预测概率
# 我们需要正类别 (成员=1) 的概率
y_pred_proba_mia_test = attack_pipeline.predict_proba(X_mia_test)[:, 1]

# 计算 AUC
auc_mia_test = roc_auc_score(y_mia_test, y_pred_proba_mia_test)

# 根据 0.5 阈值预测类别标签以计算准确率
y_pred_mia_test = attack_pipeline.predict(X_mia_test)
accuracy_mia_test = accuracy_score(y_mia_test, y_pred_mia_test)

print(f"攻击模型在真实保留数据上的表现:")
print(f"  AUC: {auc_mia_test:.4f}")
print(f"  准确率: {accuracy_mia_test:.4f}")

# 生成 ROC 曲线可视化数据
fpr, tpr, thresholds = roc_curve(y_mia_test, y_pred_proba_mia_test)
# 预期输出 (示例值):
# 攻击模型在真实保留数据上的表现:
#   AUC: 0.7231
#   准确率: 0.6800
```



![MIA 攻击模型 (在真实数据上) 的 ROC 曲线示例](plots/5697-0.json)



> 示例 ROC 曲线展示了成员推断攻击分类器在保留的真实数据上，真正率和假正率之间的权衡。蓝色线代表分类器的性能，虚线代表随机猜测（AUC=0.5）。曲线越接近左上角，表示攻击模型本身的区分能力越强。

`mia_test`集上的 AUC 明显高于 0.5，表明攻击模型已成功学习到在真实数据分布中区分成员和非成员的模式或特征。这意味着成员身份*可能*可以被推断。

### 评估合成数据中的风险

现在，我们进行主要的评估：将训练好的攻击模型应用于`synthetic_data`。分类器为每个合成记录生成的输出概率是我们关注的。这些概率反映了每个合成记录根据攻击模型看起来“有多像成员”。

```python
# 假设 synthetic_data 是一个 Pandas DataFrame，具有与 real_data 相同的特征
# 重要：使用*相同*的已拟合管道（包括缩放器）
# 在预测前转换合成数据。
# 管道会自动处理。

synthetic_pred_proba = attack_pipeline.predict_proba(synthetic_data)[:, 1]

# synthetic_pred_proba 现在包含每个合成记录的预测概率
# 表明其来源于原始训练集的可能性。

# 分析这些概率的分布
print("\n合成数据上预测成员概率的统计信息:")
# 使用 pandas describe() 提供快速摘要
print(pd.Series(synthetic_pred_proba).describe())

# 计算平均预测概率作为简单的汇总风险指标
average_risk_score = np.mean(synthetic_pred_proba)
print(f"\n合成数据的平均预测成员概率: {average_risk_score:.4f}")
# 预期输出 (示例):
# 合成数据上预测成员概率的统计信息:
# 计数    1000.000000
# 均值        0.5872
# 标准差         0.1534
# 最小值         0.1234
# 25% 分位数         0.4899
# 50% 分位数         0.5912
# 75% 分位数         0.6987
# 最大值         0.9567
# 名称: proportion, 类型: float64

# 合成数据的平均预测成员概率: 0.5872
```

### 解释

解释结果需要同时考虑攻击模型的有效性及其对合成数据的预测：

- **高 `auc_mia_test`：** 如果攻击模型在真实保留数据上表现强劲（例如，AUC > 0.7 或 0.8，此阈值取决于上下文 (context)），这证实了成员身份*可能可学习*，基于数据属性。
- **高 `synthetic_pred_proba` 分布：** 如果合成记录的预测概率分布偏向 1（高均值、中位数，或大量记录的概率 > 0.8 或 0.9），这表明合成数据生成过程可能过于紧密地复制了训练集特有的特征。这指向更高的隐私风险，因为合成数据可能无意中包含可追溯到原始成员的信息。将平均概率（`average_risk_score`）与基线预期（如果 MIA 训练数据平衡，通常为 0.5）进行比较。明显更高的平均分数表明存在风险。

相反，较低的`auc_mia_test`（接近 0.5）表明这个特定的攻击模型即使在真实数据中也难以区分成员。尽管这表明*根据此特定攻击测量的*风险较低，但这不保证能防御不同或更复杂的推断技术。

### 重要考量

- **攻击模型复杂度：** MIA 的复杂程度明显影响其发现。更复杂的分类器（例如，深度神经网络 (neural network)）可能会发现简单模型（如逻辑回归）遗漏的漏洞。可能需要进行实验。
- **数据特征：** 特征空间中成员和非成员的固有可分离性会影响 MIA 的成功。高维稀疏数据可能带来与低维密集数据不同的挑战。
- **相对表现：** 始终相对地解释 MIA 结果。将攻击在合成数据上的表现与其在真实保留集上的表现进行比较。如果攻击在真实数据上强劲但在合成数据上弱势，则隐私可能得到更好的保护。
- **全面视角：** MIA 只是隐私评估的一个组成部分。其结果应与属性推断攻击、基于距离的指标以及可能差分隐私分析（如果适用）的发现结合起来，以形成对隐私状况的全面评估。

这个实践练习为您提供了一种实现基本 MIA 的方法。仔细解释，并考虑攻击的局限性以及效用和其他隐私指标的更广范围，对于就合成数据的适用性做出明智决策非常重要。
