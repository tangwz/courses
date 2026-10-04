---
course: "evaluating-synthetic-data-quality"
chapter: "evaluating-ml-utility"
lesson: "practice-running-tstr-evaluations"
sourceId: 5675
sourceUrl: "https://apxml.com/zh/courses/evaluating-synthetic-data-quality/chapter-3-evaluating-ml-utility/practice-running-tstr-evaluations"
title: "动手实践：运行TSTR评估"
description: "使用 scikit-learn 和合成数据集实现 TSTR 评估。"
order: 6
plots: ["plots/5675-0.json"]
sourceHash: "684a189b5ff3f5c704a6f29d411585fa13e71f5b6f61c7fbb333f93862cc8c41"
sourceCorrections: []
---

实现训练-合成-测试-真实 (TSTR) 评估流程涉及使用 Python 和常用数据科学库。目标是仅使用合成数据训练机器学习 (machine learning)模型，并用未见过的真实数据评估其性能，从而直接衡量合成数据对于后续任务的实用性。

### 设置评估环境

首先，我们需要准备数据集：原始真实数据集和从中生成的合成数据集。为了评估的公平性，我们必须对真实数据进行合理划分。一部分将用于训练基准模型（训练-真实-测试-真实，即TRTR），而另一部分独立保留的数据将作为基准模型和TSTR模型*共同*的测试集。非常必要的是，这个真实测试集在合成数据生成过程中不能以任何形式被使用过。

假设你的真实数据在名为 `real_df` 的 Pandas DataFrame 中，而合成数据在 `synthetic_df` 中。两者都应包含特征和目标变量（我们称之为 `'target'`）。

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np

# 假设 real_df 和 synthetic_df 是已预加载的 Pandas DataFrame
# 示例结构:
# real_df = pd.DataFrame(np.random.rand(1000, 5), columns=[f'feat_{i}' for i in range(5)])
# real_df['target'] = np.random.randint(0, 2, 1000)
# synthetic_df = pd.DataFrame(np.random.rand(1000, 5), columns=[f'feat_{i}' for i in range(5)])
# synthetic_df['target'] = np.random.randint(0, 2, 1000)

# 分离真实数据的特征 (X) 和目标 (y)
X_real = real_df.drop('target', axis=1)
y_real = real_df['target']

# 将真实数据分割为训练集和测试集
# 测试集 (X_test_real, y_test_real) 仅保留用于评估。
X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
    X_real, y_real, test_size=0.3, random_state=42, stratify=y_real
)

# 准备合成数据（假设其列与真实数据相同）
X_synthetic = synthetic_df.drop('target', axis=1)
y_synthetic = synthetic_df['target']

print(f"真实训练集大小: {X_train_real.shape[0]}")
print(f"真实测试集大小: {X_test_real.shape[0]}")
print(f"合成训练集大小: {X_synthetic.shape[0]}")
```

确保对真实训练数据执行的任何预处理步骤（如缩放或编码）在训练TSTR模型之前也一致地应用于合成数据，并在评估之前应用于真实测试数据。

### 训练基准模型 (TRTR)

为了了解合成数据表现如何，我们需要一个参照。这个参照是通过在真实训练数据上训练模型，并在真实测试数据上进行评估来确定的。我们在此示例中使用一个标准分类器，如随机森林。

```python
# 初始化分类器
# TRTR 和 TSTR 都使用相同的模型类型和超参数
model_params = {'n_estimators': 100, 'max_depth': 10, 'random_state': 42}
baseline_model = RandomForestClassifier(**model_params)

# 在真实数据上训练基准模型
baseline_model.fit(X_train_real, y_train_real)

# 在真实测试集上评估基准模型
baseline_predictions = baseline_model.predict(X_test_real)
baseline_probabilities = baseline_model.predict_proba(X_test_real)[:, 1] # 用于计算 AUC 的概率

# 计算基准模型性能指标
baseline_accuracy = accuracy_score(y_test_real, baseline_predictions)
baseline_f1 = f1_score(y_test_real, baseline_predictions)
baseline_auc = roc_auc_score(y_test_real, baseline_probabilities)

print("\n基准模型性能 (训练真实数据, 测试真实数据):")
print(f"准确率: {baseline_accuracy:.4f}")
print(f"F1 分数: {baseline_f1:.4f}")
print(f"AUC: {baseline_auc:.4f}")

# 存储结果以供比较
results = {
    'TRTR': {'Accuracy': baseline_accuracy, 'F1': baseline_f1, 'AUC': baseline_auc}
}
```

这些 TRTR 分数表示在所选模型设置下使用原始数据可达到的性能。

### 训练 TSTR 模型

现在，我们执行 TSTR 的核心步骤。我们训练一个相同的模型（相同类型，相同超参数 (parameter) (hyperparameter)），但使用*合成*数据进行训练。评估仍需在*真实*测试集（`X_test_real`，`y_test_real`）上进行，这一点很重要。

```python
# 为 TSTR 初始化一个相同的分类器
tstr_model = RandomForestClassifier(**model_params)

# 在合成数据上训练 TSTR 模型
# 确保 X_synthetic 具有与 X_train_real 相同的特征
tstr_model.fit(X_synthetic, y_synthetic)

# 在真实测试集上评估 TSTR 模型
tstr_predictions = tstr_model.predict(X_test_real)
tstr_probabilities = tstr_model.predict_proba(X_test_real)[:, 1] # 用于计算 AUC 的概率

# 计算 TSTR 性能指标
tstr_accuracy = accuracy_score(y_test_real, tstr_predictions)
tstr_f1 = f1_score(y_test_real, tstr_predictions)
tstr_auc = roc_auc_score(y_test_real, tstr_probabilities)

print("\nTSTR 模型性能 (训练合成数据, 测试真实数据):")
print(f"准确率: {tstr_accuracy:.4f}")
print(f"F1 分数: {tstr_f1:.4f}")
print(f"AUC: {tstr_auc:.4f}")

# 存储结果以供比较
results['TSTR'] = {'Accuracy': tstr_accuracy, 'F1': tstr_f1, 'AUC': tstr_auc}
```

### 结果比较与解读

最后一步是比较 TRTR 基准和 TSTR 评估的性能指标。TSTR 指标越接近 TRTR 指标，合成数据在机器学习 (machine learning)方面的实用性就越高。

```python
# 创建一个 DataFrame 以便比较
results_df = pd.DataFrame(results).T # 转置以提高可读性
print("\n模型性能比较:")
print(results_df)

# 计算性能比率（例如，TSTR 分数 / TRTR 分数）
performance_ratio = results_df.loc['TSTR'] / results_df.loc['TRTR']
print("\n性能比率 (TSTR 分数 / TRTR 分数):")
print(performance_ratio)
```

性能比率接近 1.0 表明实用性出色；在合成数据上训练的模型性能几乎与在真实数据上训练的模型一样好。明显低于 1.0 的比率（例如 < 0.8）表明合成数据缺乏真实数据中存在的关键模式，限制了其对特定任务和模型的有用性。中等比率（例如 0.8 - 0.95）可能可接受，这取决于应用的容忍度和使用合成数据所带来的益处（如隐私）。

我们可以将此比较可视化：



![TSTR 与 TRTR 模型性能比较](plots/5675-0.json)



> 比较在真实数据 (TRTR) 和合成数据 (TSTR) 上训练的模型（两者均在相同真实测试集上评估）的性能指标（准确率、F1 分数、AUC）。TSTR 条形图越低表示实用性越差。

### 进一步考虑

- **模型选择：** 模型选择（本例中为随机森林）会影响结果。通常有益的做法是使用与您实际后续应用相关的不同类型模型（例如，逻辑回归、梯度提升、神经网络 (neural network)）重复进行 TSTR 评估。
- **超参数 (parameter) (hyperparameter)：** 我们使用了固定的超参数。如前所述，合成数据可能会导致与真实数据不同的最优超参数。分别为 TRTR 和 TSTR 情况运行超参数优化可以提供更多信息，但这会增加比较的复杂性。
- **数据大小：** 合成数据集相对于真实训练集的相对大小会影响性能。请确保您正在比较大小合理的数据集。
- **任务复杂性：** 实用性可能会因预测任务的复杂性而异（例如，二元分类 vs. 多分类 vs. 回归）。

本实践练习演示了 TSTR 的核心流程。通过系统地比较在合成数据与真实数据上训练的模型的性能，您可以获得有关合成数据对您的机器学习 (machine learning)目标所具有实际价值的量化 (quantization)证据。请记住根据您的具体用例调整模型、指标和解读。

## 参考资料

- [A Review of Synthetic Data Generation Methods and Their Evaluation](https://dl.acm.org/doi/10.1145/3544673) — Qi-Yue Ma, Sheng-Jun Huang, Wen-Bo Hu (2022)
  Journal: ACM Computing Surveys; Publisher: ACM; Volume: 55; Pages: Article 145, pp. 1-36; DOI: [10.1145/3544673](https://doi.org/10.1145/3544673)
  本综述讨论了多种合成数据生成方法，并提供了评估指标的分类，包括评估TSTR等机器学习效用的指标。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125968/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media, Inc.
  这本实用指南提供了关于使用Scikit-Learn进行机器学习模型实现、训练和评估的清晰说明，支持了所提供的代码示例和方法。
