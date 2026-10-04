---
course: "mastering-gradient-boosting-algorithms"
chapter: "lightgbm-light-gradient-boosting"
lesson: "practice-implementing-lightgbm"
sourceId: 2005
sourceUrl: "https://apxml.com/zh/courses/mastering-gradient-boosting-algorithms/chapter-5-lightgbm-light-gradient-boosting/practice-implementing-lightgbm"
title: "动手实践：实现 LightGBM"
description: "使用 LightGBM 的 Python API 构建、训练和调优 LightGBM 模型，并比较性能。"
order: 8
plots: ["plots/2005-0.json"]
sourceHash: "42bc0bad935a567c86f10200dd413a2951dc17424d7f7bf514f0667cb38e06a3"
sourceCorrections: []
---

使用 LightGBM 的 Python API 实现 LightGBM 模型，需要应用 GOSS、EFB、基于直方图的分裂和叶式生长等原理。本实现将说明如何发挥 LightGBM 的独特功能，尤其是其对分类数据的高效处理能力和运行速度。

本次练习我们将使用一个合成数据集，以便控制其特性并关注 LightGBM 的实现方式。

### 设置与数据生成

首先，请确保你已安装所需的库（`lightgbm`、`scikit-learn`、`pandas`、`numpy`）。让我们导入它们并生成一个合成的分类数据集。我们将加入混合有信息量、冗余和分类的特征。

```python
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, roc_auc_score
import time

# 生成一个合成数据集
# 为了演示，使其具有适当的复杂度
n_samples = 5000
n_features = 30
n_informative = 15
n_redundant = 5
n_categorical = 5 # 我们将最后 5 个特征视为分类特征
n_clusters_per_class = 2
random_state = 42

X, y = make_classification(n_samples=n_samples,
                           n_features=n_features,
                           n_informative=n_informative,
                           n_redundant=n_redundant,
                           n_repeated=0,
                           n_classes=2,
                           n_clusters_per_class=n_clusters_per_class,
                           weights=[0.8, 0.2], # 引入一些不平衡
                           flip_y=0.05, # 添加噪声
                           class_sep=0.8,
                           random_state=random_state)

# 转换为 Pandas DataFrame 以便更方便地处理
feature_names = [f'num_{i}' for i in range(n_features - n_categorical)] + \
                [f'cat_{i}' for i in range(n_categorical)]
X = pd.DataFrame(X, columns=feature_names)

# 通过离散化最后 n_categorical 列来模拟分类特征
# 在实际情况中，这些特征通常已经是分类类型（字符串或整数）
for i in range(n_categorical):
    cat_col_name = f'cat_{i}'
    # 将浮点数转换为离散整数类别（例如：0, 1, 2, 3, 4）
    X[cat_col_name] = pd.qcut(X[cat_col_name], q=5, labels=False, duplicates='drop').astype(int)

# 识别分类特征的索引或名称
categorical_features_indices = [X.columns.get_loc(col) for col in feature_names if col.startswith('cat_')]
categorical_features_names = [col for col in feature_names if col.startswith('cat_')]

# 将分类列转换为 pandas 的 'category' dtype 以提高清晰度
# LightGBM 通常可以推断出这些，但显式声明更安全
for col in categorical_features_names:
    X[col] = X[col].astype('category')

print("数据集形状:", X.shape)
print("目标分布:", np.bincount(y))
print("已识别的分类特征:", categorical_features_names)
# print(X.info()) # 可选：检查数据类型
# print(X.head())  # 可选：查看数据

# 将数据分成训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=random_state, stratify=y)
```

### LightGBM 模型基础训练

我们先从训练一个默认的 LightGBM 分类器开始。LightGBM 兼容 scikit-learn 的 API 使这变得简单直接。

```python
# 初始化 LightGBM 分类器
lgbm_clf_default = lgb.LGBMClassifier(random_state=random_state)

# 训练模型
print("正在训练默认 LightGBM 模型...")
start_time = time.time()
lgbm_clf_default.fit(X_train, y_train)
end_time = time.time()
print(f"默认模型训练时间: {end_time - start_time:.2f} 秒")

# 进行预测
y_pred_default = lgbm_clf_default.predict(X_test)
y_pred_proba_default = lgbm_clf_default.predict_proba(X_test)[:, 1] # 正类的概率

# 评估模型
accuracy_default = accuracy_score(y_test, y_pred_default)
auc_default = roc_auc_score(y_test, y_pred_proba_default)

print(f"\n默认模型性能:")
print(f"准确率: {accuracy_default:.4f}")
print(f"AUC: {auc_default:.4f}")
```

这建立了一个使用默认参数 (parameter)的基准性能和训练时间。注意，我们尚未明确处理分类特征；LightGBM 可能会将其视为连续特征或进行合理猜测，但明确处理通常会更好。

### 使用 LightGBM 的特征

现在，让我们明确告知 LightGBM 我们的分类特征，并调整一些通常用于性能和正则化 (regularization)的参数 (parameter)。

1. **分类特征：** 使用 `categorical_feature` 参数。支持传入特征名称（如果使用 Pandas DataFrame）或索引。在 Pandas 中使用 `category` 数据类型通常能让 LightGBM 自动识别它们，但明确声明可以消除歧义。
2. **叶式生长控制：** `num_leaves` 是一个主要参数，控制 LightGBM 叶式生长策略中的模型复杂度。默认值为 31。增加它可提高准确性，但也存在过拟合 (overfitting)的风险。它通常与 `max_depth`（限制深度，尽管叶式生长受其约束较少）和 `min_child_samples`（叶子中所需的最少数据点）一起调整。
3. **学习率和估计器：** `learning_rate`（收缩率）和 `n_estimators`（提升轮数）共同作用。较小的学习率通常需要更多的估计器才能收敛，但常常能带来更好的泛化能力。
4. **子采样：** `feature_fraction`（相当于 `colsample_bytree`）和 `bagging_fraction`（`subsample`）分别控制特征和数据子采样，有助于正则化和提高速度。`bagging_freq` 决定了进行 bagging 的频率。

让我们训练另一个模型，并结合这些方面。

```python
# 初始化一个已配置的 LightGBM 分类器
lgbm_clf_configured = lgb.LGBMClassifier(
    objective='binary',            # 明确设置目标
    metric='auc',                  # 用于内部验证/提前停止的评估指标
    n_estimators=500,             # 增加提升轮数
    learning_rate=0.05,            # 降低学习率
    num_leaves=64,                 # 增加叶子数量
    max_depth=-1,                  # 无深度限制（叶式生长的典型设置）
    min_child_samples=20,          # 每个叶子的最少样本数
    feature_fraction=0.8,          # 特征子采样（类似于 colsample_bytree）
    bagging_fraction=0.8,          # 数据子采样（类似于 subsample）
    bagging_freq=5,                # 每 5 次迭代执行一次 bagging
    reg_alpha=0.1,                 # L1 正则化
    reg_lambda=0.1,                # L2 正则化
    n_jobs=-1,                     # 使用所有可用的 CPU 核心
    random_state=random_state
)

# 训练模型，明确传入分类特征信息
# 在使用 category 数据类型的 Pandas DataFrame 上训练时使用特征名称
# 或者，使用索引：categorical_feature=categorical_features_indices
print("\n正在训练带有分类特征的配置 LightGBM 模型...")
start_time = time.time()
lgbm_clf_configured.fit(X_train, y_train,
                        eval_set=[(X_test, y_test)],      # 提供评估集用于提前停止
                        eval_metric='auc',                # 用于评估的指标
                        callbacks=[lgb.early_stopping(100, verbose=False)], # 如果 AUC 在 100 轮内没有改善则停止
                        categorical_feature=categorical_features_names # 明确声明
                       )
end_time = time.time()
print(f"配置模型训练时间: {end_time - start_time:.2f} 秒")
print(f"找到的最佳迭代次数: {lgbm_clf_configured.best_iteration_}")

# 进行预测
y_pred_configured = lgbm_clf_configured.predict(X_test)
y_pred_proba_configured = lgbm_clf_configured.predict_proba(X_test)[:, 1]

# 评估配置模型
accuracy_configured = accuracy_score(y_test, y_pred_configured)
auc_configured = roc_auc_score(y_test, y_pred_proba_configured)

print(f"\n配置模型性能:")
print(f"准确率: {accuracy_configured:.4f}")
print(f"AUC: {auc_configured:.4f}")
```

**观察结果：**

- **分类处理：** 通过传入 `categorical_feature=categorical_features_names`，我们确保 LightGBM 对这些特征使用其优化的算法（例如 Fisher 最优分裂），这与将其视为连续特征或使用独热编码（这会大幅增加维度）相比，可能同时提高速度和准确性。请注意，在输入 DataFrame 中使用 `category` 数据类型是推荐的方式，因为 LightGBM 的内部机制针对它进行了优化。
- **训练时间：** 比较训练时间。虽然配置的模型具有更多的估计器，但像基于直方图的分裂、GOSS 和 EFB 等技术通常能让 LightGBM 保持非常快的速度，尤其是在数据量增长时。使用提前停止也能避免不必要的计算。
- **性能：** 比较准确率和 AUC 分数。调整 `num_leaves`、`learning_rate` 等参数，并结合正则化和子采样，通常有助于找到比默认设置更好的模型，尽管需要仔细调优（通常使用第 8 章中的技术）。提前停止有助于防止在估计器数量增加时出现过拟合。

### 可视化（可选）：特征重要性

LightGBM 提供内置的特征重要性计算。这有助于理解模型认为哪些特征的预测能力最强。

```python
import matplotlib.pyplot as plt
import seaborn as sns

# 获取特征重要性
importance_df = pd.DataFrame({
    'feature': lgbm_clf_configured.booster_.feature_name(),
    'importance': lgbm_clf_configured.feature_importances_
}).sort_values('importance', ascending=False)

# 绘制特征重要性
plt.figure(figsize=(10, 8))
sns.barplot(x='importance', y='feature', data=importance_df.head(20), palette='viridis') # 显示前 20 名
plt.title('LightGBM 特征重要性（配置模型）')
plt.tight_layout()
plt.show() # 在 Web 环境中，你可能会使用 Plotly 等库来渲染它

# 使用 Plotly 进行 Web 渲染的示例
import plotly.graph_objects as go
top_n = 20
fig = go.Figure(go.Bar(
    x=importance_df['importance'][:top_n],
    y=importance_df['feature'][:top_n],
    orientation='h',
    marker_color='#1f77b4' # 示例颜色
))
fig.update_layout(
    title=f'前 {top_n} 个特征重要性 (LightGBM)',
    yaxis={'categoryorder':'total ascending'},
    xaxis_title='重要性',
    yaxis_title='特征',
    height=500,
    margin=dict(l=120, r=20, t=50, b=50) # 调整标签边距
)
# fig.show() # 在支持 Plotly 渲染的笔记本或环境中运行此行

# 要嵌入到 Web 中，通常会输出 JSON：
# print(fig.to_json()) # 这会生成 JSON 字符串
```



![前 20 个特征重要性 (LightGBM)](plots/2005-0.json)



> 由配置的 LightGBM 模型计算的特征重要性，显示了前 20 个特征的相对贡献。请注意分类特征（例如 `分类_2`、`分类_0`）如何与数值特征同时出现。

本次实践练习体现了使用 LightGBM 的核心工作流程：模型初始化、模型训练（可选地使用提前停止和评估集）、进行预测，以及更重要的，配置模型以发挥其优势，例如原生的分类特征处理能力。对 `num_leaves`、`learning_rate`、`feature_fraction` 和 `bagging_fraction` 等参数 (parameter)进行试验对于优化你在特定任务上的性能非常重要，我们将在超参数 (hyperparameter)优化章节中详细介绍这一主题。

## 参考资料

- [LightGBM Documentation](https://lightgbm.readthedocs.io/en/latest/) — Microsoft and LightGBM Contributors (2024)
  官方文档提供了关于Python API、参数配置以及使用LightGBM推荐实践的全面信息。
- [Hands-On Machine Learning with Scikit-Learn, Keras, & TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125974/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media; Pages: 850
  这本实用指南涵盖了使用scikit-learn等Python库实现各种机器学习模型，包括梯度提升算法。（第三版）
- [Greedy Function Approximation: A Gradient Boosting Machine](https://projecteuclid.org/journals/annals-of-statistics/volume-29/issue-5/Greedy-Function-Approximation-A-Gradient-Boosting-Machine/10.1214/aos/1013203451.full) — Jerome H. Friedman (2001)
  Journal: The Annals of Statistics; Volume: 29; Pages: 1189-1232; DOI: [10.1214/aos/1013203451](https://doi.org/10.1214/aos/1013203451)
  这篇基础论文介绍了通用梯度提升机算法，为LightGBM等现代实现提供了理论基础。
