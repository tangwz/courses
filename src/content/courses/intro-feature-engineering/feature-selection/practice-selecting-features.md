---
course: "intro-feature-engineering"
chapter: "feature-selection"
lesson: "practice-selecting-features"
sourceId: 1421
sourceUrl: "https://apxml.com/zh/courses/intro-feature-engineering/chapter-6-feature-selection/practice-selecting-features"
title: "动手实践：特征选择"
description: "使用Scikit-learn应用过滤、包装（RFE）和嵌入（Lasso，树重要性）方法。"
order: 12
plots: []
sourceHash: "20e056b95ceb6869e97b0f774b5ea7e45d2eb118c9c93f36c4f9a4a7c2f3dfb9"
sourceCorrections: []
---

特征选择技术的实际应用将得到演示，主要关注过滤、包装和嵌入 (embedding)方法。使用Python的Scikit-learn库，将这些方法应用到一个人造数据集上，演示如何有效降低维度。

“首先，让我们搭建环境并生成一些数据。我们将使用`make_classification`创建一个分类数据集，其中包含一些有用特征、一些冗余特征和一些噪声特征。这种设置模拟了并非所有收集到的数据都对模型性能有积极作用的场景。”

```python
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.feature_selection import RFE, RFECV
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns # 使用seaborn便于绘图样式设置

# 生成一个人造数据集
# 共20个特征：8个有用，4个冗余，8个噪声
X, y = make_classification(n_samples=500, n_features=20,
                           n_informative=8, n_redundant=4,
                           n_repeated=0, n_classes=2,
                           n_clusters_per_class=2,
                           flip_y=0.05, # 为标签添加一些噪声
                           class_sep=0.7,
                           random_state=42)

# 转换为DataFrame便于处理
feature_names = [f'feature_{i}' for i in range(X.shape[1])]
X = pd.DataFrame(X, columns=feature_names)

# 将数据分割为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 缩放数据通常是好的做法，特别是对于RFE与逻辑回归或Lasso等方法
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 将缩放后的数组转换回DataFrame以保持清晰（可选，但有助于跟踪特征名称）
X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_names)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_names)

print("Original training data shape:", X_train.shape)
```

我们初始数据集有20个特征。我们的目标是运用特征选择技术，找出并保留最相关的特征。

### 应用过滤方法

过滤方法根据特征的自身属性进行评估，与任何特定模型无关。

#### 方差阈值法

这是最简单的方法，它移除方差不满足特定阈值的特征。它有助于消除常量或准常量特征。让我们移除方差为零的特征（尽管我们的人造数据可能不会有，但这是好的实践做法）。

```python
# 初始化VarianceThreshold（默认阈值0.0会移除常量特征）
selector_vt = VarianceThreshold()

# 在训练数据上拟合
selector_vt.fit(X_train_scaled)

# 获取要保留的特征
features_to_keep = X_train_scaled.columns[selector_vt.get_support()]
print(f"Features kept after VarianceThreshold: {len(features_to_keep)}/{X_train_scaled.shape[1]}")
# print("Kept features:", features_to_keep.tolist()) # 取消注释以查看名称

# 转换数据（通常您会同时转换训练集和测试集）
X_train_vt = selector_vt.transform(X_train_scaled)
X_test_vt = selector_vt.transform(X_test_scaled)

print("Shape after VarianceThreshold:", X_train_vt.shape)
```

在这种情况下，`VarianceThreshold`可能保留了所有特征，因为`make_classification`除非特别指定，否则通常不会产生常量特征。如果需要，您可以调整`threshold`参数 (parameter)来移除低方差特征。

#### 单变量选择

这些方法使用统计检验来评估每个特征与目标变量的关系。我们将使用`SelectKBest`和ANOVA F值检验（`f_classif`），它适用于数值特征和分类目标。

```python
# 根据ANOVA F值选择前10个特征
k = 10
selector_kbest = SelectKBest(score_func=f_classif, k=k)

# 在缩放后的训练数据和目标上拟合
selector_kbest.fit(X_train_scaled, y_train)

# 获取选定的特征名称
kbest_features = X_train_scaled.columns[selector_kbest.get_support()]
print(f"Selected top {k} features using SelectKBest (f_classif):")
print(kbest_features.tolist())

# 转换数据
X_train_kbest = selector_kbest.transform(X_train_scaled)
X_test_kbest = selector_kbest.transform(X_test_scaled)

print("Shape after SelectKBest:", X_train_kbest.shape)

# 您还可以查看分数
feature_scores = pd.DataFrame({'Feature': X_train_scaled.columns, 'Score': selector_kbest.scores_})
print("\nTop 5 features by ANOVA F-score:")
print(feature_scores.sort_values(by='Score', ascending=False).head())
```

`SelectKBest`提供了一种快速方式，可以根据所选的统计检验对特征的个体预测能力进行排名。请记住，`f_classif`评估的是线性关系；非线性关系可能会被忽略。对于分类特征，您可以使用`chi2`。

### 应用包装方法

包装方法使用特定的机器学习 (machine learning)模型来评估特征子集。

#### 递归特征消除 (RFE)

RFE递归地拟合模型，对特征进行排序（根据系数大小或特征重要性），并移除最弱的一个或多个，直到剩下所需数量。我们将使用`LogisticRegression`作为估计器。

```python
# 初始化估计器
estimator = LogisticRegression(solver='liblinear', random_state=42)

# 初始化RFE以选择8个特征
# 注意：RFE最适合提供系数权重或特征重要性的估计器
selector_rfe = RFE(estimator=estimator, n_features_to_select=8, step=1) # step=1表示每次迭代移除1个特征

# 在缩放后的训练数据上拟合RFE
selector_rfe.fit(X_train_scaled, y_train)

# 获取选定的特征名称
rfe_features = X_train_scaled.columns[selector_rfe.support_]
print(f"Selected {selector_rfe.n_features_} features using RFE (LogisticRegression):")
print(rfe_features.tolist())

# 转换数据
X_train_rfe = selector_rfe.transform(X_train_scaled)
X_test_rfe = selector_rfe.transform(X_test_scaled)

print("Shape after RFE:", X_train_rfe.shape)

# 带有交叉验证的RFE（RFECV）可以找出最佳特征数量
# estimator_cv = LogisticRegression(solver='liblinear', random_state=42)
# selector_rfecv = RFECV(estimator=estimator_cv, step=1, cv=5, scoring='accuracy') # 使用适当的评分指标
# selector_rfecv.fit(X_train_scaled, y_train)
# print(f"\nOptimal number of features found by RFECV: {selector_rfecv.n_features_}")
# rfecv_features = X_train_scaled.columns[selector_rfecv.support_]
# print("Selected features by RFECV:", rfecv_features.tolist())
```

RFE比过滤方法计算量更大，因为它涉及多次训练估计器。然而，它通过模型的评估隐式考虑特征间的关联性。`RFECV`根据交叉验证的性能，自动找出最佳特征数量。

### 应用嵌入 (embedding)方法

嵌入方法将特征选择作为模型训练过程的一部分进行。

#### Lasso (L1 正则化 (regularization))

带有L1正则化的线性模型（如Lasso）倾向于将不那么重要特征的系数精确地收缩到零，从而有效地执行特征选择。我们将使用带有L1惩罚的`LogisticRegression`。

```python
# 使用带有L1惩罚的逻辑回归
# 'C'参数是正则化强度的倒数；C越小表示正则化越强
# 我们将使用固定的C，但通常使用LassoCV或GridSearchCV来找出最佳的C值
l1_estimator = LogisticRegression(penalty='l1', C=0.1, solver='liblinear', random_state=42)
l1_estimator.fit(X_train_scaled, y_train)

# 非零系数对应于被选择的特征
l1_coeffs = l1_estimator.coef_[0]
l1_selected_features = X_train_scaled.columns[l1_coeffs != 0]

print(f"Selected features using Logistic Regression (L1 penalty, C=0.1): {len(l1_selected_features)}")
print(l1_selected_features.tolist())

# 为系数创建DataFrame
coef_df = pd.DataFrame({'Feature': X_train_scaled.columns, 'Coefficient': l1_coeffs})
print("\nL1 Coefficients:")
# 为简洁起见，只显示非零系数
print(coef_df[coef_df['Coefficient'] != 0].sort_values(by='Coefficient', key=abs, ascending=False))
```

Lasso高效且将选择与模型拟合直接集成。正则化强度（由`LogisticRegression`中的`C`或`LassoCV`中的`alpha`控制）决定了保留多少特征。

#### 基于树的特征重要性

基于树的集成方法，如随机森林，在训练过程中自然计算特征重要性，这是基于每个特征对降低所有树中的不纯度（例如，基尼不纯度或熵）的贡献程度。

````python
# 初始化随机森林分类器
rf_estimator = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# 拟合模型
rf_estimator.fit(X_train_scaled, y_train)

# 获取特征重要性
importances = rf_estimator.feature_importances_
importance_df = pd.DataFrame({'Feature': X_train_scaled.columns, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("\nFeature Importances from RandomForest:")
print(importance_df)

# 根据重要性选择特征（例如，保留前10个或高于某个阈值）
threshold = 0.02 # 示例阈值 - 根据分布调整
rf_selected_features = importance_df[importance_df['Importance'] > threshold]['Feature']

print(f"\nSelected features with importance > {threshold}: {len(rf_selected_features)}")
print(rf_selected_features.tolist())

# 可视化特征重要性
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(15), palette='viridis') # 显示前15个
plt.title('Top 15 Feature Importances from RandomForestClassifier')
plt.tight_layout()
plt.show() # 在Web环境中，您可以使用Plotly等库来渲染此图

# 生成用于Web渲染的Plotly JSON（前10个的示例）
top_10_importance = importance_df.head(10).sort_values(by='Importance', ascending=True) # 升序排列，用于水平条形图
plotly_fig_json = f'''
```plotly
{{
    "data": [
        {{
            "type": "bar",
            "y": {top_10_importance['Feature'].tolist()},
            "x": {top_10_importance['Importance'].tolist()},
            "orientation": "h",
            "marker": {{"color": "#20c997"}}
        }}
    ],
    "layout": {{
        "title": "随机森林前10位特征重要性",
        "yaxis": {{"title": "特征"}},
        "xaxis": {{"title": "重要性得分"}},
        "height": 400,
        "margin": {{"l": 120, "r": 20, "t": 50, "b": 50}}
    }}
}}
````

'''
print(plotly\_fig\_json)

````
> 随机森林模型计算的特征重要性，显示了每个特征对模型预测的相对贡献。分数越高表示越重要。

基于树的重要性评估很有效，因为它们可以捕获非线性关系和特征间的关联。然而，相关特征可能会分散重要性，可能低估其总价值。

### 将选择集成到管道中

特征选择的一个重要方面是在机器学习工作流中正确应用它，尤其是在使用交叉验证时。特征选择理想情况下应在每个交叉验证折叠的*内部*进行，仅使用该折叠的训练数据，以避免验证集数据泄露到选择过程中。Scikit-learn的`Pipeline`对象非常适合此用途。

```python
# 示例：结合RFE和逻辑回归的管道
pipeline = Pipeline([
    ('scaler', StandardScaler()), # 步骤1：缩放数据
    ('selector', RFE(estimator=LogisticRegression(solver='liblinear'), n_features_to_select=8)), # 步骤2：选择特征
    ('classifier', LogisticRegression(solver='liblinear')) # 步骤3：训练最终模型
])

# 现在，您可以拟合整个管道
pipeline.fit(X_train, y_train) # 在原始训练数据上拟合

# 在测试集上评估
accuracy = pipeline.score(X_test, y_test)
print(f"\nPipeline Accuracy (Scaler -> RFE -> LogisticRegression): {accuracy:.4f}")

# 您还可以使用带有管道的GridSearchCV来调整超参数
# 包括RFE中要选择的特征数量或Lasso中的C参数。
````

本次动手实践演示了应用过滤、包装和嵌入式特征选择方法，使用Scikit-learn。您了解了如何移除低方差特征、根据统计检验选择特征、使用基于模型的递归消除，以及运用正则化或基于树的重要性评估。请记住，最佳方法取决于数据集特点、所选模型和计算限制。将选择集成到`Pipeline`中可确保其在模型开发和评估过程中得到正确应用。

## 参考资料

- [Feature selection](https://scikit-learn.org/stable/modules/feature_selection.html) — scikit-learn developers (2023)
  提供 Scikit-learn 中过滤法、包裹法和嵌入法特征选择方法的实践指导和 API 文档。
- [Pipelines and composite estimators](https://scikit-learn.org/stable/modules/compose.html#pipeline-chaining-estimators) — scikit-learn developers (2023)
  阐释如何使用管道构建稳健的机器学习工作流程，包含特征选择步骤，以避免数据泄漏。
- [An Introduction to Statistical Learning: With Applications in R](https://www.statlearning.com/) — Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani (2021)
  Publisher: Springer; DOI: [10.1007/978-1-0716-1418-1](https://doi.org/10.1007/978-1-0716-1418-1)
  涵盖统计学习方法、正则化和模型选择的基础著作，为多种特征选择技术提供理论依据。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  一本关于机器学习模型和管道实现的实践指南，包含使用 Scikit-learn 进行特征选择的特定章节。
