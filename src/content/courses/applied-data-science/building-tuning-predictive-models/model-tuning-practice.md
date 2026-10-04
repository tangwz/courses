---
course: "applied-data-science"
chapter: "building-tuning-predictive-models"
lesson: "model-tuning-practice"
sourceId: 1873
sourceUrl: "https://apxml.com/zh/courses/applied-data-science/chapter-3-building-tuning-predictive-models/model-tuning-practice"
title: "动手实践：模型训练与超参数优化"
description: "实践构建、评估和调优各种监督学习模型。"
order: 8
plots: ["plots/1873-0.json"]
sourceHash: "cf6ee780a446bf78a0aaef5429943e642fa9dda2cb7b930812ed2081de8ea8ac"
sourceCorrections: []
---

本实践练习将指导用户使用 scikit-learn 构建、比较和优化预测模型。它应用监督模型训练、评估和调优的技术。我们将使用一个已准备好的数据集，其中数据获取和特征工程步骤已完成，以生成合适的训练集和测试集。

对于本练习，假设我们有一个分类任务。我们将使用 `X_train`、`y_train` 进行训练和验证，并使用 `X_test`、`y_test` 进行最终评估。请确保您已安装并导入 `pandas`、`numpy` 和 `scikit-learn`。

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
# 假设X和y已加载（例如，从CSV文件）
# 为了演示，我们生成合成数据
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                           n_redundant=5, n_classes=2, random_state=42)

# 将数据拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 缩放数值特征（对逻辑回归很重要，通常也是好做法）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"训练集形状: {X_train_scaled.shape}")
print(f"测试集形状: {X_test_scaled.shape}")
```

### 基线模型：逻辑回归

我们从一个简单的模型——逻辑回归开始，以设定基准性能。我们将在训练数据上使用5折交叉验证（$k=5$），以获取对其性能的可靠评估。目前，我们将使用准确率作为主要指标，但稍后会查看其他指标。

```python
# 初始化模型
log_reg = LogisticRegression(random_state=42, max_iter=1000) # 增加max_iter以帮助收敛

# 执行5折交叉验证
cv_scores_log_reg = cross_val_score(log_reg, X_train_scaled, y_train, cv=5, scoring='accuracy')

print(f"逻辑回归交叉验证准确率分数: {cv_scores_log_reg}")
print(f"平均交叉验证准确率: {np.mean(cv_scores_log_reg):.4f}")
print(f"交叉验证准确率标准差: {np.std(cv_scores_log_reg):.4f}")
```

这为我们提供了各折的平均准确率分数及其标准差。这是我们的初始基准。任何后续的模型或调优都应旨在提高此平均准确率，理想情况下同时保持较低的标准差。

### 尝试一个更复杂的模型：随机森林

现在，我们尝试一个更复杂的、基于树的集成模型，例如随机森林。我们将首先使用默认超参数 (parameter) (hyperparameter)，并采用相同的交叉验证策略对其进行评估。

```python
# 初始化模型
rf_clf = RandomForestClassifier(random_state=42)

# 执行5折交叉验证
cv_scores_rf = cross_val_score(rf_clf, X_train_scaled, y_train, cv=5, scoring='accuracy')

print(f"随机森林交叉验证准确率分数: {cv_scores_rf}")
print(f"平均交叉验证准确率: {np.mean(cv_scores_rf):.4f}")
print(f"交叉验证准确率标准差: {np.std(cv_scores_rf):.4f}")
```

比较随机森林与逻辑回归基线的平均准确率。通常，更复杂的模型可以开箱即用地获得更高的性能，但它们也有更多的超参数，这些参数可以显著影响其行为。我们假设随机森林表现更好，并继续对其进行调优。

### 使用网格搜索进行超参数 (parameter) (hyperparameter)调优

网格搜索系统地尝试多种参数组合，交叉验证每个组合，以根据指定的评分指标确定哪个提供最佳性能。我们来调优 `RandomForestClassifier` 的 `n_estimators`（树的数量）和 `max_depth`（每棵树的最大深度）。

```python
# 定义参数网格
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

# 初始化网格搜索
# 我们使用缩放后的训练数据
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                           param_grid=param_grid,
                           cv=3, # 在本例中使用3折交叉验证以加快搜索速度
                           scoring='accuracy',
                           n_jobs=-1, # 使用所有可用的CPU核心
                           verbose=1) # 显示进度

# 将网格搜索拟合到数据
grid_search.fit(X_train_scaled, y_train)

# 打印最佳参数及对应的分数
print(f"网格搜索找到的最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证准确率分数: {grid_search.best_score_:.4f}")

# 直接访问最佳估计器
best_rf_grid = grid_search.best_estimator_
```

网格搜索检查 `param_grid` 中定义的所有组合。在本例中，有 $3 \times 3 \times 2 = 18$ 种组合。每个组合都使用3折交叉验证进行评估，这意味着执行了 $18 \times 3 = 54$ 次模型拟合。`n_jobs=-1` 会并行化此过程。

### 使用随机搜索进行超参数 (parameter) (hyperparameter)调优

当超参数空间很大时，网格搜索的计算成本可能很高。随机搜索提供了一种更有效的替代方案，它从指定的分布或列表中采样固定数量的参数组合。

我们尝试调优相同的随机森林模型，但这次使用 `RandomizedSearchCV` 尝试更多参数或更宽的范围。

```python
from scipy.stats import randint

# 定义参数分布
param_dist = {
    'n_estimators': randint(50, 300), # 采样50到299之间的整数
    'max_depth': [None, 10, 20, 30], # 从列表中选择
    'min_samples_split': randint(2, 11), # 采样2到10之间的整数
    'min_samples_leaf': randint(1, 11), # 采样1到10之间的整数
    'bootstrap': [True, False] # 布尔选项
}

# 初始化随机搜索
random_search = RandomizedSearchCV(estimator=RandomForestClassifier(random_state=42),
                                   param_distributions=param_dist,
                                   n_iter=20, # 采样的参数设置数量
                                   cv=3, # 使用3折交叉验证
                                   scoring='accuracy',
                                   n_jobs=-1,
                                   random_state=42, # 为了可重现性
                                   verbose=1)

# 将随机搜索拟合到数据
random_search.fit(X_train_scaled, y_train)

# 打印最佳参数及对应的分数
print(f"随机搜索找到的最佳参数: {random_search.best_params_}")
print(f"最佳交叉验证准确率分数: {random_search.best_score_:.4f}")

# 访问最佳估计器
best_rf_random = random_search.best_estimator_
```

`RandomizedSearchCV` 执行 `n_iter`（此处为20）次试验。在每次试验中，它会从指定的分布或列表中随机采样每个超参数的值。与穷举网格搜索相比，这通常能更快地找到非常好的参数组合，尤其当某些超参数比其他参数更具影响力时。

### 在测试集上的最终评估

调优后，我们选择找到的最佳模型（本例中使用随机搜索找到的模型，假设它给出了最佳交叉验证分数）。我们用*整个*训练集（`X_train_scaled`，`y_train`）训练此最终模型配置，然后在*未见过*的测试集（`X_test_scaled`，`y_test`）上评估其性能。这提供了模型在新数据上预期表现的无偏估计。

```python
# 使用随机搜索找到的最佳估计器
final_model = random_search.best_estimator_

# 注意：来自GridSearchCV/RandomizedSearchCV的best_estimator_
# 在refit=True（默认）时已在整个训练集上训练。
# 如果refit=False，您需要显式训练它：
# final_model.fit(X_train_scaled, y_train)

# 在测试集上进行预测
y_pred_test = final_model.predict(X_test_scaled)
y_pred_proba_test = final_model.predict_proba(X_test_scaled)[:, 1] # 用于ROC AUC的概率

# 评估最终模型
test_accuracy = accuracy_score(y_test, y_pred_test)
test_roc_auc = roc_auc_score(y_test, y_pred_proba_test)

print(f"\n--- 测试集上的最终模型评估 ---")
print(f"测试准确率: {test_accuracy:.4f}")
print(f"测试ROC AUC: {test_roc_auc:.4f}")
print("\n分类报告:")
print(classification_report(y_test, y_pred_test))
print("\n混淆矩阵:")
print(confusion_matrix(y_test, y_pred_test))
```

分类报告为每个类别提供精确度、召回率和 $F_1$ 分数，比单独的准确率提供更详细的性能视图。混淆矩阵显示真阳性、真阴性、假阳性和假阴性的数量。ROC AUC 分数评估模型根据预测概率区分类别的能力。

让我们可视化最终调优的随机森林模型的特征重要性。这可以提供关于模型最依赖哪些特征的信息。

````python
# 获取特征重要性
importances = final_model.feature_importances_
indices = np.argsort(importances)[::-1] # 按重要性排序特征

# 准备绘图数据（前10个特征）
n_top_features = 10
feature_names = [f"Feature {i}" for i in range(X_train_scaled.shape[1])] # 占位符名称
top_indices = indices[:n_top_features]
top_importances = importances[top_indices]
top_feature_names = [feature_names[i] for i in top_indices]

# 创建Plotly条形图JSON
plotly_json = {
    "data": [{
        "type": "bar",
        "x": top_importances[::-1], # 反转以在图中升序显示
        "y": top_feature_names[::-1],
        "orientation": "h",
        "marker": {"color": "#228be6"} # 蓝色
    }],
    "layout": {
        "title": "前10个特征重要性（随机森林）",
        "xaxis": {"title": "重要性分数"},
        "yaxis": {"title": "特征"},
        "margin": {"l": 100, "r": 20, "t": 50, "b": 50} # 调整边距
    }
}
import json
print("```plotly")
print(json.dumps(plotly_json))
print("```")
````



![前10个特征重要性（随机森林）](plots/1873-0.json)



> 特征重要性得自于最终调优的随机森林模型，显示了根据基尼重要性度量的前10个最具影响力的特征。

### 总结

在本实践部分，您实践了监督学习 (supervised learning)模型训练、评估和调优的完整流程：

1. 使用逻辑回归和交叉验证设定了基准性能。
2. 训练了一个更复杂的模型（随机森林）并比较了其初始性能。
3. 应用 `GridSearchCV` 系统地搜索了定义的超参数 (parameter) (hyperparameter)空间。
4. 利用 `RandomizedSearchCV` 对更大空间或分布进行了更高效的搜索。
5. 根据交叉验证性能选择了最佳超参数配置。
6. 在独立测试集上使用多个指标（准确率、ROC AUC、分类报告、混淆矩阵）评估了最终的调优模型，以获取无偏的性能估计。
7. 可视化了特征重要性，以从模型中获取理解。

这种训练、评估和调优的迭代过程对于开发有效的机器学习 (machine learning)模型很重要。请记住，模型、超参数、评估指标和调优策略的选择，很大程度上取决于具体问题、数据集特点和可用的计算资源。

## 参考资料

- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html) — Scikit-learn Developers (2024)
  提供关于使用scikit-learn库进行模型训练、交叉验证和超参数优化的全面信息。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  一本实用书籍，指导读者使用scikit-learn实现机器学习模型，包括集成方法、交叉验证和超参数调优。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer
  本书提供各种机器学习方法背后的详细统计理论，例如分类、集成模型和模型评估技术。
- [Random Search for Hyper-Parameter Optimization](https://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf) — James Bergstra and Yoshua Bengio (2012)
  Journal: Journal of Machine Learning Research; Publisher: JMLR.org; Volume: 13; Pages: 281-305; DOI: [10.5555/2283516.2283526](https://doi.org/10.5555/2283516.2283526)
  一篇研究论文，介绍了随机搜索作为一种高效的超参数优化策略，通常通过关注更具影响力的参数来优于网格搜索。
