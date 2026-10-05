# 实践：构建基础GBM模型

来源：[原文](https://apxml.com/zh/courses/mastering-gradient-boosting-algorithms/chapter-2-gradient-boosting-algorithm-depth/practice-basic-gbm)

[返回章节目录](README.md) · [返回课程目录](../README.md)

Scikit-learn的梯度提升机实现（`GradientBoostingRegressor` 和 `GradientBoostingClassifier`）提供了一种构建和训练基础GBM模型的实用方法。尽管XGBoost和LightGBM等库提供了显著的性能提升和功能增强（我们将在稍后介绍），但理解Scikit-learn版本仍能提供一个稳固且易于理解的基础。这个基础直接展示了梯度提升的根本概念，例如叠加性、损失函数 (loss function)、收缩和子采样。

我们将逐步讲解如何设置、训练和评估GBM，分别用于回归和分类任务。

### 环境配置

首先，请确保您已安装必要的库。我们将主要使用Scikit-learn，以及Pandas用于数据处理，NumPy用于数值运算。

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, roc_auc_score
from sklearn.datasets import fetch_california_housing, load_breast_cancer
import matplotlib.pyplot as plt
import seaborn as sns

# 设置图表的一致样式
sns.set_style("whitegrid")
```

### 实践：用于回归的GBM

让我们使用加利福尼亚房价数据集来解决一个回归问题。我们的目标是根据各种特征预测房屋中位价。

1. **加载和准备数据**：我们加载数据集并将其分割为训练集和测试集。

   ```python
   # 加载数据
   housing = fetch_california_housing()
   X = pd.DataFrame(housing.data, columns=housing.feature_names)
   y = housing.target

   # 分割数据
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

   print(f"训练特征形状: {X_train.shape}")
   print(f"测试特征形状: {X_test.shape}")
   ```
2. **实例化并训练模型**：我们创建一个`GradientBoostingRegressor`实例。下面我们来看一些源于理论讨论的重要参数 (parameter)：

   - `n_estimators`: 要执行的提升阶段（树）的数量。这对应于我们叠加模型公式 $F_M(x) = \sum_{m=1}^{M} \gamma_m h_m(x)$ 中的 $M$。
   - `learning_rate`: 这是收缩参数 $\nu$。它缩放每个树的贡献。较小的值需要更多的树（`n_estimators`）才能获得可比的性能，但通常会提高泛化能力。
   - `loss`: 指定要优化的损失函数 (loss function)。默认的`'squared_error'`（或`'ls'`）对应于最小化实际值和预测值之间的平方差之和，其中负梯度就是残差 $y_i - F_{m-1}(x_i)$。其他选项如`'absolute_error'`（对异常值具有鲁棒性）或`'huber'`（一种组合）可用。
   - `max_depth`: 控制单个回归估计器（树）的最大深度。这是控制模型复杂性并防止过拟合 (overfitting)的主要方法。
   - `subsample`: 如果小于1.0，这将通过在训练数据的随机部分上拟合树来启用随机梯度提升。这引入了随机性并充当正则化 (regularization)器。0.8左右的值很常见。

   ```python
   # 实例化GBM回归器
   gbr = GradientBoostingRegressor(
       n_estimators=100,      # 树的数量
       learning_rate=0.1,     # 收缩因子
       max_depth=3,           # 每棵树的最大深度
       subsample=0.8,         # 用于拟合每棵树的样本比例
       loss='squared_error',
       random_state=42
   )

   # 训练模型
   print("正在训练GradientBoostingRegressor...")
   gbr.fit(X_train, y_train)
   print("训练完成。")
   ```
3. **进行预测和评估**：我们使用训练好的模型在测试集上进行预测，并使用均方误差（MSE）和R方（$R^2$）评估性能。

   ```python
   # 在测试集上进行预测
   y_pred_reg = gbr.predict(X_test)

   # 评估模型
   mse = mean_squared_error(y_test, y_pred_reg)
   r2 = r2_score(y_test, y_pred_reg)

   print(f"测试集均方误差: {mse:.4f}")
   print(f"测试集R方: {r2:.4f}")
   ```

   您应该会观察到合理的性能指标。调整`n_estimators`、`learning_rate`和`max_depth`将显著影响这些结果。例如，增加`n_estimators`同时降低`learning_rate`通常会得到更好的模型，尽管这会增加训练时间。

### 实践：用于分类的GBM

现在，我们使用威斯康星乳腺癌数据集将GBM应用于二元分类问题。

1. **加载和准备数据**：

   ```python
   # 加载数据
   cancer = load_breast_cancer()
   X_c = pd.DataFrame(cancer.data, columns=cancer.feature_names)
   y_c = cancer.target

   # 分割数据
   X_c_train, X_c_test, y_c_train, y_c_test = train_test_split(X_c, y_c, test_size=0.2, random_state=42, stratify=y_c)

   print(f"训练分类特征形状: {X_c_train.shape}")
   print(f"测试分类特征形状: {X_c_test.shape}")
   ```
2. **实例化并训练模型**：我们使用`GradientBoostingClassifier`。重要参数 (parameter)与回归器类似，但损失函数 (loss function)不同。

   - `loss`: 默认的`'log_loss'`（以前是`'deviance'`）适用于二元和多元分类，优化逻辑损失函数。在这种情况下，负梯度涉及概率。`'exponential'`使用AdaBoost指数损失函数。

   ```python
   # 实例化GBM分类器
   gbc = GradientBoostingClassifier(
       n_estimators=100,
       learning_rate=0.1,
       max_depth=3,
       subsample=0.8,
       loss='log_loss',
       random_state=42
   )

   # 训练模型
   print("正在训练GradientBoostingClassifier...")
   gbc.fit(X_c_train, y_c_train)
   print("训练完成。")
   ```
3. **进行预测和评估**：我们使用准确率和ROC AUC分数等标准分类指标进行评估。我们还可以使用`predict_proba`获取概率估计。

   ```python
   # 在测试集上进行预测
   y_pred_class = gbc.predict(X_c_test)
   y_pred_proba = gbc.predict_proba(X_c_test)[:, 1] # 正类的概率

   # 评估模型
   accuracy = accuracy_score(y_c_test, y_pred_class)
   roc_auc = roc_auc_score(y_c_test, y_pred_proba)

   print(f"测试集准确率: {accuracy:.4f}")
   print(f"测试集ROC AUC分数: {roc_auc:.4f}")
   ```

   同样，调优超参数 (hyperparameter)对于获得最佳性能非常重要。

### 特征重要性

梯度提升模型根据每个特征在所有树中对减少损失函数 (loss function)的贡献程度，提供特征重要性的估计。Scikit-learn通过`feature_importances_`属性提供此信息。

```python
# 获取回归模型的特征重要性
importances_reg = gbr.feature_importances_
feature_names_reg = X.columns
importance_df_reg = pd.DataFrame({'Feature': feature_names_reg, 'Importance': importances_reg})
importance_df_reg = importance_df_reg.sort_values(by='Importance', ascending=False)

# 绘制特征重要性
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df_reg.head(10), palette='viridis') # 绘制前10名
plt.title('前10名特征重要性（GBM回归器）')
plt.xlabel('重要性分数')
plt.ylabel('特征')
plt.tight_layout()
plt.show()

# 获取分类模型的特征重要性
importances_cls = gbc.feature_importances_
feature_names_cls = X_c.columns
importance_df_cls = pd.DataFrame({'Feature': feature_names_cls, 'Importance': importances_cls})
importance_df_cls = importance_df_cls.sort_values(by='Importance', ascending=False)

# 绘制特征重要性
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df_cls.head(10), palette='magma') # 绘制前10名
plt.title('前10名特征重要性（GBM分类器）')
plt.xlabel('重要性分数')
plt.ylabel('特征')
plt.tight_layout()
plt.show()
```

> 训练好的Scikit-learn GBM模型得出的回归（加利福尼亚房价）和分类（乳腺癌）任务的特征重要性图。这些图显示了每个特征在模型决策中的相对贡献。

### 讨论

本次实际操作练习展示了使用Scikit-learn应用GBM的核心工作流程。您实例化了模型，配置了与GBM理论相关的核心超参数 (parameter) (hyperparameter)（估计器数量、学习率、树深度、子采样），训练并评估了它们的性能。

请记住，Scikit-learn的`GradientBoostingRegressor`和`GradientBoostingClassifier`对于理解算法的运行机制非常有价值，但对于大型数据集或复杂情况，可能不是性能最佳的选择。它们缺少一些高级正则化 (regularization)技术、优化的分裂寻找算法以及对稀疏或类别数据的高效处理能力，而这些在XGBoost、LightGBM和CatBoost等库中可以找到。

将此练习视为一个起点。您现在对标准GBM的运作方式有了实践认知。在接下来的章节中，我们将在在此基础上继续学习，了解那些已经成为现代梯度提升应用主力工具的专用算法。

## 参考资料

- [Greedy Function Approximation: A Gradient Boosting Machine](https://www.jstor.org/stable/2677051) — Jerome H. Friedman (2001)
  Journal: The Annals of Statistics; Publisher: Institute of Mathematical Statistics; Volume: 29; Pages: 1189-1232; DOI: [10.1214/aos/1013203451](https://doi.org/10.1214/aos/1013203451)
  介绍梯度提升机（GBM）算法的原始学术论文，确立了其理论基础。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://hastie.su.domains/ElemStatLearn/ESLII_print12.pdf) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer; Pages: Chapter 10, "Boosting and Additive Trees"; DOI: [10.1007/978-0-387-84858-7_10](https://doi.org/10.1007/978-0-387-84858-7_10)
  一本标准教科书，在加性模型的背景下介绍了梯度提升的统计学基础。
- [Scikit-learn User Guide: Gradient Tree Boosting](https://scikit-learn.org/stable/modules/ensemble.html#gradient-tree-boosting) — Scikit-learn developers (2023)
  Scikit-learn 官方文档，介绍了 `GradientBoostingRegressor` 和 `GradientBoostingClassifier` 的参数及使用方法。

---

[上一节](07-%E4%BD%BF%E7%94%A8%20Scikit-learn%20%E5%AE%9E%E7%8E%B0%20GBM.md) · [下一节](../03-%E6%A2%AF%E5%BA%A6%E6%8F%90%E5%8D%87%E4%B8%AD%E7%9A%84%E6%AD%A3%E5%88%99%E5%8C%96/01-%E6%8F%90%E5%8D%87%E7%AE%97%E6%B3%95%E4%B8%AD%E7%9A%84%E8%BF%87%E6%8B%9F%E5%90%88%E9%97%AE%E9%A2%98.md)
