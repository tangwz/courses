---
course: "mastering-gradient-boosting-algorithms"
chapter: "gradient-boosting-specialized-tasks"
lesson: "multi-output-gradient-boosting"
sourceId: 2076
sourceUrl: "https://apxml.com/zh/courses/mastering-gradient-boosting-algorithms/chapter-9-gradient-boosting-specialized-tasks/multi-output-gradient-boosting"
title: "多输出梯度提升"
description: "处理多个目标变量问题的策略。"
order: 7
plots: []
sourceHash: "57ac05f43076a6f48de5cf9836256bf111668b86af5011d299d7011c106a9d50"
sourceCorrections: []
---

标准梯度提升在预测单个目标变量方面表现良好，但许多问题需要根据同一组输入特征同时预测多个输出。这被称为多输出回归（预测多个连续变量）或多输出分类（预测多个类别或二元标签）。例如，您可能希望根据传感器读数预测温度、湿度和气压（三个回归目标），或者根据多个属性对图像进行分类（例如，包含猫、包含狗、在户外——三个二元分类目标）。

梯度提升算法，如XGBoost、LightGBM、CatBoost以及Scikit-learn的`GradientBoostingRegressor`/`Classifier`等库中实现的，通常设计用于优化单个目标变量$y$。它们期望目标变量是一个一维数组或向量 (vector)。当遇到目标$Y$是矩阵（每列代表一个不同输出）的多输出情况时，这些模型无法直接以其标准配置使用。

不过，您可以通过几种实用方法来使梯度提升适应这些任务。

### 方法一：独立模型（每个输出一个模型）

最直接的方式是将多输出问题视为一系列独立的单输出问题。您为每个目标变量训练一个单独的梯度提升模型。

- **对于多输出回归：** 如果您有$k$个目标变量（$Y_1, Y_2, ..., Y_k$），您将训练$k$个独立的回归模型。模型$j$使用输入特征$X$进行训练，以预测目标$Y_j$。
- **对于多输出分类：** 类似地，如果您有$k$个目标标签，您将训练$k$个独立的分类模型。模型$j$使用$X$进行训练，以预测标签$Y_j$。

**优点：**

- 简单：使用标准提升库，无需修改即可轻松理解和实现。
- 灵活：如果需要，您可以为每个输出模型独立调整超参数 (parameter) (hyperparameter)，可能提升特定目标的性能。

**缺点：**

- 忽略关联：此方法不明确地建模目标变量之间的任何潜在关联或依赖关系。如果输出高度关联，联合建模可能会产生更好结果。
- 计算成本：训练$k$个单独的模型计算成本可能较高，特别是当$k$很大或数据集庞大时。预测也需要运行$k$个模型。

以下是一个使用XGBoost的Python代码片段：

```python
import xgboost as xgb
import numpy as np

# 假设 X_train, Y_train (形状为 样本数, 输出数)
# 假设 X_test
k_outputs = Y_train.shape[1]
models = []
Y_pred_test = np.zeros((X_test.shape[0], k_outputs))

for i in range(k_outputs):
    print(f"正在训练输出 {i+1} 的模型...")
    # 为每个输出创建一个新的 XGBoost 模型
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42+i) # 示例参数
    
    # 使用 X_train 和 Y_train 的第 i 列进行训练
    model.fit(X_train, Y_train[:, i])
    
    # 存储训练好的模型
    models.append(model)
    
    # 对该输出在测试集上进行预测
    Y_pred_test[:, i] = model.predict(X_test)

# Y_pred_test 现在包含所有 k 个输出的预测
```

### 方法二：使用 Scikit-learn 多输出包装器

Scikit-learn 提供便捷的元估计器（包装器），可以自动化为每个目标拟合一个估计器的过程。这些是`MultiOutputRegressor`和`MultiOutputClassifier`。您可以包装任何与Scikit-learn兼容的单输出估计器，包括XGBoost、LightGBM或CatBoost模型（使用它们的Scikit-learn API）。

在内部，这些包装器本质上实现了方法一：它们克隆基础估计器并为每个目标变量拟合一个克隆。

**优点：**

- 便捷：与手动循环和管理多个模型相比，简化了代码。与Scikit-learn生态系统（例如，管道、交叉验证）很好地结合。

**缺点：**

- 与方法一相同：内部仍训练独立模型，忽略目标关联，并可能带来高计算成本。

以下是如何使用`MultiOutputRegressor`与LightGBM的示例：

```python
import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor
import numpy as np

# 假设 X_train, Y_train (形状为 样本数, 输出数)
# 假设 X_test
k_outputs = Y_train.shape[1]

# 定义基础的单输出估计器
lgbm = lgb.LGBMRegressor(objective='regression_l1', n_estimators=100, random_state=42) # 示例参数

# 创建 MultiOutputRegressor 包装器
multi_output_model = MultiOutputRegressor(estimator=lgbm, n_jobs=-1) # 使用所有可用核心

# 训练包装器模型
# 它将在内部为 Y_train 的每一列拟合一个 LGBMRegressor
multi_output_model.fit(X_train, Y_train)

# 在测试集上进行预测
# 返回形状为 (样本数, 输出数) 的预测结果
Y_pred_test = multi_output_model.predict(X_test)

# 如果需要，可以访问各个估计器
# individual_estimators = multi_output_model.estimators_
```

下图展示了这两种常见方法的核心思想。方法一涉及手动管理模型，而方法二使用内部处理此事的包装器。两者都为每个输出生成独立模型。

> 该图对比了训练独立模型的手动方式（方法一）与使用Scikit-learn包装器（如`MultiOutputRegressor`）的方式（方法二），后者在内部自动拟合独立的模型克隆。

### 方法三：原生多输出梯度提升（高级/研究）

一种理论上更复杂的方式涉及修改梯度提升算法本身，以直接处理多个输出。这可能意味着：

1. **多输出损失函数 (loss function)：** 定义一个能同时计算所有输出误差的损失函数。梯度计算将涉及对所有输出的梯度。
2. **多目标分裂准则：** 修改树构建过程，使得分裂的选择基于其减少所有输出综合误差的能力，可能捕获关联。

尽管该领域存在研究，但XGBoost、LightGBM和CatBoost中的标准实现目前不提供内置的、通用型的原生多输出功能。实现这样的系统将需要对底层C++或CUDA代码进行大量定制，或者可能需要查找专门的库或研究实现。对于大多数实践者来说，方法一和方法二是标准且推荐的做法。

### 选择合适的方法

对于大多数使用梯度提升的多输出问题，选择在于手动管理独立模型（方法一）或使用Scikit-learn包装器（方法二）。

- 使用\*\*方法二（包装器）\*\*以求便捷并与Scikit-learn管道结合，特别是当输出数量可控且您不需要针对每个输出进行高度定制化调优时。
- 如果需要对每个特定输出的训练过程或超参数 (parameter) (hyperparameter)调优进行精细控制，或者您未使用提升库的Scikit-learn API，则可考虑**方法一（手动模型）**。

如果输出数量非常大，请注意计算方面的影响。在这种情况下，可以首先尝试对输出空间进行降维处理，或者考虑专门设计用于多标签/多输出任务的模型（这些模型可能并非基于梯度提升），这可能是不错的替代方案。

## 参考资料

- [sklearn.multioutput.MultiOutputRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputRegressor.html) — scikit-learn developers (2024)
  Publisher: scikit-learn
  提供了Scikit-learn中多输出回归封装器的官方文档，与每个输出独立模型的实际实现直接相关。
- [XGBoost: A Scalable Tree Boosting System](https://doi.org/10.1145/2939672.2939785) — Tianqi Chen, Carlos Guestrin (2016)
  Journal: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining; Publisher: ACM; Pages: 785-794; DOI: [10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785)
  介绍了高效且可扩展的XGBoost算法，展示了其针对单输出预测任务的设计和优化，阐明了在多输出场景中进行适应的必要性。
- [Multi-Output Regression with Ensembles of Extremely Randomized Trees](https://dl.acm.org/doi/10.1007/s10994-006-0036-7) — Pierre Geurts, Louis Wehenkel, and Damien Ernst (2006)
  Journal: Machine Learning; Publisher: Springer US; Volume: 65; Pages: 135-156; DOI: [10.1007/s10994-006-0036-7](https://doi.org/10.1007/s10994-006-0036-7)
  探讨了使用集成树进行多输出回归的方法，比较了独立模型与考虑输出相关性的方法，这与原生多输出建模的高级策略相关。
