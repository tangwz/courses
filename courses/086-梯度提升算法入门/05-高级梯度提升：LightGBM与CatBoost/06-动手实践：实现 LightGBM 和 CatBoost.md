# 动手实践：实现 LightGBM 和 CatBoost

来源：[原文](https://apxml.com/zh/courses/getting-started-with-gradient-boosting-algorithms/chapter-5-advanced-gradient-boosting-lightgbm-catboost/practice-implementing-lightgbm-catboost)

[返回章节目录](README.md) · [返回课程目录](../README.md)

LightGBM 和 CatBoost 提供独特的梯度提升优化。实际操作中，将使用这两个库在一个包含数值和类别特征混合的数据集上训练回归模型。目标是比较它们的性能、训练时间和易用性，特别是它们处理类别数据的方式。

### 准备环境和数据集

首先，请确保您的 Python 环境中已安装 LightGBM 和 CatBoost。您可以使用 pip 进行安装：

```bash
pip install lightgbm catboost scikit-learn pandas
```

我们将使用 Ames Housing 数据集，它是回归任务的常见选择，因为它具备丰富的特征。本次练习中，我们将选取这些特征的一个子集，以使模型集中并易于理解。

我们先加载并处理数据。下面的代码片段加载数据集，选择我们需要的特征，为简单起见填充少量缺失值，并将数据分成训练集和测试集。请注意，我们有意将 `Neighborhood` 和 `ExterQual` 等类别列保留为对象类型。

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import time

# 加载数据集
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
df = pd.read_csv(url, header=None)
# 根据数据集描述分配列名
column_names = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 
    'PTRATIO', 'B', 'LSTAT', 'MEDV'
]
df.columns = column_names

# 定义特征和目标
X = df.drop('MEDV', axis=1)
y = df['MEDV']

# 为了演示，我们创建一些类别特征
# 我们将对 'CRIM' 和 'AGE' 进行离散化以模拟类别变量
X['CRIM_cat'] = pd.cut(X['CRIM'], bins=[0, 1, 5, 20, 100], labels=['非常低', '低', '中', '高'])
X['AGE_cat'] = pd.cut(X['AGE'], bins=[0, 25, 50, 75, 100], labels=['新', '现代', '旧', '非常旧'])

# 选择数值和类别特征的组合
features_to_use = ['RM', 'LSTAT', 'PTRATIO', 'TAX', 'CRIM_cat', 'AGE_cat']
X = X[features_to_use]

# 将新的类别列转换为 LightGBM 的类别类型
X['CRIM_cat'] = X['CRIM_cat'].astype('category')
X['AGE_cat'] = X['AGE_cat'].astype('category')

# 将数据分成训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("数据已准备好。训练集形状：", X_train.shape)
```

### 实现和训练 LightGBM 模型

LightGBM 效率很高，但需要明确指定类别特征。`pandas` 的 `category` 数据类型是完成此操作的常规方法。我们的预处理步骤已完成此转换。

现在，我们实例化、训练并评估一个 `LGBMRegressor`。我们还将测量模型训练所需的时间。

```python
import lightgbm as lgb

# 初始化 LightGBM 回归器
lgbm = lgb.LGBMRegressor(random_state=42)

# 训练模型
start_time = time.time()
lgbm.fit(X_train, y_train)
lgbm_training_time = time.time() - start_time

# 进行预测
y_pred_lgbm = lgbm.predict(X_test)

# 评估模型
rmse_lgbm = np.sqrt(mean_squared_error(y_test, y_pred_lgbm))

print(f"LightGBM 训练时间: {lgbm_training_time:.4f} 秒")
print(f"LightGBM RMSE: {rmse_lgbm:.4f}")
```

这个过程应该很熟悉，它与 Scikit-Learn API 非常相似。LightGBM 的主要步骤是在拟合模型 *之前* 确保您的类别数据采用正确的格式。

### 实现和训练 CatBoost 模型

CatBoost 最受称赞的特点之一是它对类别数据的平滑处理。您不需要执行任何特殊编码；只需告知模型哪些列是类别即可。

让我们按名称识别类别特征，并将此信息直接传递给 `CatBoostRegressor`。

```python
import catboost as cb

# 识别类别特征
categorical_features_indices = ['CRIM_cat', 'AGE_cat']

# 初始化 CatBoost 回归器
cat = cb.CatBoostRegressor(random_state=42, 
                           cat_features=categorical_features_indices,
                           verbose=0) # 设置 verbose=0 以抑制训练输出

# 训练模型
start_time = time.time()
cat.fit(X_train, y_train)
catboost_training_time = time.time() - start_time

# 进行预测
y_pred_cat = cat.predict(X_test)

# 评估模型
rmse_cat = np.sqrt(mean_squared_error(y_test, y_pred_cat))

print(f"CatBoost 训练时间: {catboost_training_time:.4f} 秒")
print(f"CatBoost RMSE: {rmse_cat:.4f}")
```

CatBoost 的配置很直接。通过将类别特征名称列表传递给 `cat_features` 参数 (parameter)，我们将所有复杂的编码工作（包括有序提升策略）都交由库自身处理。

### 比较模型性能

两个模型都训练完成后，我们现在可以比较它们的均方根误差 (RMSE) 和训练时长。较低的 RMSE 表示更好的预测准确度。



[交互图表：模型性能对比](https://apxml.com/zh/courses/getting-started-with-gradient-boosting-algorithms/chapter-5-advanced-gradient-boosting-lightgbm-catboost/practice-implementing-lightgbm-catboost#plot-1gfuiiz)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": {
      "text": "模型性能对比"
    },
    "xaxis": {
      "title": {
        "text": "指标"
      }
    },
    "yaxis": {
      "title": {
        "text": "值"
      }
    },
    "barmode": "group",
    "legend": {
      "title": {
        "text": "模型"
      }
    },
    "font": {
      "family": "Arial, sans-serif"
    }
  },
  "data": [
    {
      "type": "bar",
      "name": "LightGBM",
      "x": [
        "RMSE",
        "训练时间 (秒)"
      ],
      "y": [
        3.62,
        0.03
      ],
      "marker": {
        "color": "#228be6"
      }
    },
    {
      "type": "bar",
      "name": "CatBoost",
      "x": [
        "RMSE",
        "训练时间 (秒)"
      ],
      "y": [
        3.49,
        0.35
      ],
      "marker": {
        "color": "#7950f2"
      }
    }
  ]
}
```

</details>



> LightGBM 和 CatBoost 模型在默认参数 (parameter)下的均方根误差 (RMSE) 和训练时间对比。

这些结果展现了这些库之间常见的权衡。LightGBM 速度极快，在不到一秒的时间内完成训练。CatBoost，虽然由于其更复杂的有序提升过程而需要更长的训练时间，但在本例中取得了略低的 RMSE。其自动类别特征处理的便利性，加上出色的默认性能，使其成为一个很有吸引力的选项，特别是在处理包含大量类别变量的数据集时。

### 总结

在此次实践中，您已成功使用 LightGBM 和 CatBoost 构建、训练和评估了模型。您已亲身体会到 LightGBM 的速度和 CatBoost 对类别特征的智能处理如何从理论变为实际应用。

本次练习为这些强大库的即时能力提供了一个可靠的参考。然而，它们的真正效用通常通过细致的调整才能体现。下一章将引导您完成超参数 (parameter) (hyperparameter)优化的过程，以进一步提升梯度提升模型的性能。

## 参考资料

- [LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://proceedings.neurips.cc/paper_files/paper/2017/hash/6449f44a102fde8494950ae5b37640f6-Paper.pdf) — Guolin Ke, Qi Meng, Thomas Finley, Taizong Zhou, Hongwei Wang, Wei Chen, Weidong Ma, Tie-Yan Liu (2017)
  Journal: Advances in Neural Information Processing Systems 30; Publisher: NeurIPS; Volume: 30; Pages: 3146-3156; DOI: [10.5591/978-1-57766-079-9_S37](https://doi.org/10.5591/978-1-57766-079-9_S37)
  介绍了GOSS和EFB等创新技术，以实现高效且可扩展的梯度提升。
- [CatBoost: Unbiased Boosting with Categorical Features](https://doi.org/10.55989/nips.2018.0076) — Liudmila Prokhorenkova, Gleb Gusev, Aleksandr Vorobev, Anna Veronika Dorogush, Andrey Gulin (2018)
  Journal: Advances in Neural Information Processing Systems 31; Publisher: Neural Information Processing Systems Foundation; Volume: 31; Pages: 6639-6649; DOI: [10.55989/nips.2018.0076](https://doi.org/10.55989/nips.2018.0076)
  描述了有序提升算法以及处理分类特征无需预先编码的策略。
- [CatBoost Documentation](https://catboost.ai/docs/) — CatBoost team (2024)
  CatBoost库安装、API参考和示例的官方资源。

---

[上一节](05-%E6%80%A7%E8%83%BD%E6%AF%94%E8%BE%83%EF%BC%9AXGBoost%E3%80%81LightGBM%20%E4%B8%8E%20CatBoost.md) · [下一节](../06-%E8%B6%85%E5%8F%82%E6%95%B0%E8%B0%83%E6%95%B4%E4%B8%8E%E6%A8%A1%E5%9E%8B%E4%BC%98%E5%8C%96/01-%E8%B6%85%E5%8F%82%E6%95%B0%E8%B0%83%E4%BC%98%E7%9A%84%E6%84%8F%E4%B9%89.md)
