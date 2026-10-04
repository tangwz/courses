---
course: "applied-data-science"
chapter: "building-tuning-predictive-models"
lesson: "cross-validation-strategies"
sourceId: 1871
sourceUrl: "https://apxml.com/zh/courses/applied-data-science/chapter-3-building-tuning-predictive-models/cross-validation-strategies"
title: "交叉验证策略"
description: "实现K折及其他交叉验证方法以进行可靠的表现估计。"
order: 7
plots: []
sourceHash: "d6ce8def7f0e1f953499b79a10557454087689cea53ae09c96bef62596c6fc29"
sourceCorrections: []
---

在上一节中，我们考察了多种评估模型表现的指标。然而，即使使用精密的指标，仅在*一次*训练-测试分割上评估模型有时也可能具有误导性。恰好落在测试集中的特定数据点会极大地影响计算出的表现得分。如果分割“幸运”，模型可能看起来比实际更好；“不幸运”的分割可能会不公正地惩罚一个好模型。我们需要一种更可靠的方式来评估模型在新数据上的泛化能力。交叉验证正是在此发挥作用。

交叉验证通过系统地使用数据的不同子集进行训练和测试，提供更稳定、更值得信赖的模型表现估计。我们不只进行一次分割，而是进行多次分割并平均结果。

### 单次分割的局限性

设想您将数据按80%用于训练、20%用于测试进行分割。您训练模型，在测试集上评估，得到85%的准确率。您对这个数字有多大信心？如果您重新打乱数据并创建*不同的*80/20分割，您可能会得到82%或88%的准确率。这种变异性出现的原因是表现对测试集选取的特定样本很敏感。交叉验证方法旨在降低这种方差，让我们更好地了解模型在新数据上的*平均*表现。

### K折交叉验证

最常用的交叉验证策略是**K折交叉验证**。其工作原理如下：

1. **分割：** 整个数据集被随机分成$K$个大小相等（或近似相等）的子集，通常称为“折”。$K$的常见选择是5或10。
2. **迭代：** 流程迭代$K$次。在每次迭代$i$（从1到$K$）中：
   - 第$i$折被作为**测试集**（验证集）保留。
   - 其余的$K-1$折组合形成**训练集**。
   - 模型在训练集上训练，并在测试集（第$i$折）上评估。记录本次迭代的评估分数。
3. **汇总：** $K$次迭代后，我们得到$K$个评估分数。最终的交叉验证表现指标通常是这$K$个分数的平均值。

> 5折交叉验证的视觉表示。数据集被分成5个折叠。在每次迭代中，一个折叠用作测试集，而其他折叠构成训练集。表现将对所有迭代进行平均。

**选择 K：**

- 较高的$K$表示每次迭代有更多的训练数据（有利于模型表现）和更多的迭代次数（计算量更大）。它会导致泛化表现的估计偏差较低，但由于训练集在不同折叠之间非常相似，可能会有更高的方差。
- 较低的$K$意味着计算成本更低，但可能产生更高的偏差，因为每次折叠中用于训练的数据较少。
- $K=5$或$K=10$被广泛采用，因为它们通常能在计算成本和可靠的表现估计之间提供良好的平衡。

在scikit-learn中，您可以使用`sklearn.model_selection`模块中的`cross_val_score`函数或`KFold`类来实现K折交叉验证。

```python
# 使用scikit-learn的例子
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
import numpy as np

# 假设X（特征）和y（目标）是已定义的NumPy数组或Pandas DataFrame/Series
# 示例占位数据
X = np.random.rand(100, 10)
y = np.random.randint(0, 2, 100)

model = LogisticRegression()
# 定义K折策略（例如，5折）
# shuffle=True 确保在分割前随机打乱
# random_state 确保结果可复现
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# 执行交叉验证
# 'scoring' 可以是 'accuracy', 'roc_auc', 'f1' 等
scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')

print(f"各折分数: {scores}")
print(f"平均交叉验证准确率: {np.mean(scores):.4f}")
print(f"交叉验证准确率的标准差: {np.std(scores):.4f}")
```

输出显示了5个折叠中每个折叠的准确率分数以及平均准确率，这比单次训练-测试分割提供了更好的估计。标准差则提示了分数在不同折叠间的变异性。

### 分层K折交叉验证

在处理分类问题时，特别是那些类别不平衡（一个类别的出现频率远高于其他类别）的问题，标准K折可能会导致某些折叠中的类别分布出现偏差。例如，一个折叠可能偶然地包含非常少甚至没有少数类样本。这会严重影响训练过程或导致误导性的评估分数。

**分层K折**通过确保每个折叠中各类别样本的百分比与完整数据集中保持一致来解决这个问题。如果您的数据集包含80%的A类和20%的B类，分层K折会确保每个折叠也包含（大约）80%的A类和20%的B类样本。

这是分类任务的首选方法，因为它确保模型在代表整体类别分布的折叠上进行训练和评估。

在scikit-learn中，您可以使用`StratifiedKFold`类，或者在执行分类时，直接将整数值传递给`cross_val_score`的`cv`参数 (parameter)；scikit-learn通常会为分类估计器默认使用分层K折。

```python
# 使用scikit-learn进行分层K折的例子
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# 假设X和y已定义，y可能不平衡
# 示例占位的不平衡数据
X = np.random.rand(100, 10)
y = np.array([0]*80 + [1]*20) # 80% 类别 0, 20% 类别 1
np.random.shuffle(y) # 打乱以混合类别

model = RandomForestClassifier(random_state=42)

# 定义分层K折策略（例如，5折）
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 使用分层K折执行交叉验证
# 对于分类，cross_val_score通常默认使用StratifiedKFold
# 如果传递了y，但明确指定它会更清晰。
scores = cross_val_score(model, X, y, cv=stratified_kfold, scoring='roc_auc') # 使用ROC AUC

print(f"各折分层分数: {scores}")
print(f"平均交叉验证ROC AUC: {np.mean(scores):.4f}")
print(f"交叉验证ROC AUC的标准差: {np.std(scores):.4f}")
```

### 其他交叉验证策略

虽然K折和分层K折是最常用的，但也存在其他策略：

- **留一法交叉验证 (LOOCV)：** 这种K折交叉验证中，$K$等于数据集中的样本数量($N$)。在每次迭代中，一个样本被用作测试集，其余$N-1$个样本用于训练。它的计算成本非常高（$N$次迭代），并且可能产生测试误差的高方差估计，但对于非常小的数据集而言可能有用。
- **随机打乱分割 (Shuffle Split)：** 这种策略创建指定数量的独立训练/测试分割。在每次分割中，数据首先被随机打乱，然后选择一定比例（例如20%）作为测试集。与K折不同，样本可以在不同迭代中多次出现在测试集中，并且折叠不是互斥的。`ShuffleSplit`和`StratifiedShuffleSplit`（后者保持类别比例）提供了在独立控制迭代次数和测试集大小方面的灵活性。

### 交叉验证用于超参数 (parameter) (hyperparameter)调优

交叉验证在超参数调优方法中发挥着重要作用，例如我们之前讨论过的网格搜索和随机搜索。当使用scikit-learn的`GridSearchCV`或`RandomizedSearchCV`等工具时，流程如下：

1. **外层循环：** 搜索方法提出超参数的组合（例如，SVM的`C=1.0`，`kernel='rbf'`）。
2. **内层循环（交叉验证）：** 为了评估*此特定组合*，训练数据使用K折（或分层K折）进行分割。使用所选超参数的模型被训练和评估$K$次。
3. **汇总分数：** 计算该超参数组合在$K$个折叠上的平均表现分数。
4. **重复：** 步骤1-3针对网格中定义的所有超参数组合（网格搜索）或固定数量的随机组合（随机搜索）重复进行。
5. **选择最佳：** 产生最佳平均交叉验证分数的超参数组合被选为最优参数集。
6. **最终训练：** 最后，具有这些最佳超参数的模型通常会在*整个*原始训练数据集上重新训练（而不仅仅是交叉验证期间使用的K-1个折叠），以在最终的保留测试集上评估之前最大化使用数据量。

这确保了超参数是基于泛化表现的稳定估计来选择的，而不是基于单次验证分割的表现。

通过采用交叉验证，我们从可能不可靠的单次分割评估转向我们的模型在新数据上可能表现的估计。这对于建立我们模型选择和超参数调优过程的信心是不可或缺的。

## 参考资料

- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://hastie.su.domains/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer
  这本经典教科书为交叉验证和其他机器学习技术提供了全面的统计和理论基础。
- [Cross-validation: evaluating estimator performance](https://scikit-learn.org/stable/modules/cross_validation.html) — scikit-learn developers (2023)
  scikit-learn 官方文档提供了在 Python 中实现各种交叉验证策略的实用指南、示例和详细解释。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGid2_a_kLjr3hqm8OszrdImfyNhmL9g0s8Mns72enJnNydYrS-aOilCsFm6RjGuN9fU1i2G91JdGE9nF-HM7BxiJMxEAoFFYBBEfVL6jzF8l0sxrBbMvQL5Ydn_GULSPI90hb_n0wOD-jJyeqCnrRMacBlUXkcBe_fyzYSJ6wwYpGq0tNKZCr-1m111s05YY5zGOPEEWFau5QFVA==) — Aurélien Géron (2022)
  Publisher: O'Reilly Media, Inc.; Pages: 864
  这本广受认可的书籍提供了使用流行 Python 库实现机器学习概念（包括交叉验证）的实用、代码驱动的示例。
