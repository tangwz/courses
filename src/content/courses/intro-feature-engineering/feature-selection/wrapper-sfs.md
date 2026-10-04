---
course: "intro-feature-engineering"
chapter: "feature-selection"
lesson: "wrapper-sfs"
sourceId: 1412
sourceUrl: "https://apxml.com/zh/courses/intro-feature-engineering/chapter-6-feature-selection/wrapper-sfs"
title: "封装方法：序列特征选择 (SFS)"
description: "应用前向或后向序列特征选择算法。"
order: 8
plots: []
sourceHash: "dcda74e60eb64fad027695e9e3d24b2b08cadd61a88cd8da2e4fbe26f7373116"
sourceCorrections: []
---

递归特征消除 (RFE) 的工作方式是，从所有特征开始并移除最不重要的那些，而**序列特征选择 (SFS)** 提供替代的贪婪搜索策略。SFS 不像 RFE 那样仅仅依赖模型系数或特征重要性，而是直接在不同的特征子集上评估模型性能（使用选定的评分指标）。它迭代地构建（前向选择）或缩减（后向选择）特征集。

SFS 方法属于封装类别，因为它们将特征选择过程封装在特定的机器学习 (machine learning)模型周围，并使用其性能作为目标函数来引导搜索。这使得它们比过滤方法计算成本更高，但可能更符合所选模型的需要。

### 前向选择

序列前向选择 (SFS) 从一个空特征集开始。在每次迭代中，它评估添加当前不在已选集中的每个特征。添加后能带来最高性能提升（根据选定的评分指标，通常通过交叉验证评估）的特征被添加到集合中。这个过程持续进行，直到选择了预定数量的特征，或者直到添加任何剩余特征都未能带来显著的性能提升。

**算法步骤（前向选择）：**

1. 从空特征集 $S = \emptyset$ 开始。
2. 指定目标特征数量 $k$。
3. 当 $|S| < k$ 时：
   - 对于每个不在 $S$ 中的特征 $f$：
     - 使用特征 $S \cup \{f\}$ 评估所选估计器的性能。
   - 选择能带来最佳性能的特征 $f^*$。
   - 更新已选集：$S = S \cup \{f^*\}$。
4. 返回最终特征集 $S$。

想象一下构建一个工具箱。你从零开始，一次添加一个工具，总是选择与现有工具配合使用时能最好地完成任务的工具。

### 后向选择

序列后向选择 (SBS)，有时也称为序列后向消除，以相反方向操作。它从所有可用特征的完整集合开始。在每次迭代中，它评估移除当前集合中的每个特征。移除后导致模型性能下降最小（或提升最大）的特征被移除。这个过程持续进行，直到达到所需数量的特征。

**算法步骤（后向选择）：**

1. 从完整的特征集 $S = F$ 开始（其中 $F$ 是所有特征的集合）。
2. 指定目标特征数量 $k$。
3. 当 $|S| > k$ 时：
   - 对于当前在 $S$ 中的每个特征 $f$：
     - 使用特征 $S \setminus \{f\}$ 评估所选估计器的性能。
   - 选择移除后能带来最佳性能（或最少性能下降）的特征 $f^*$。
   - 更新已选集：$S = S \setminus \{f^*\}$。
4. 返回最终特征集 $S$。

这就像从一个杂乱的工具箱开始，一次移除一个工具，丢弃你最不怀念的那个，直到你拥有一个精简且有效的集合。

### 使用 Scikit-learn 实现

Scikit-learn 提供了 `SequentialFeatureSelector` 类用于执行前向和后向选择。

```python
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 生成合成分类数据
X, y = make_classification(n_samples=200, n_features=15, n_informative=5,
                           n_redundant=5, n_repeated=0, n_classes=2,
                           n_clusters_per_class=2, random_state=42)
X = pd.DataFrame(X, columns=[f'feature_{i+1}' for i in range(15)])

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建包含缩放和逻辑回归的管道
# SFS 对于许多估计器来说，在缩放数据上表现更好
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(solver='liblinear', random_state=42))
])

# --- 前向选择 ---
print("正在执行前向选择...")
sfs_forward = SequentialFeatureSelector(
    estimator=pipe.named_steps['model'], # 使用管道中的模型部分
    n_features_to_select=5,        # 目标特征数量
    direction='forward',           # 指定前向选择
    scoring='accuracy',            # 性能指标
    cv=5,                          # 交叉验证折叠数
    n_jobs=-1                      # 使用所有可用的 CPU 核心
)

# 注意：SFS 理想情况下应在缩放数据上拟合。
# 我们先缩放数据，然后使用缩放后的数据和基础模型拟合 SFS。
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

sfs_forward.fit(X_train_scaled, y_train)

# 获取选定的特征索引和名称
selected_features_mask_fwd = sfs_forward.get_support()
selected_feature_names_fwd = X.columns[selected_features_mask_fwd]
print(f"选定特征（前向）：{selected_feature_names_fwd.tolist()}")
print(f"选定特征数量：{sfs_forward.n_features_to_select_}")


# --- 后向选择 ---
print("\n正在执行后向选择...")
sfs_backward = SequentialFeatureSelector(
    estimator=pipe.named_steps['model'],
    n_features_to_select=5,        # 目标特征数量
    direction='backward',          # 指定后向选择
    scoring='accuracy',
    cv=5,
    n_jobs=-1
)

sfs_backward.fit(X_train_scaled, y_train)

# 获取选定的特征索引和名称
selected_features_mask_bwd = sfs_backward.get_support()
selected_feature_names_bwd = X.columns[selected_features_mask_bwd]
print(f"选定特征（后向）：{selected_feature_names_bwd.tolist()}")
print(f"选定特征数量：{sfs_backward.n_features_to_select_}")

# 您可以转换数据以仅保留选定特征
# X_train_scaled_sfs_fwd = sfs_forward.transform(X_train_scaled)
# X_test_scaled_sfs_fwd = sfs_forward.transform(X_test_scaled)
```

`SequentialFeatureSelector` 的重要参数 (parameter)：

- `estimator`：用于评估特征子集的机器学习 (machine learning)模型。
- `n_features_to_select`：目标特征数量。可以是整数、'auto'（使用 `tol` 参数），或介于 0 和 1 之间的浮点数（表示特征比例）。使用 'auto' 通常计算成本高昂。
- `direction`：`'forward'` 或 `'backward'`。
- `scoring`：用于评估性能的指标（例如，'accuracy'、'roc\_auc'、'r2'、'neg\_mean\_squared\_error'）。必须是有效的 Scikit-learn 评分字符串或可调用评分器。
- `cv`：交叉验证折叠数或交叉验证分割策略。对性能评估必不可少。
- `n_jobs`：在交叉验证期间用于并行执行的 CPU 核心数。`-1` 表示使用所有可用核心。

注意，前向选择和后向选择不一定产生相同的特征集，因为它们以不同的方式查看特征空间。

### 优点和缺点

**优点：**

- **模型特定性：** 通过所选模型的视角考虑特征之间的相互影响，与过滤方法相比，可能为该特定模型找到更好的子集。
- **简单：** 贪婪方法相对容易理解。
- **可能优于 RFE：** RFE 通常依赖于可能随特征移除而变化的系数/重要性，而 SFS 在每一步都基于实际模型性能重新评估子集。

**缺点：**

- **计算成本高昂：** 需要在每次迭代中针对添加或移除的每个特征多次训练估计器，并乘以交叉验证折叠数。如果从大量特征开始，后向选择可能会特别慢。
- **贪婪性质：** 在每一步都做出局部最优选择，这不保证找到全局最佳特征子集。过早添加（前向）或保留（后向）的特征可能会妨碍后续找到更好的组合。
- **敏感性：** 结果显著依赖于所选的估计器、评分指标和交叉验证策略。

### 何时使用 SFS

SFS 是一种有价值的封装方法，当出现以下情况时：

- 你怀疑特征交互对你选择的模型很重要。
- 计算成本可接受（例如，特征数量适中）。
- 你希望基于特定的性能指标直接优化特征选择。
- 你需要与 RFE 或嵌入 (embedding)方法相比的另一种视角。

像其他封装方法一样，SFS 需要仔细考虑潜在的模型性能提升与所需的计算资源之间的权衡。通常最好将其结果与过滤方法和嵌入方法的结果进行比较。

## 参考资料

- [sklearn.feature_selection.SequentialFeatureSelector](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SequentialFeatureSelector.html) — Scikit-learn developers (2023)
  Scikit-learn中顺序特征选择器的官方文档，包含参数和使用示例。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, Jerome Friedman (2009)
  Publisher: Springer; DOI: [10.1007/978-0-387-84858-7](https://doi.org/10.1007/978-0-387-84858-7)
  一本统计学习方法的经典书籍，包含特征选择和模型评估的基础概念。
- [Feature Engineering and Selection: A Practical Approach for Developers and Data Scientists](https://www.crcpress.com/Feature-Engineering-and-Selection-A-Practical-Approach-for-Developers-and/Kuhn-Johnson/p/book/9781032128711) — Max Kuhn, Kjell Johnson (2019)
  Publisher: Chapman and Hall/CRC; DOI: [10.1201/9781351111977](https://doi.org/10.1201/9781351111977)
  一本侧重于特征工程和选择技术的实用指南，包含对顺序特征选择等封装方法的详细讨论。
