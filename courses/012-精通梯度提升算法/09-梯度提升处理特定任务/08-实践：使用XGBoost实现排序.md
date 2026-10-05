# 实践：使用XGBoost实现排序

来源：[原文](https://apxml.com/zh/courses/mastering-gradient-boosting-algorithms/chapter-9-gradient-boosting-specialized-tasks/practice-implementing-ranking-xgboost)

[返回章节目录](README.md) · [返回课程目录](../README.md)

目标是训练一个模型，它在接收到一个查询时，能预测文档（或项目）的分数，使得按这些分数排序的结果接近真实的关联性排序。我们将采用 `rank:pairwise` 目标函数，它为了减少同一查询组内文档对的错误排序数量。

### 设置环境和数据

首先，请确保已安装所需的库：

```bash
pip install xgboost numpy pandas scikit-learn
```

对于LTR，我们的数据需要有特定的结构：

- 每个文档/项目的特征。
- 每个文档/项目的关联性分数（标签）。通常，这些是分级的关联性级别（例如，0=不相关，1=有点相关，2=高度相关）。
- 一个查询ID (`qid`)，用于将同一查询检索到的文档分组。排序目标是在这些组内运作。

让我们模拟一个小型数据集，表示不同查询的搜索结果。

```python
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore", category=UserWarning) # 抑制XGBoost警告，仅为演示目的

# 模拟LTR数据
np.random.seed(42)
n_queries = 10
n_docs_per_query = 15
n_features = 5

# 生成特征
X = np.random.rand(n_queries * n_docs_per_query, n_features)

# 生成查询ID
qids = np.repeat(np.arange(n_queries), n_docs_per_query)

# 生成关联性分数（关联性越高与第一个特征越相关）
# 模拟不完美的关联性并添加噪声
base_relevance = X[:, 0] * 2 + np.random.randn(X.shape[0]) * 0.5
# 根据每个查询内的分位数分配离散的关联性级别（例如，0、1、2）
y = np.zeros_like(base_relevance, dtype=int)
for qid in range(n_queries):
    query_mask = (qids == qid)
    query_relevance = base_relevance[query_mask]
    # 根据查询内的关联性分位数分配标签
    q_75 = np.percentile(query_relevance, 75)
    q_25 = np.percentile(query_relevance, 25)
    y[query_mask & (query_relevance >= q_75)] = 2 # 高度相关
    y[query_mask & (query_relevance >= q_25) & (query_relevance < q_75)] = 1 # 略微相关
    # 其他保持为0（不相关）

# 创建DataFrame
df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
df['qid'] = qids
df['relevance'] = y

print("模拟数据集头部：")
print(df.head())
print("\n数据集信息：")
df.info()
print("\n关联性分布：")
print(df['relevance'].value_counts())
```

这就得到了一个包含特征、查询ID (`qid`) 和关联性分数 (`relevance`) 的DataFrame `df`。

### 为XGBoost LTR准备数据

XGBoost的排序目标函数要求了解每个查询组的大小。我们还需要在分割数据时，确保来自同一查询的文档保留在同一分割中（无论是训练集还是测试集）。scikit-learn中的 `GroupKFold` 适用于此。

```python
# --- 遵循分组原则的数据分割 ---
# 我们将使用GroupKFold来获取一个分割的索引
gkf = GroupKFold(n_splits=5) # 使用5个分割，取第一个用于训练/测试
train_idx, test_idx = next(gkf.split(df, groups=df['qid']))

X_train, X_test = df.iloc[train_idx].drop(['qid', 'relevance'], axis=1), df.iloc[test_idx].drop(['qid', 'relevance'], axis=1)
y_train, y_test = df.iloc[train_idx]['relevance'], df.iloc[test_idx]['relevance']
qids_train, qids_test = df.iloc[train_idx]['qid'], df.iloc[test_idx]['qid']

# --- 特征缩放（可选但通常是好的做法）---
# 仅基于训练数据缩放特征
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 计算分组大小 ---
# XGBoost需要知道每个组的大小（每个查询的文档数量）
# 首先按qid排序数据，以确保组是连续的
train_order = np.argsort(qids_train.values)
X_train_scaled = X_train_scaled[train_order]
y_train = y_train.iloc[train_order]
qids_train = qids_train.iloc[train_order]
group_train = qids_train.value_counts().sort_index().values

test_order = np.argsort(qids_test.values)
X_test_scaled = X_test_scaled[test_order]
y_test = y_test.iloc[test_order]
qids_test = qids_test.iloc[test_order]
group_test = qids_test.value_counts().sort_index().values

# --- 创建DMatrix ---
# 特殊的DMatrix用于向XGBoost传递数据和组信息
dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
dtrain.set_group(group_train)

dtest = xgb.DMatrix(X_test_scaled, label=y_test)
dtest.set_group(group_test)

print(f"\n训练集：{len(X_train)}个样本，{len(group_train)}个查询")
print(f"测试集：{len(X_test)}个样本，{len(group_test)}个查询")
print(f"训练组大小（前5个）：{group_train[:5]}")
print(f"测试组大小（前5个）：{group_test[:5]}")
```

这里的重要步骤：

1. 我们使用 `GroupKFold` 来确保给定 `qid` 的所有文档都在训练集或测试集中，从而防止数据泄露。
2. 特征仅基于训练数据使用 `StandardScaler` 进行了缩放。
3. 关键的是，我们在按 `qid` 排序数据后计算了每个查询组的大小（`group_train`，`group_test`）。这个数组告诉XGBoost第一个查询、第二个查询等有多少文档。
4. 我们创建了 `xgb.DMatrix` 对象，传入特征矩阵和标签。LTR 的必要步骤是调用 `dtrain.set_group(group_train)` 和 `dtest.set_group(group_test)`。

### 训练XGBoost排序模型

现在，我们配置并训练XGBoost模型。我们将目标函数设置为 `rank:pairwise`，并使用 `ndcg@k` (K截断归一化 (normalization)折损累积增益) 作为评估指标。NDCG 通过将其与基于真实关联性分数的理想排序进行比较，来衡量排序的质量。

```python
# --- LTR的XGBoost参数 ---
params = {
    'objective': 'rank:pairwise',  # 成对排序目标函数
    'eval_metric': ['ndcg@5', 'ndcg@10'], # 使用NDCG在截断点5和10处进行评估
    'eta': 0.1,                    # 学习率
    'gamma': 1.0,                  # 分割所需的最小损失减少量
    'min_child_weight': 1,         # 子节点中所需的最小实例权重和
    'max_depth': 4,                # 最大树深度
    'seed': 42
}

num_boost_round = 100  # 提升轮次数量
evals = [(dtrain, 'train'), (dtest, 'test')] # 训练期间用于评估的数据集

# --- 训练模型 ---
print("\n正在训练XGBoost LTR模型...")
bst = xgb.train(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    evals=evals,
    verbose_eval=20 # 每20轮打印评估结果
)
```

训练输出显示，训练集和测试集上的NDCG分数在提升轮次中有所提升。我们监控测试集性能（`ndcg@5-test`，`ndcg@10-test`）来评估模型的泛化能力。

### 预测与评估

训练好的模型 (`bst`) 会为每个文档预测一个分数。分数越高表示预测的关联性越高。为了评估性能，我们需要：

1. 获取测试集的预测分数。
2. 对于测试集中的每个查询，根据这些预测分数对文档进行排序。
3. 使用排序后的列表和真实的关联性标签计算NDCG@k。

```python
# --- 进行预测 ---
# 预测结果是关联性分数，分数越高表示越相关
y_pred_scores = bst.predict(dtest)

# --- 评估性能（NDCG计算）---
# 我们需要一个函数来计算整个测试集的NDCG@k

def calculate_ndcg_at_k(y_true, y_pred_scores, groups, k):
    """计算所有查询组的平均NDCG@k。"""
    ndcg_scores = []
    start_idx = 0
    for group_size in groups:
        end_idx = start_idx + group_size
        # 获取当前组的真实关联性分数和预测分数
        group_y_true = y_true[start_idx:end_idx]
        group_y_pred = y_pred_scores[start_idx:end_idx]

        # 按预测分数降序排序文档
        sorted_indices = np.argsort(group_y_pred)[::-1]
        sorted_y_true = group_y_true[sorted_indices]

        # 计算DCG@k（折损累积增益）
        actual_k = min(k, group_size)
        dcg = np.sum((2**sorted_y_true[:actual_k] - 1) / np.log2(np.arange(2, actual_k + 2)))

        # 计算IDCG@k（理想DCG）
        ideal_sorted_y_true = np.sort(group_y_true)[::-1] # 将真实关联性分数降序排序
        idcg = np.sum((2**ideal_sorted_y_true[:actual_k] - 1) / np.log2(np.arange(2, actual_k + 2)))

        # 计算该组的NDCG@k
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcg_scores.append(ndcg)

        start_idx = end_idx

    return np.mean(ndcg_scores)

# 获取与测试集顺序对应的真实关联性标签
y_test_ordered = y_test.values # 确保它是一个NumPy数组

# 计算测试集上的NDCG@5和NDCG@10
ndcg_at_5 = calculate_ndcg_at_k(y_test_ordered, y_pred_scores, group_test, k=5)
ndcg_at_10 = calculate_ndcg_at_k(y_test_ordered, y_pred_scores, group_test, k=10)

print(f"\n测试集评估：")
print(f"计算出的平均NDCG@5: {ndcg_at_5:.4f}")
print(f"计算出的平均NDCG@10: {ndcg_at_10:.4f}")

# 与XGBoost在最后一轮训练中的内部评估结果进行比较
print("\n与训练最终轮次的指标比较：")
print(f"XGBoost报告的NDCG@5-test: {bst.eval(dtest).split()[1].split(':')[1]}")
print(f"XGBoost报告的NDCG@10-test: {bst.eval(dtest).split()[2].split(':')[1]}")
```

我们手动计算的结果应与XGBoost在训练期间报告的最终NDCG非常接近，这确认了我们对该指标的理解和实现。

### 特征重要性

我们仍然可以分析排序模型的特征重要性，以了解哪些特征对预测的关联性分数贡献最大。

```python
# --- 特征重要性 ---
importance = bst.get_score(importance_type='gain') # 'gain', 'weight', 'cover'
sorted_importance = sorted(importance.items(), key=lambda item: item[1], reverse=True)

print("\n特征重要性（增益）：")
for feature, score in sorted_importance:
    print(f"{feature}: {score:.4f}")

# 可选：可视化特征重要性
try:
    import matplotlib.pyplot as plt
    xgb.plot_importance(bst, importance_type='gain', max_num_features=10, height=0.8, title='特征重要性（增益）')
    plt.tight_layout()
    plt.show()
except ImportError:
    print("\n请安装matplotlib以可视化特征重要性：pip install matplotlib")
```

本次实践呈现了使用XGBoost执行排序学习任务的端到端过程。值得注意的方面包括所需的数据准备（查询分组）、`rank:pairwise` 等排序目标函数的使用，以及使用NDCG等指标进行评估。这种方法能让您发挥梯度提升的优势来优化项目的顺序，这是搜索引擎、推荐系统和问答中的常见需求。请记住，还有其他目标函数如 `rank:ndcg` 或 `rank:map` 可用，并且可能会根据具体数据集和评估标准带来更好的结果。

## 参考资料

- [Learning to Rank](https://xgboost.readthedocs.io/en/stable/tutorials/learning_to_rank.html) — XGBoost Contributors (2024)
  解释了如何使用XGBoost进行排序学习，包括`rank:pairwise`目标、用于查询组信息的`DMatrix.set_group`以及如何设置NDCG等评估指标。
- [Learning to Rank for Information Retrieval and Text Mining](https://link.springer.com/book/10.1007/978-3-642-00508-6) — Tie-Yan Liu (2009)
  Publisher: Springer; DOI: [10.1007/978-3-642-00508-6](https://doi.org/10.1007/978-3-642-00508-6)
  一本基础性著作，全面介绍了排序学习方法，包括逐点、逐对和逐列表方法，以及NDCG等常用评估指标。
- [From RankNet to LambdaMART: An Overview](https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdamart-an-overview/) — Christopher J. C. Burges (2010)
  Journal: Microsoft Research Technical Report MSR-TR-2010-82
  描述了逐对排序算法的演变，最终形成了LambdaMART，这是一种用于排序学习的梯度提升决策树方法。它为高效的排序梯度提升提供了理论基础。
- [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754) — Tianqi Chen, Carlos Guestrin (2016)
  Journal: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining; Publisher: ACM; DOI: [10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785)
  介绍了XGBoost算法的原始论文，详细阐述了其优化的分布式梯度提升框架。提供了用于排序任务的底层算法背景知识。

---

[上一节](07-%E5%A4%9A%E8%BE%93%E5%87%BA%E6%A2%AF%E5%BA%A6%E6%8F%90%E5%8D%87.md)
