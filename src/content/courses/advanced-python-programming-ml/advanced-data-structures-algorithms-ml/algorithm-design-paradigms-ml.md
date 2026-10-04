---
course: "advanced-python-programming-ml"
chapter: "advanced-data-structures-algorithms-ml"
lesson: "algorithm-design-paradigms-ml"
sourceId: 2337
sourceUrl: "https://apxml.com/zh/courses/advanced-python-programming-ml/chapter-4-advanced-data-structures-algorithms-ml/algorithm-design-paradigms-ml"
title: "机器学习中的算法设计方法（贪心算法、动态规划）"
description: "应用贪心算法和动态规划等算法设计技术来解决机器学习中的优化问题。"
order: 6
plots: []
sourceHash: "3685594eb27c74376d45064342b2dc86c3377f71368d67c5d94c1916bebf726d"
sourceCorrections: []
---

高级库提供了强大的抽象功能，但了解底层的算法机制变得重要。标准库函数可能并非总能为特定问题限制提供最佳方法，或者您可能需要完全实现自定义算法。理解贪心算法和动态规划等基本的算法设计方法变得非常有价值。这些技术提供了结构化的方式来处理机器学习 (machine learning)中常遇到的复杂优化问题，有可能显著降低计算复杂度，例如将指数级的 $O(2^n)$ 时间降至多项式时间。

## 机器学习 (machine learning)中的贪心算法

贪心算法一步步构建解决方案，总是选择当前步骤中最明显和直接有利的选项。它做出一个*局部最优*的选择，希望这一系列局部最优选择能带来一个*全局最优*的解决方案。这种方法简单，并且通常计算效率高。

为了使贪心策略保证全局最优，问题必须展现出两个特性：

1. **贪心选择性质：** 通过进行局部最优选择可以获得全局最优解决方案。选择*当前*最佳选项不会妨碍稍后达到整体最佳解决方案。
2. **最优子结构：** 问题的最优解决方案包含其子问题的最优解决方案。

**在机器学习中的应用：**

- **特征选择：** 顺序前向选择 (SFS) 或顺序后向选择 (SBS) 等算法属于贪心算法。SFS 从一个空的特征集开始，然后迭代地添加单个特征，该特征在当前步骤中能带来最佳的模型性能提升。SBS 从所有特征开始，然后迭代地移除对性能影响最小（或提升最大）的特征。这里的“贪心”部分是指在每个阶段选择添加或移除单个最佳特征，而无需提前考虑多个步骤。
- **决策树构建：** CART（分类与回归树）或 ID3 等算法采用贪心方法。在每个节点，它们为*该特定节点*选择能产生最佳划分的特征和划分点（例如，最大化信息增益或最小化基尼不纯度），而不考虑其对整个树结构后续部分的全局影响。
- **聚类（初始化）：** 一些聚类算法可能使用贪心方法进行初始质心选择，例如顺序地选择彼此相距较远的初始质心。

**示例：简化的顺序前向选择**

假设根据某种模型性能指标（例如准确率）从集合 $\{A, B, C, D\}$ 中选择最佳的两个特征。

1. **步骤 1：** 使用单个特征评估模型：$\{A\}$、$\{B\}$、$\{C\}$、$\{D\}$。假设 $\{B\}$ 产生最高准确率。选择 $B$。当前集合：$\{B\}$。
2. **步骤 2：** 使用当前集合加上一个额外特征来评估模型：$\{B, A\}$、$\{B, C\}$、$\{B, D\}$。假设 $\{B, D\}$ 产生最高准确率。选择 $D$。最终集合：$\{B, D\}$。

这是贪心的，因为在每一步中，我们都添加了单个最佳特征，而没有回溯或最初考虑像 $\{A, C\}$ 这样的特征对。

```python
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression # 示例分类器
from sklearn.model_selection import cross_val_score

# 假设 X (n_samples, n_features), y (n_samples,) 可用
# features = ['A', 'B', 'C', 'D'] # 对应 X 中列的名称

def evaluate_features(X_subset, y, model):
  """使用交叉验证评估特征子集。"""
  if X_subset.shape[1] == 0:
    return 0.0 # 没有特征，得分为零
  # 使用 cross_val_score 提高鲁棒性
  scores = cross_val_score(model, X_subset, y, cv=3, scoring='accuracy')
  return np.mean(scores)

def sequential_forward_selection(X, y, k, model, feature_names):
  """简单的 SFS 实现。"""
  num_features = X.shape[1]
  selected_indices = []
  remaining_indices = list(range(num_features))

  current_score = 0.0

  print(f"目标特征数量: {k}\n")

  for i in range(k):
    best_new_score = -1.0
    best_feature_index = -1

    print(f"--- 正在选择特征 {i+1} ---")

    for idx in remaining_indices:
      trial_indices = selected_indices + [idx]
      X_trial = X[:, trial_indices]
      score = evaluate_features(X_trial, y, model)

      print(f"  尝试特征 '{feature_names[idx]}' (索引: {trial_indices}): 分数 = {score:.4f}")

      if score > best_new_score:
        best_new_score = score
        best_feature_index = idx

    if best_feature_index != -1:
      print(f"\n选定的特征 '{feature_names[best_feature_index]}' 分数 {best_new_score:.4f}\n")
      selected_indices.append(best_feature_index)
      remaining_indices.remove(best_feature_index)
      current_score = best_new_score
    else:
      print("未找到改进，停止。")
      break # 无法改进

  selected_feature_names = [feature_names[i] for i in selected_indices]
  print(f"最终选定的特征 ({len(selected_indices)}): {selected_feature_names}")
  print(f"最终分数: {current_score:.4f}")
  return selected_indices, selected_feature_names

# --- 示例用法（需要实际数据 X, y） ---
# feature_names = ['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5']
# X = np.random.rand(100, 5) # 占位符数据
# y = (X[:, 0] + X[:, 1] * 0.5 + np.random.randn(100) * 0.1 > 0.6).astype(int) # 占位符标签
# model = LogisticRegression()
# k_features_to_select = 3

# selected_indices, selected_names = sequential_forward_selection(X, y, k_features_to_select, model, feature_names)
# print("\n选定的索引:", selected_indices)
```

**局限性：**

贪心算法的主要缺点是它们并非总能得到全局最优解决方案。局部最优的选择可能导致整体最佳解决方案无法实现。例如，在特征选择中，当前添加单个最佳特征可能会妨碍稍后选择*一对*单独表现平平但组合在一起效果出色的特征。

## 机器学习 (machine learning)中的动态规划

动态规划（DP）是一种强大的技术，通过将优化问题分解为更简单、重叠的子问题来解决。它每个子问题只解决一次并存储其解决方案，通常通过表格（制表法）或记忆化来完成。当再次遇到相同的子问题时，动态规划直接查找存储的结果，而不是重新计算。

当问题展现出以下特点时，动态规划适用：

1. **重叠子问题：** 问题可以分解为多次重复使用的子问题。
2. **最优子结构：** 问题的最优解决方案可以由其子问题的最优解决方案构成。

这与分治算法（如归并排序）不同，分治算法中的子问题通常是独立的，并且不显著重叠。

**在机器学习中的应用：**

- **序列比对（自然语言处理/生物信息学）：** 使用 Needleman-Wunsch（全局比对）或 Smith-Waterman（局部比对）等算法寻找两个序列（例如字符串、DNA序列）之间的相似性。核心思想是计算序列前缀的最佳比对分数。编辑距离（Levenshtein 距离）是一个常见的例子，它衡量将一个单词转换为另一个单词所需的最少单字符编辑（插入、删除、替换）次数。
- **隐马尔可夫模型 (HMMs)：** Viterbi 算法，用于找到导致一系列观测事件的最可能的隐藏状态序列，是一种经典的动态规划算法。它应用于自然语言处理中的词性标注或生物信息学中的基因识别等任务。
- **强化学习 (reinforcement learning)：** 动态规划构成了许多强化学习算法（如价值迭代和策略迭代）的理论基础，这些算法用于在已知动态（基于模型的强化学习）的环境中找到最优策略。
- **最优分箱/离散化：** 根据某种标准（例如，最小化信息损失）将连续特征离散化为固定数量的箱子的最佳方法有时可以转化为动态规划问题。

**示例：Levenshtein 距离（编辑距离）**

让我们来查找“PYTHON”和“PYTHONS”之间的编辑距离。我们可以将 $D(i, j)$ 定义为字符串 1 的前 $i$ 个字符与字符串 2 的前 $j$ 个字符之间的编辑距离。递推关系如下：


$$
D(i, j) = \min
\begin{cases}
D(i-1, j) + 1 & \text{(从 str1 删除)} \\
D(i, j-1) + 1 & \text{(向 str1 插入)} \\
D(i-1, j-1) + \text{cost}(str1[i], str2[j]) & \text{(匹配/替换)}
\end{cases}
$$


其中，如果 $c1 == c2$，则 $\text{cost}(c1, c2)$ 为 0，否则为 1。基本情况是 $D(i, 0) = i$ 和 $D(0, j) = j$。

我们可以使用制表法（一种自底向上构建表格的方法）来计算。

```python
import numpy as np

def levenshtein_distance(s1, s2):
  """使用动态规划（制表法）计算 Levenshtein 距离。"""
  m, n = len(s1), len(s2)

  # 创建一个动态规划表 (m+1 x n+1)
  dp_table = np.zeros((m + 1, n + 1), dtype=int)

  # 初始化基本情况
  for i in range(m + 1):
    dp_table[i, 0] = i # 从 s1 删除 i 个字符以得到空字符串的代价
  for j in range(n + 1):
    dp_table[0, j] = j # 向空字符串插入 j 个字符以得到 s2[:j] 的代价

  # 使用递推关系填充表格
  for i in range(1, m + 1):
    for j in range(1, n + 1):
      cost = 0 if s1[i - 1] == s2[j - 1] else 1 # 匹配代价为 0，替换代价为 1

      deletion_cost = dp_table[i - 1, j] + 1
      insertion_cost = dp_table[i, j - 1] + 1
      substitution_cost = dp_table[i - 1, j - 1] + cost

      dp_table[i, j] = min(deletion_cost, insertion_cost, substitution_cost)

  # 最终距离在右下角单元格中
  print("动态规划表：")
  print(dp_table)
  return dp_table[m, n]

# 示例
string1 = "PYTHON"
string2 = "PYTHONS"
distance = levenshtein_distance(string1, string2)
print(f"\n'{string1}' 和 '{string2}' 之间的 Levenshtein 距离: {distance}") 
# 预期输出：1（一次插入：'S'）

string3 = "KITTEN"
string4 = "SITTING"
distance2 = levenshtein_distance(string3, string4)
print(f"\n'{string3}' 和 '{string4}' 之间的 Levenshtein 距离: {distance2}")
# 预期输出：3（替换 K->S，替换 E->I，插入 G）
```

> 动态规划表显示了两个字符串前缀之间的最小编辑距离。`dp_table[i, j]` 保存 `s1[:i]` 和 `s2[:j]` 之间的距离。

或者，动态规划问题也可以使用*记忆化*（一种带有缓存的自上而下递归方法）来解决。一个字典或数组会存储子问题计算后的结果。如果函数调用遇到一个已经解决的子问题，它会返回缓存的结果。

```python
def levenshtein_memoization(s1, s2):
    """使用动态规划（记忆化）计算 Levenshtein 距离。"""
    memo = {}

    def compute_dist(i, j):
        if (i, j) in memo:
            return memo[(i, j)]

        # 基本情况
        if i == 0:
            return j # 插入 j 个字符的代价
        if j == 0:
            return i # 删除 i 个字符的代价

        # 递归步骤
        cost = 0 if s1[i - 1] == s2[j - 1] else 1

        deletion = compute_dist(i - 1, j) + 1
        insertion = compute_dist(i, j - 1) + 1
        substitution = compute_dist(i - 1, j - 1) + cost

        result = min(deletion, insertion, substitution)
        memo[(i, j)] = result
        return result

    return compute_dist(len(s1), len(s2))

# 使用记忆化的示例
distance_memo = levenshtein_memoization(string1, string2)
print(f"\n'{string1}' 和 '{string2}' 的记忆化结果: {distance_memo}") 
distance2_memo = levenshtein_memoization(string3, string4)
print(f"'{string3}' 和 '{string4}' 的记忆化结果: {distance2_memo}")
```

记忆化在递归思维下有时更直观实现，而制表法避免了递归深度限制，并且在某些 Python 实现中由于开销较低而可能效率稍高。

## 贪心算法与动态规划的选择

- **贪心算法：** 更简单、更快、内存占用较少。当您能证明（或有很强的直觉）贪心选择性质成立时，或当近似或足够好的解决方案可接受时使用。适用于局部改进自然地趋向全局最优的问题。
- **动态规划：** 设计和实现更复杂，可能更高的内存使用（用于动态规划表/记忆化缓存）。当问题具有重叠子问题和最优子结构，并且需要全局最优解决方案时使用。如果适用，保证最优性。

理解这些方法不仅仅是关于实现教科书上的算法。它更是关于识别您在机器学习 (machine learning)中遇到的优化问题的*结构*。问题是否可以分解？子问题是否重叠？局部选择能否带来全局解决方案？提出这些问题有助于您设计更高效的自定义组件，优化现有流程（如特征工程或超参数 (parameter) (hyperparameter)调整策略），并更好地理解机器学习库中不同算法方法所涉及的权衡。它们提供了一个强大的思维工具，用于应对计算挑战，而不是仅仅依赖现成的函数。

## 参考资料

- [Introduction to Algorithms](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition) — Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, and Clifford Stein (2022)
  Publisher: MIT Press
  一本基础性教材，严谨地涵盖了算法设计范式，包括贪心算法和动态规划，并提供全面的理论分析和实例。
- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: MIT Press
  强化学习领域的开创性著作，阐述了动态规划在价值迭代和策略迭代等基本算法中，为实现最优控制所发挥的作用。
- [Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  Publisher: Stanford University
  自然语言处理的综合性资源，详细介绍了动态规划在序列比对（如莱文斯坦距离）和隐马尔可夫模型（如维特比算法）中的应用。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://hastie.su.domains/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, Jerome Friedman (2009)
  Publisher: Springer; DOI: [10.1007/978-0-387-84858-7](https://doi.org/10.1007/978-0-387-84858-7)
  一本经典的机器学习教材，讨论了多种算法，包括决策树和特征选择方法，这些算法在构建过程中常采用贪心策略。
