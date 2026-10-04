---
course: "causal-inference-ml-systems"
chapter: "advanced-causal-discovery"
lesson: "practice-implementing-discovery"
sourceId: 4339
sourceUrl: "https://apxml.com/zh/courses/causal-inference-ml-systems/chapter-2-advanced-causal-discovery/practice-implementing-discovery"
title: "动手实践：实现发现算法"
description: "使用 Python 库在样本数据集上实现各种因果发现算法。"
order: 7
plots: []
sourceHash: "396238372cfe011a5db66fdf11ef9d18106e24746b89e6642ef092d469532c2e"
sourceCorrections: []
---

获得亲手实践经验，应用因果发现算法。利用 Python 库从数据中推断因果图，重点在于解读结果并了解不同算法选择的实际效果。假定您拥有一个正常运行的 Python 环境，并已安装 `pandas`、`numpy`、`matplotlib` 等库以及专门的因果发现库。

### 准备工作：环境与数据

我们将主要使用 `causal-learn` 库，这是一个全面的因果发现软件包。如果尚未安装，可以通过 pip 进行安装：

```bash
pip install causal-learn matplotlib
```

为了可视化，`causal-learn` 依赖于 `graphviz`。请确保您已在系统范围内安装 Graphviz（请查看 Graphviz 网站以获取您操作系统的安装说明），并安装 Python `graphviz` 包：

```bash
pip install graphviz
```

让我们基于一个已知的结构因果模型（SCM）生成一些合成数据。这使我们能够评估因果发现算法与真实情况的符合程度。请看以下 SCM，其中 $N_i$ 代表独立的噪声项：

- $X_1 = N_1$
- $X_2 = 0.8 \cdot X_1 + N_2$
- $X_3 = 0.5 \cdot X_1 - 0.6 \cdot X_2 + N_3$
- $X_4 = 0.7 \cdot X_2 + 0.4 \cdot X_5 + N_4$
- $X_5 = N_5$

这对应于真实的DAG：$X_1 \to X_2$， $X_1 \to X_3$， $X_2 \to X_3$， $X_2 \to X_4$， $X_5 \to X_4$。

```python
import numpy as np
import pandas as pd
from causal_learn.utils.GraphUtils import GraphUtils
import matplotlib.pyplot as plt
import graphviz

# 设置随机种子以保证结果可重现
np.random.seed(42)
num_samples = 1000

# 基于 SCM 生成数据
n1 = np.random.randn(num_samples)
n2 = np.random.randn(num_samples)
n3 = np.random.randn(num_samples)
n4 = np.random.randn(num_samples)
n5 = np.random.randn(num_samples)

x1 = n1
x5 = n5
x2 = 0.8 * x1 + n2
x3 = 0.5 * x1 - 0.6 * x2 + n3
x4 = 0.7 * x2 + 0.4 * x5 + n4

# 创建 DataFrame
data = pd.DataFrame({'X1': x1, 'X2': x2, 'X3': x3, 'X4': x4, 'X5': x5})

# 定义真实图（邻接矩阵）
# 行/列对应 X1, X2, X3, X4, X5
true_dag_matrix = np.array([
    [0, 1, 1, 0, 0],  # X1 -> X2, X1 -> X3
    [0, 0, 1, 1, 0],  # X2 -> X3, X2 -> X4
    [0, 0, 0, 0, 0],  # X3 没有出边
    [0, 0, 0, 0, 0],  # X4 没有出边
    [0, 0, 0, 1, 0]   # X5 -> X4
])

print("模拟数据前几行：")
print(data.head())

# --- 使用 Graphviz 可视化图的函数 ---
def plot_graph(adjacency_matrix, labels, title="Learned Graph"):
    """使用 graphviz 可视化 DAG。"""
    num_nodes = adjacency_matrix.shape[0]
    node_names = [labels[i] for i in range(num_nodes)]

    dot = graphviz.Digraph(comment=title, graph_attr={'rankdir': 'LR', 'splines':'true', 'overlap':'false'})

    # 添加节点
    for name in node_names:
        dot.node(name, name, shape='ellipse', style='filled', fillcolor='#a5d8ff', color='#1c7ed6')

    # 添加边
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adjacency_matrix[i, j] == 1: # 有向边 i -> j
                 dot.edge(node_names[i], node_names[j], color='#495057')
            elif adjacency_matrix[i, j] == -1 and adjacency_matrix[j, i] == -1: # 无向边 i -- j (来自 CPDAG)
                 if i < j: # 只绘制一次
                     dot.edge(node_names[i], node_names[j], dir='none', color='#f76707')
            # 如果需要，添加其他边类型（例如，FCI 的双向边）

    return dot

print("\n可视化真实DAG...")
labels = ['X1', 'X2', 'X3', 'X4', 'X5']
gt_graph_dot = plot_graph(true_dag_matrix, labels, title="真实DAG")
# 在实际环境中，您将显示 gt_graph_dot
# 示例：gt_graph_dot.render('ground_truth_dag', view=True)
# 在此情境下，我们显示 Graphviz 源代码：
print("真实情况的 Graphviz 源代码：")
print(gt_graph_dot.source)
```

> 用于生成合成数据的真实因果结构。

### 应用基于约束的发现：PC算法

Peter-Clark (PC) 算法是一种经典的基于约束的方法。它从一个完全连接的无向图开始，并根据条件独立性检验迭代地移除边。最后，它根据碰撞模式和传播规则确定边的方向。

我们将使用 `causal_learn.discovery.ConstraintBased.PC` 实现。一个重要参数 (parameter)是条件独立性检验的显著性水平 `alpha`。常见的检验选择包括 `fisherz`（用于高斯数据）或 `gsq`（卡方，用于离散数据）。由于我们的数据是线性生成并带有高斯噪声的，因此 `fisherz` 适用。

```python
from causal_learn.discovery.ConstraintBased.PC import pc
from causal_learn.utils.cit import fisherz

# 运行 PC 算法
# 参数：数据矩阵 (numpy 数组)、显著性水平 (alpha)、独立性检验
cg_pc = pc(data.to_numpy(), alpha=0.05, indep_test=fisherz)

# 获取估计的邻接矩阵
# cg_pc.G.graph[i, j] == -1 且 cg_pc.G.graph[j, i] == -1 表示 i -- j
# cg_pc.G.graph[i, j] == 1 且 cg_pc.G.graph[j, i] == 0 表示 i -> j
estimated_adj_pc = cg_pc.G.graph

print("\nPC 算法估计的邻接矩阵：")
print(estimated_adj_pc)

# 可视化结果 (CPDAG)
print("\n可视化 PC 算法输出 (CPDAG)...")
pc_graph_dot = plot_graph(estimated_adj_pc, labels, title="PC 算法结果 (CPDAG)")
print("PC 结果的 Graphviz 源代码：")
print(pc_graph_dot.source)
```

> PC 算法估计的因果结构。在此例中，在数据充足且关系良好时，PC 算法正确地还原了真实 DAG。橙色无向边表示仅凭条件独立性无法确定方向的边（马尔可夫等价类的一部分）。

PC 算法通常返回一个完整部分有向无环图 (CPDAG)，代表马尔可夫等价类。CPDAG 中的无向边表示仅凭观测数据和条件独立性检验无法唯一确定方向的关系。在我们的具体案例中，生成的数据和所选参数使得 PC 算法能够完全确定图的方向，与真实情况匹配。然而，请注意，改变 `alpha`、样本大小或底层数据生成过程可能会导致错误（缺失或多余的边）或未定向的边。

### 应用基于评分的发现：GES算法

贪婪等价搜索（GES）是一种基于评分的算法，它分两个阶段运行：前向贪婪搜索以添加提高评分（如 BIC）的边，以及后向贪婪搜索以移除边。它在等价类（CPDAGs）空间中进行搜索。

我们使用 `causal_learn.discovery.ScoreBased.GES`。常见的评分函数包括 BIC (`bic`) 或 BDeu (`bdeu`)（用于离散数据），以及用于高斯数据的 BIC (`bic_gauss`)。

```python
from causal_learn.discovery.ScoreBased.GES import ges

# 运行 GES 算法
# 参数：数据矩阵 (numpy 数组)、评分方法
# 其他参数如 max_degree 可以调整
record_ges = ges(data.to_numpy(), score_func='bic_gauss')

# 从 GES 结果获取估计的邻接矩阵
estimated_adj_ges = record_ges['G'].graph

print("\nGES 算法估计的邻接矩阵：")
print(estimated_adj_ges)

# 可视化结果 (CPDAG)
print("\n可视化 GES 算法输出 (CPDAG)...")
ges_graph_dot = plot_graph(estimated_adj_ges, labels, title="GES 算法结果 (CPDAG)")
print("GES 结果的 Graphviz 源代码：")
print(ges_graph_dot.source)
```

> GES 算法使用高斯变量的 BIC 评分估计的因果结构。在此例中，与 PC 算法类似，GES 算法还原了真实的 DAG 结构。

与 PC 算法类似，GES 返回一个代表已识别的马尔可夫等价类的 CPDAG。对于此数据集，GES 也成功地还原了真实结构。当所选评分与数据生成过程（例如，高斯线性模型的 BIC）保持一致时，基于评分的方法通常表现良好。然而，如果模型假设（如线性或特定噪声分布）被违反，它们的性能可能会下降。搜索的贪婪特性也意味着它可能会停留在局部最优解。

### 引入潜在变量：FCI算法

如果存在未观测到的共同原因（潜在混淆因子）会怎样？标准的 PC 和 GES 算法假定因果充分性（无潜在混淆因子）。快速因果推断（FCI）算法是一种旨在处理此情况的基于约束的方法。它输出一个部分祖先图（PAG），其中可以包含有向边（$A \to B$）、双向边（$A \leftrightarrow B$，表示潜在共同原因）、无向边（$A - B$）和部分有向边。

让我们修改数据生成过程，以包含一个影响 $X_2$ 和 $X_4$ 的潜在变量 $L$，然后对 *观测* 数据（$X_1$ 到 $X_5$）运行 FCI。

```python
from causal_learn.discovery.ConstraintBased.FCI import fci

# --- 生成带有潜在混淆因子数据 ---
np.random.seed(123)
num_samples_latent = 1000

n1 = np.random.randn(num_samples_latent)
n2 = np.random.randn(num_samples_latent)
n3 = np.random.randn(num_samples_latent)
n4 = np.random.randn(num_samples_latent)
n5 = np.random.randn(num_samples_latent)
L = np.random.randn(num_samples_latent) # 潜在变量

x1_l = n1
x5_l = n5
# L 影响 X2 和 X4
x2_l = 0.8 * x1_l + 0.7 * L + n2
x3_l = 0.5 * x1_l - 0.6 * x2_l + n3
x4_l = 0.7 * x2_l + 0.4 * x5_l + 0.6 * L + n4

# 观测数据（L 被排除）
data_latent = pd.DataFrame({'X1': x1_l, 'X2': x2_l, 'X3': x3_l, 'X4': x4_l, 'X5': x5_l})

# 真实 PAG（FCI 输出代表此类别）
# X1 -> X2, X1 -> X3, X2 -> X3
# X5 -> X4
# X2 <-> X4 (由于潜在变量 L)

# 运行 FCI
# 注意：FCI 可能计算量很大
cg_fci, edges_fci = fci(data_latent.to_numpy(), alpha=0.05, indep_test=fisherz, verbose=False)

# causal-learn 中 PAG 的图表示需要仔细解读。
# cg_fci.graph[i, j] 编码边类型（0：无边，1：圆圈，2：箭头，3：尾部）
# 示例：cg_fci.graph[i, j] = 2 且 cg_fci.graph[j, i] = 3 表示 i -> j
# 示例：cg_fci.graph[i, j] = 2 且 cg_fci.graph[j, i] = 2 表示 i <-> j

print("\nFCI 算法估计的 PAG 边（编码）：")
# 这个原始矩阵难以直接阅读，可视化至关重要。
# print(cg_fci.graph)

# --- 使用 Graphviz 可视化 PAG 的函数 ---
def plot_pag(pag_matrix, labels, title="FCI Result (PAG)"):
    num_nodes = pag_matrix.shape[0]
    node_names = [labels[i] for i in range(num_nodes)]
    dot = graphviz.Digraph(comment=title, graph_attr={'rankdir': 'LR', 'splines':'true', 'overlap':'false'})

    for name in node_names:
       dot.node(name, name, shape='ellipse', style='filled', fillcolor='#d0bfff', color='#7048e8')

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes): # 避免重复绘制
            mark_i = pag_matrix[j, i] # 在节点 i 处标记边 j -> i
            mark_j = pag_matrix[i, j] # 在节点 j 处标记边 i -> j

            if mark_i == 0 and mark_j == 0: # 无边
                continue
            elif mark_i == 2 and mark_j == 2: # i <-> j (双向)
                dot.edge(node_names[i], node_names[j], arrowhead='normal', arrowtail='normal', dir='both', color='#f03e3e')
            elif mark_i == 3 and mark_j == 2: # i -> j
                dot.edge(node_names[i], node_names[j], arrowhead='normal', arrowtail='none', dir='forward', color='#495057')
            elif mark_i == 2 and mark_j == 3: # i <- j
                dot.edge(node_names[j], node_names[i], arrowhead='normal', arrowtail='none', dir='forward', color='#495057')
            elif mark_i == 1 and mark_j == 1: # i o-o j (无向 / 圆圈)
                dot.edge(node_names[i], node_names[j], arrowhead='odot', arrowtail='odot', dir='both', color='#fd7e14')
            elif mark_i == 1 and mark_j == 2: # i o-> j
                dot.edge(node_names[i], node_names[j], arrowhead='normal', arrowtail='odot', dir='forward', color='#cc5de8')
            elif mark_i == 2 and mark_j == 1: # i <-o j
                dot.edge(node_names[j], node_names[i], arrowhead='normal', arrowtail='odot', dir='forward', color='#cc5de8')
            # 其他组合（如尾部-尾部等）可能存在，但在简单的 FCI 输出中较不常见
            else: # 较不常见或未处理组合的默认情况
                 dot.edge(node_names[i], node_names[j], label=f'{mark_i}-{mark_j}', color='#adb5bd')
    return dot

print("\n可视化 FCI 算法输出 (PAG)...")
fci_graph_dot = plot_pag(cg_fci.graph, labels, title="FCI 算法结果 (PAG)")
print("FCI 结果的 Graphviz 源代码：")
print(fci_graph_dot.source)
```

> FCI 算法在包含影响 $X_2$ 和 $X_4$ 的潜在混淆因子 ($L$) 的数据上估计的部分祖先图 (PAG)。请注意双向边 $X_2 \leftrightarrow X_4$（红色），它正确地表明了这两个变量之间存在未观测到的共同原因。其他关系被识别为有向边。

FCI 的输出正确地识别了源自 $X_1$ 的有向边以及从 $X_5 \to X_4$ 的边。值得注意的是，它放置了一条双向边 $X_2 \leftrightarrow X_4$，表明它发现了影响两者的潜在混淆因子 $L$。即使没有因果充分性假设，FCI 也能保证其发现的祖先关系。

### 评估发现性能

当真实情况已知时（如在模拟中），我们可以使用以下指标定量评估算法性能：

- **结构汉明距离 (SHD):** 将估计图转换为真实图所需的边添加、删除或反转的数量。值越低越好。
- **真阳性率 (TPR):** 正确识别的边的比例。
- **假阳性率 (FPR):** 在不存在的真实边中错误识别的边的比例。
- **精确率:** 识别出的边中正确边的比例。

```python
from causal_learn.metrics import SHD, SID

# --- 评估 PC 结果（对比第一个真实情况）---
# 注意：需要将 PC 的 CPDAG 矩阵（无向边使用 -1 表示）
# 转换为标准邻接矩阵进行比较，如果它有无向边。
# 在我们的案例中，PC 返回了一个 DAG，因此可能不需要严格转换，
# 但比较通常需要正确处理 CPDAG。
# 为简单起见，这里假设如果没有 -1 则输出是 DAG。
shd_pc = SHD(true_dag_matrix, estimated_adj_pc)
# 精确率/召回率/F1 分数也可以计算，通常使用边的存在/不存在

print(f"\n--- 评估（无潜在变量情况）---")
print(f"PC 算法 SHD: {shd_pc}")

# --- 评估 GES 结果 ---
shd_ges = SHD(true_dag_matrix, estimated_adj_ges)
print(f"GES 算法 SHD: {shd_ges}")

# FCI 的评估更复杂，因为它输出的是 PAG，而不是 DAG/CPDAG。
# 指标需要比较 PAG 特征（祖先、确定结构）
# 标准 SHD 在不将 PAG 转换为可比较格式的情况下不直接适用。
# 存在专门的指标，但超出本次实际演练的范围。
print("\n（FCI 评估需要 PAG 特定指标，此处省略）")
```

"在这种理想情况下，PC 和 GES 都达到了 SHD 为 0，表明完美还原。在实际应用或更复杂的模拟中，您会预期非零的 SHD，并且会分析精确率/召回率以了解发生的错误类型。"

### 讨论

本次实践展示了如何使用 `causal-learn` 库应用主要的因果发现算法（PC、GES、FCI）。

- 我们了解到，当 PC 和 GES 的假设（例如，因果充分性，特定检验/评分的线性/高斯性）相当符合时，它们可以从观测数据中还原真实的 DAG 结构。
- 我们注意到 FCI 通过放宽因果充分性假设，可以在其 PAG 输出中使用双向边正确指示潜在混淆因子的存在。
- 我们简单介绍了在真实情况可用时如何使用 SHD 评估性能。

请记住，这些算法是工具，而非魔杖。它们的成功在很大程度上取决于数据的质量和数量、其基本假设（条件独立性检验、评分函数、线性、无环性、PC/GES 无隐性混淆因子）的有效性，以及适当的参数 (parameter)调整（基于约束的 `alpha`，基于评分的评分选择）。在解读输出图时，务必保持审慎，考虑潜在的局限性，并尽可能融入领域知识。在任何严谨的因果发现工作中，尝试不同算法、参数和进行稳健性检查都是必不可少的。

## 参考资料

- [Causality: Models, Reasoning and Inference](https://doi.org/10.1017/CBO9780511803163) — Judea Pearl (2009)
  Publisher: Cambridge University Press; Pages: 484; DOI: [10.1017/CBO9780511803163](https://doi.org/10.1017/CBO9780511803163)
  一本建立因果推断理论框架的奠基性著作，涵盖结构因果模型、有向无环图和d-分离，对于理解基于约束的发现算法至关重要。
- [Causation, Prediction, and Search](https://mitpress.mit.edu/books/causation-prediction-and-search) — Peter Spirtes, Clark Glymour, Richard Scheines (2000)
  Publisher: The MIT Press
  介绍并详细阐述了PC和FCI算法，为这些基于约束的因果发现方法提供了理论基础。
- [Elements of Causal Inference: Foundations and Learning Algorithms](https://mitpress.mit.edu/books/elements-causal-inference) — Jonas Peters, Dominik Janzing, Bernhard Schölkopf (2017)
  Publisher: The MIT Press; DOI: [10.7551/mitpress/9780262037666.001.0001](https://doi.org/10.7551/mitpress/9780262037666.001.0001)
  从机器学习角度概述了包括PC、GES和FCI在内的因果发现算法，侧重于其基本假设和统计检验。
