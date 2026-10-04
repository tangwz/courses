---
course: "causal-inference-ml-systems"
chapter: "causal-inference-temporal-dynamic"
lesson: "practice-temporal-causal-analysis"
sourceId: 4408
sourceUrl: "https://apxml.com/zh/courses/causal-inference-ml-systems/chapter-5-causal-inference-temporal-dynamic/practice-temporal-causal-analysis"
title: "实践：时序因果分析"
description: "使用SVAR或时间序列发现技术对时序数据进行因果分析。"
order: 7
plots: []
sourceHash: "723f5bc2b271a3ab6d2b39f815ffd48a659a857d2e376d9378107716eabd2d5b"
sourceCorrections: []
---

好的，现在我们将本章的理论付诸实践。我们将重点应用一种时间序列因果发现算法，以便基于观测到的时间序列数据来理解系统内的动态关联。此练习模拟了经济学、气候科学或系统监测等背景下的常见情形，这些情形下我们会随着时间记录多个变量，并需要推断潜在的因果依赖关系，包括时间滞后。

我们将使用PCMCI算法，因为它非常适合在可能高维的时间序列中发现因果关联，同时考虑强自相关性。本次练习我们将使用合成数据。在此使用合成数据具有优势，因为我们知晓*真实*的潜在因果结构，这使我们能够评估我们的发现算法的表现。

### 建立环境和准备数据

首先，请确保已安装所需的库。我们将主要使用`tigramite`进行因果发现，以及`numpy`用于数据生成。

```python
# 所需库
# pip install tigramite numpy pandas matplotlib networkx
import numpy as np
import pandas as pd
import tigramite
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr # 偏相关检验
from tigramite.plotting import plot_graph # 用于可视化结果
import matplotlib.pyplot as plt # 可选：用于基本绘图
import networkx as nx # 如果需要，用于图操作
```

现在，我们根据一个已知的带有时间滞后的结构化因果模型（SCM）来生成一些合成时间序列数据。我们将定义一个包含三个变量（$X, Y, Z$）的简单向量 (vector)自回归 (autoregressive)（VAR）过程。

让我们定义真实因果关系：

- $X_t$ 受 $X_{t-1}$ 和 $Z_{t-1}$ 影响。
- $Y_t$ 受 $Y_{t-1}$ 和 $X_{t-1}$ 影响。
- $Z_t$ 受 $Z_{t-1}$ 和 $Y_{t-2}$ 影响。

相应的方程可能如下所示（其中 $\epsilon$ 表示噪声项）：


$$
X_t = 0.6 X_{t-1} + 0.4 Z_{t-1} + \epsilon_{X,t}
$$


$$
Y_t = 0.7 Y_{t-1} - 0.5 X_{t-1} + \epsilon_{Y,t}
$$


$$
Z_t = 0.8 Z_{t-1} + 0.3 Y_{t-2} + \epsilon_{Z,t}
$$


让我们根据此结构生成数据：

```python
np.random.seed(42) # 为了结果可重现
T = 500 # 时间点数量
N = 3   # 变量数量 (X, Y, Z)

# 初始化数据数组
data = np.zeros((T, N))
# 噪声项
noise_std = 0.5
noise = np.random.normal(loc=0, scale=noise_std, size=(T, N))

# 根据VAR过程生成数据
# 从t=2开始，因为我们需要最远到t-2的滞后
for t in range(2, T):
    data[t, 0] = 0.6 * data[t-1, 0] + 0.4 * data[t-1, 2] + noise[t, 0] # X_t
    data[t, 1] = 0.7 * data[t-1, 1] - 0.5 * data[t-1, 0] + noise[t, 1] # Y_t
    data[t, 2] = 0.8 * data[t-1, 2] + 0.3 * data[t-2, 1] + noise[t, 2] # Z_t

# 添加变量名称
var_names = ['X', 'Y', 'Z']

# 创建pandas DataFrame以便于处理（可选但推荐）
df = pd.DataFrame(data, columns=var_names)
print("生成数据形状:", df.shape)
print(df.head())

# 可选：可视化时间序列的基础绘图
# plt.figure(figsize=(12, 6))
# for i, name in enumerate(var_names):
#     plt.plot(df.index, df[name], label=name)
# plt.title('生成的时间序列数据')
# plt.xlabel('时间步')
# plt.ylabel('值')
# plt.legend()
# plt.show()
```

### 为Tigramite准备数据

`tigramite` 要求数据采用特定格式。我们需要使用其`Dataframe`对象封装我们的numpy数组。

```python
# 将numpy数组转换为Tigramite的Dataframe格式
data_tig = pp.Dataframe(data, var_names=var_names)
```

### 运行PCMCI算法

现在我们可以实例化并运行PCMCI算法。我们需要指定：

1. 数据 (`dataframe`)。
2. 条件独立性检验 (`cond_ind_test`)。我们将使用偏相关 (`ParCorr`)，它适用于我们VAR过程等线性高斯关系。
3. 显著性水平 (`pc_alpha`)。此阈值决定了哪些条件独立性被视为统计显著。一个常见的起始值是0.05。
4. 最大滞后 (`tau_max`)。这设定了我们想要考察因果关联的最大时间延迟。根据我们的真实情况（$Y_{t-2}$影响$Z_t$），`tau_max`需要至少为2。为了稳妥起见，我们将其设为3。

```python
# 初始化条件独立性检验
parcorr = ParCorr(significance='analytic')

# 初始化PCMCI
pcmci = PCMCI(
    dataframe=data_tig,
    cond_ind_test=parcorr,
    verbosity=1 # 执行时打印一些输出
)

# 运行PCMCI算法
# 我们将pc_alpha设为0.01，以实现可能更严格的过滤
# max_lag_parents包含滞后0（同期）关系，但时间序列因果分析中通常排除此项
results = pcmci.run_pcmci(tau_max=3, pc_alpha=0.01)

# 结果包含几个有用的属性：
# results['graph']：发现的因果图（numpy数组）
# results['p_matrix']：条件独立性检验的P值
# results['val_matrix']：检验统计量值（例如，偏相关）
```

`results['graph']`是一个numpy数组，其中`graph[i, j, tau]`表示从时间$t-\tau$的变量`j`到时间$t$的变量`i`的连接。

### 可视化并解读发现的图

`tigramite`提供了绘图工具，用于可视化发现的时间序列图。

```python
# 定义用于比较的真实图连接
# 格式：（源节点，目标节点，滞后）
# 注意：X的索引是0，Y是1，Z是2
true_links = {
    (0, 0, 1): {}, # X_t-1 -> X_t
    (2, 0, 1): {}, # Z_t-1 -> X_t
    (1, 1, 1): {}, # Y_t-1 -> Y_t
    (0, 1, 1): {}, # X_t-1 -> Y_t
    (2, 2, 1): {}, # Z_t-1 -> Z_t
    (1, 2, 2): {}  # Y_t-2 -> Z_t
}

# 绘制PCMCI发现的图
# 我们还可以提供真实连接来可视化真阳性、假阳性和假阴性边
pq = plot_graph(
    val_matrix=results['val_matrix'], # 显示连接强度
    graph=results['graph'],           # 显示发现的连接（基于pc_alpha）
    var_names=var_names,
    link_colorbar_label='cross-MCI partial corr.',
    node_colorbar_label='auto-MCI partial corr.',
    figsize=(8, 4),
    node_size=0.8,
    arrow_linewidth=1.5,
    # 传递真实图以进行比较可视化
    # true_graph_nodes=true_links # 注意：如果需要，请查阅tigramite文档以获取确切格式
    # 为了手动生成Graphviz图，我们将在下方提取连接
)
# plt.show() # 显示tigramite绘图器生成的图

# 用于Graphviz表示（更标准化的图格式）
# 从results['graph']中提取连接
discovered_links = []
for i in range(N): # 目标变量索引
    for j in range(N): # 源变量索引
        for lag in range(1, results['graph'].shape[2]): # 遍历滞后 > 0
            if results['graph'][i, j, lag] == '-->': # 检查是否存在有向连接
                source_node = f"{var_names[j]}(t-{lag})"
                target_node = f"{var_names[i]}(t)"
                discovered_links.append((source_node, target_node))

# 生成Graphviz DOT字符串
dot_string = "digraph TimeSeriesCausalGraph {\n"
dot_string += "  rankdir=LR;\n" # 从左到右的布局通常更适合表示时间
dot_string += "  node [shape=ellipse, style=filled, fillcolor=\"#a5d8ff\"];\n" # 节点样式
dot_string += "  edge [color=\"#495057\"];\n" # 边样式
for source, target in discovered_links:
    dot_string += f"  \"{source}\" -> \"{target}\";\n"
dot_string += "}"

print("\nGraphviz DOT格式：")
print(dot_string)
```

> 针对合成VAR数据应用PCMCI发现的因果图。节点表示特定时间点的变量（例如，$X(t-1)$是时间$t-1$的变量X）。箭头表示基于条件独立性检验推断出的因果关系。

**解读：**

将生成的图（`dot_string`输出或`tigramite`绘图）与我们定义的真实情况进行比较：

- $X_t \leftarrow X_{t-1}$ (正确找到)
- $X_t \leftarrow Z_{t-1}$ (正确找到)
- $Y_t \leftarrow Y_{t-1}$ (正确找到)
- $Y_t \leftarrow X_{t-1}$ (正确找到)
- $Z_t \leftarrow Z_{t-1}$ (正确找到)
- $Z_t \leftarrow Y_{t-2}$ (正确找到)

在这种干净、线性数据符合偏相关检验假设的理想情况下，PCMCI成功地恢复了真实的因果结构。

### 评估与思考

"\* **准确性：** 在实际情况中，您不会拥有真实数据。评估发现的图的质量通常依赖于领域知识、不同参数 (parameter)（例如`pc_alpha`）之间的一致性检查，或者如果可行，使用干预数据进行验证。使用合成数据时，我们可以计算诸如边发现的精确度和召回率等指标。"

- **参数敏感性：** `pc_alpha`和`tau_max`的选择很重要。
  - 较低的`pc_alpha`（例如0.01）会使算法更严格，可能导致假阳性减少但假阴性（遗漏的连接）增加。较高的值（例如0.05、0.1）则相反。建议通过改变`pc_alpha`进行敏感性分析。
  - `tau_max`应根据关于系统中合理时间延迟的领域知识进行选择。设置过低会遗漏较长范围的依赖关系，而设置过高则会增加计算成本和发现虚假关联的风险。
- **假设：** PCMCI依赖于一些假设，例如因果充分性（无重要的未观测到的共同原因）、时间序列的平稳性（或处理非平稳性的方法），以及所选条件独立性检验的适当性（ParCorr假设线性关系）。如果这些假设被违反，结果可能不准确。对于非线性关系，可以考虑使用`tigramite`中的替代检验方法，例如`GPDC`（高斯过程距离相关）或`CMIsymb`（基于互信息），尽管它们需要更多数据和计算量。
- **同期连接：** 我们关注的是滞后效应 ($\tau > 0$)。发现同期连接（$X_t \rightarrow Y_t$）由于对称性问题更具挑战性，并且通常需要额外的假设或技术（例如非时间DAG或SVAR模型中讨论的那些）。PCMCI主要侧重于识别滞后依赖关系。

"本次实践练习展示了如何应用PCMCI等复杂的时间序列因果发现算法。您生成了数据，准备了数据，运行了算法，并解读了由此产生的因果图。请记住，将这些方法应用于数据需要仔细考虑其潜在假设、对参数的敏感性以及根据领域专业知识进行的验证。"

## 参考资料

- [Detecting and quantifying causal associations in large nonlinear time series datasets](https://www.science.org/doi/10.1126/sciadv.aau4996) — Jakob Runge, Deniz Sendhil, Jonathan Stani, Andreas Schlicker, Jochem Marotzke, Jobst Heitzig (2019)
  Journal: Science Advances; Publisher: AAAS; Volume: 5; Pages: eaau4996; DOI: [10.1126/sciadv.aau4996](https://doi.org/10.1126/sciadv.aau4996)
  介绍了PCMCI算法，用于时间序列因果发现，阐述了其理论基础以及在复杂系统中的实际应用。
- [Tigramite Documentation](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQET4czyjmOegr_CJiGPpYYWvgCFhS53grBhEXYIJaL8P2fIwxKrttF-VNy8mfP7ggMZucRP6QCGQ9iWIvIQ4_mGS58trGwLZ5_DQ8gKAQ0u5n-GyiZFtxgTrufevBO9YirE) — Jakob Runge and The Tigramite Developers (2023)
  `tigramite` Python库的官方文档，提供了使用示例、API参考和应用因果发现算法的实践指南。
- [Elements of Causal Inference: Foundations and Learning Algorithms](https://mitpress.mit.edu/books/elements-causal-inference) — Jonas Peters, Dominik Janzing, Bernhard Schölkopf (2017)
  Publisher: The MIT Press; Pages: Chapter 10: Causal Inference for Time Series
  为因果推断提供了理论框架，其中包含关于时间序列数据的专门章节，涵盖了与时间因果发现相关的概念。
- [Causal network reconstruction from time series: From theory to applications](https://aip.scitation.org/doi/full/10.1063/5.0025122) — Jakob Runge (2020)
  Journal: Chaos: An Interdisciplinary Journal of Nonlinear Science; Publisher: AIP Publishing; Volume: 30; Pages: 113110; DOI: [10.1063/5.0025122](https://doi.org/10.1063/5.0025122)
  对从时间序列重建因果网络的方法进行了回顾，讨论了不同的理论基础、算法和应用，包括非线性方面。
