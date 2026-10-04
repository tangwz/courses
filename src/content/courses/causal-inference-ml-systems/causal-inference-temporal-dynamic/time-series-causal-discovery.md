---
course: "causal-inference-ml-systems"
chapter: "causal-inference-temporal-dynamic"
lesson: "time-series-causal-discovery"
sourceId: 4405
sourceUrl: "https://apxml.com/zh/courses/causal-inference-ml-systems/chapter-5-causal-inference-temporal-dynamic/time-series-causal-discovery"
title: "时间序列因果发现方法"
description: "实现PCMCI等算法，用于发现时间序列数据集中的因果关系。"
order: 5
plots: []
sourceHash: "fbc579b6074613358ee10f656dba15a31e1bab9c2ae3512b06b58ebc3401714f"
sourceCorrections: []
---

格兰杰因果关系提供了基于预测能力的认识，而SVAR模型则在特定假设下帮助评估作用，但它们通常无法完全展现时间序列数据中底层的因果图结构。格兰杰因果关系本质上是非结构性的，SVAR则需要关于同期关系的强大先验知识，这可能并不总是可得的。为处理这一问题，专门的因果发现算法已被设计用于时间数据，旨在直接从观测数据中重构滞后和同期影响的网络。

这些方法通常使用*时间序列图*来表示因果关系。在此表示中，节点对应特定时间点（例如 $X_t$，$Y_{t-1}$）的变量，有向边表示因果影响。从 $X_{t-k}$ 到 $Y_t$ 的边表明变量 $X$ 在时间 $t-k$ 对变量 $Y$ 在时间 $t$ 施加直接因果作用，相对于系统中其他测量变量而言。常见的底层假设包括时间同质性（因果关系随时间保持不变）以及某种形式的平稳性，尽管放宽这些限制的方法是当前研究的热点。

### 基于约束的时间序列因果发现：PCMCI

PCMCI（全称“多时间序列条件独立性的PC算法”）是一种广泛使用且适用于时间序列的基于约束的算法。它基于Peter-Clark（PC）算法的逻辑，但对其进行了修改，以有效处理时间序列固有的自依赖性以及因考虑多个时间滞后而产生的高维性。

PCMCI 分两个主要阶段进行：

1. **父候选变量选择（PC-阶段）：** 对于时间序列中的每个变量 $X^i_t$ 以及每个潜在滞后 $\tau > 0$ 直到最大滞后 $\tau_{max}$，此阶段识别一组*候选父变量* $\mathcal{P}(X^i_t)$。这些候选变量从所有其他变量 $X^j_{t-\tau'}$（包括变量自身的过去值，即 $i=j$）在相关的过去滞后 $\tau'$ 中选择。与标准PC算法类似，它采用一系列条件独立性（CI）测试，逐步以更大的潜在分离变量集为条件进行测试。此剪枝步骤通过消除在给定其他变量条件下与目标变量条件独立的变量，显著减少了搜索空间。对时间序列的适应之处在于系统地测试*滞后*依赖关系。
2. **瞬时条件独立性（MCI）测试：** 一旦为每个目标变量 $X^i_t$ 识别了候选父变量 $\mathcal{P}(X^i_t)$，MCI测试就确定每个候选 $X^j_{t-\tau}$ 是否存在直接因果联系 $X^j_{t-\tau} \to X^i_t$。它执行一个特定的条件独立性测试：

   
   $$
   X^j_{t-\tau} \perp X^i_t \mid \mathcal{P}(X^i_t) \setminus \{X^j_{t-\tau}\}, \mathcal{P}(X^j_{t-\tau})
   $$
   

   条件集经过精心构建。它包括目标变量 $X^i_t$ 已识别的候选父变量（但排除了正在测试的特定潜在父变量 $X^j_{t-\tau}$），并且通常也包括*源*变量 $X^j_{t-\tau}$ 的父变量。此条件设置有助于解释自相关以及通过其他变量中介的潜在间接因果路径。如果条件独立性假设被拒绝（表示存在依赖性），则表示因果联系 $X^j_{t-\tau} \to X^i_t$ 的有向边被添加到估计的时间序列图中。

PCMCI 提供了一种处理多元时间序列中常见的高维性和自相关挑战的方法。通过首先缩小潜在因果影响范围（PC-阶段），然后应用有针对性的MCI测试，它管理了高维时间设置中条件独立性测试的复杂性。

应用 PCMCI 时，需要仔细考量多个实际方面：

- **最大时间滞后（$\tau_{max}$）：** 选择最大滞后 $\tau_{max}$ 是一个重要决定。设置过低可能导致遗漏在更长时间尺度上运作的真实因果联系。相反，设置过高会增加计算需求并带来发现虚假关联的风险。相关专业知识或初步时间序列分析（例如检查偏自相关函数）可以提供有价值的指导。
- **条件独立性测试：** CI 测试的选择非常重要，取决于数据和关系的性质。对于线性高斯系统，偏相关（`ParCorr`）通常适用。对于非线性关系或非高斯噪声，基于核的方法（如`GPDC`）或基于互信息的方法（如`CMIknn`）可能更合适。使用不当的测试可能导致发现的图结构出现显著错误。
- **显著性水平（$\alpha$）：** CI 测试中使用的显著性水平 $\alpha$ 决定了拒绝独立性零假设的阈值。它控制着发现真实但可能微弱联系（更高敏感度，更低的 $\alpha$）与避免虚假联系（更高特异性，更高的 $\alpha$）之间的权衡。

### 时间序列因果图示例

以下图表展示了三个时间序列变量（X、Y、Z）之间可能存在的因果结构，这些结构可以由像PCMCI这样的算法发现。

> 一个时间序列图，显示了每个变量内部的自依赖关系（例如，$X_{t-1} \to X_t$）以及跨时间步不同变量之间的滞后因果影响（例如，$X_{t-1} \to Y_t$，$Z_{t-2} \to Y_t$，$Y_{t-1} \to Z_t$）。

### 基于分数的及其他时间因果发现方法

基于约束的方法和基于分数的算法也可以适用于时间序列因果发现。例如，存在针对时间数据定制的贪婪等价搜索（GES）的变体。此外，使用函数因果模型的方法提供了其他视角。示例包括为向量 (vector)自回归 (autoregressive)过程设计的线性非高斯无环模型（LiNGAM）的扩展（例如 VAR-LiNGAM, TiMINo），它们可以在非高斯噪声分布或特定函数形式的假设下识别因果结构。最近的进展包括使用神经网络 (neural network)为潜在非线性动力学设计的方法，例如 DYNOTEARS，它在一个适用于线性动力学的连续优化框架内，优化了一个结合格兰杰因果思想的评分函数。

### 处理潜在变量和评估发现结果

未观测到的混杂因素仍然是时间序列因果发现中的一个主要障碍，正如在静态设置中一样。像快速因果推断（FCI）算法（例如 tsFCI）的时间序列适应版本已被设计。这些方法旨在识别即使存在潜在变量也成立的因果关系。输出图通常包含额外的边类型（例如双向边或部分有向边），以明确表示因果方向的不确定性或未观测到的共同原因的潜在存在。

时间序列因果发现算法的性能评估通常使用从已知因果结构生成的人工数据完成。标准指标包括结构汉明距离（SHD），用于测量估计图和真实图之间的差异，以及精确度（发现的真实边的比例）和召回率（发现的真实边的比例）。在实际应用中，发现图的实用性可以通过其改善下游任务的能力来评估，例如预测准确性或基于推断结构设计的干预措施的有效性。

### 实际实现说明

专门的软件包使得这些高级方法的应用更加方便。Python 中的 `tigramite` 库是一个全面的框架，实现了 PCMCI、各种适用于时间序列的条件独立性测试以及相关的分析工具。

```python
# 使用 tigramite 的 Python 工作流程（示例性）
import numpy as np
import tigramite
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr, GPDC # 示例 CI 测试

# 假设 'data' 是一个形状为 (时间步数, 变量数量) 的 NumPy 数组
# 假设 'var_names' 是一个列表，例如 ['X', 'Y', 'Z']

# 示例数据（请替换为实际时间序列数据）
# data = np.random.randn(100, 3)

# 1. 使用 tigramite 的 DataFrame 准备数据
#    处理缺失值遮蔽等
dataframe = pp.DataFrame(data, var_names=var_names)

# 2. 选择条件独立性测试
#    对于线性高斯假设，使用 ParCorr
#    对于非线性情况，使用 GPDC（高斯过程排放系数）
# cond_ind_test = ParCorr()
cond_ind_test = GPDC(significance='analytic') # 非线性示例

# 3. 初始化 PCMCI 算法实例
pcmci = PCMCI(
    dataframe=dataframe,
    cond_ind_test=cond_ind_test,
    verbosity=1) # 设置输出的详细程度

# 4. 执行 PCMCI 算法
#    指定最大滞后（tau_max）和 PC 阶段的显著性水平（pc_alpha）
#    MCI 测试的 alpha_level 通常在需要时单独设置
results = pcmci.run_pcmci(tau_max=5, pc_alpha=0.05, alpha_level=0.01)

# 5. 处理和解读结果
#    结果字典包含 p 值、连接强度和图数组
print("显著连接 (父 --> 子):")
pcmci.print_significant_links(
    p_matrix=results['p_matrix'],
    val_matrix=results['val_matrix'],
    alpha_level=0.01) # 打印在 MCI alpha 水平下显著的连接

#    图结构可以被可视化
# from tigramite import plotting
# tp = plotting.TipPlot(results)
# tp.plot_graph(
#     show_colorbar=False,
#     var_names=var_names,
#     link_colorbar_label='cross-MCI'
# )
# tp.show() # 显示图表
```

此代码展示了标准的工作流程：在 `tigramite` 框架内进行数据准备，根据数据特性选择合适的 CI 测试，配置和执行 PCMCI 算法，并指定 `tau_max` 和显著性水平等参数 (parameter)，以及对发现的因果图进行分析和可视化。仔细调整算法参数并验证底层假设是获取时间数据可靠因果认识的必要步骤。

这些因果发现方法提供了摆脱时间序列中相关性分析的重要工具。通过旨在理解底层数据生成过程，它们支持更可靠的预测、干预模拟以及对动态系统形成更透彻的认识。然而，与所有因果推断技术一样，它们的有效性建立在必须在特定应用背景下仔细评估的假设之上。

## 参考资料

- [Causal network reconstruction from time series: From statistics to applications](https://doi.org/10.1073/pnas.1714782115) — Jakob Runge, Lydia Nowack, Marlene Kretschmer, Veronika Eyring, Manfred Milinski (2018)
  Journal: Proceedings of the National Academy of Sciences; Publisher: National Academy of Sciences; Volume: 115; Pages: E6070-E6079; DOI: [10.1073/pnas.1714782115](https://doi.org/10.1073/pnas.1714782115)
  介绍了用于时间序列因果发现的PCMCI基础算法，详细阐述了其处理自相关性和高维性的两阶段方法。
- [Discovering causality in complex systems with tigramite](https://doi.org/10.21105/joss.02470) — Jakob Runge (2020)
  Journal: Journal of Open Source Software; Publisher: Open Journals; Volume: 5; Pages: 2470; DOI: [10.21105/joss.02470](https://doi.org/10.21105/joss.02470)
  介绍了`tigramite` Python库，一个用于应用PCMCI及其他时间序列因果发现方法的综合框架，提供了多种条件独立性测试。
- [A review of causal discovery algorithms for time series data](https://doi.org/10.1007/s10462-021-10041-3) — Muhammad Malik, Shahbaz Saquib, Imtiaz Shafi (2022)
  Journal: Artificial Intelligence Review; Publisher: Springer Netherlands; Volume: 55; Pages: 2821-2856; DOI: [10.1007/s10462-021-10041-3](https://doi.org/10.1007/s10462-021-10041-3)
  一篇近期综述文章，概述了适用于时间序列数据的各种因果发现算法，包括本节中提到的基于约束和基于评分的方法。
- [Elements of Causal Inference: Foundations and Learning Algorithms](https://mitpress.mit.edu/books/elements-causal-inference) — Jonas Peters, Dominik Janzing, Bernhard Schölkopf (2017)
  Publisher: MIT Press
  一本关于因果推断的基础教科书，涵盖了结构因果模型、条件独立性测试和PC算法，这些是理解PCMCI等基于约束方法的基础。
