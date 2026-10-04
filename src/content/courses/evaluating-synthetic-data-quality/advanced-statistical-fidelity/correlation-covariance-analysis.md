---
course: "evaluating-synthetic-data-quality"
chapter: "advanced-statistical-fidelity"
lesson: "correlation-covariance-analysis"
sourceId: 5649
sourceUrl: "https://apxml.com/zh/courses/evaluating-synthetic-data-quality/chapter-2-advanced-statistical-fidelity/correlation-covariance-analysis"
title: "相关性和协方差结构分析"
description: "比较相关矩阵和协方差结构的方法。"
order: 3
plots: ["plots/5649-0.json"]
sourceHash: "f7d6ca7067cc6bdeb59233941bca7fa6fd58103f9d7ada1a95cc2cd947b3946c"
sourceCorrections: []
---

匹配单个特征的分布是必要的检查，但这远不足以确立统计保真度。数据集很少由独立的变量组成。相反，特征之间通常存在复杂的相互依赖关系。一个合成数据集可能完美地复制了特征A和特征B各自的均值和标准差，但却完全未能捕获数据中两者之间观察到的强正向关系。分析相关性和协方差结构通过关注这些成对关系来解决这一不足。

### 比较相关矩阵

相关性衡量两个数值变量之间*线性*关系的强度和方向，其范围从-1（完全负相关）到+1（完全正相关），0则表示没有线性相关。相关矩阵提供了数据集中所有成对线性关系的全面视图。对于一个包含$n$个特征的数据集，它是一个$n \times n$的对称矩阵，其中元素$(i, j)$表示特征$i$和特征$j$之间的相关系数。对角线元素始终为1，表示特征与其自身的相关性。

为了评估保真度，我们计算真实数据集的相关矩阵，称之为$R_{corr}$，以及合成数据集的相关矩阵，$S_{corr}$。目标是比较$R_{corr}$和$S_{corr}$。

#### 使用热图进行视觉比较

一种直接且通常有帮助的方法是使用热图并排显示两个矩阵。模式、颜色或强度上的差异会立即显示捕获的线性结构中的不一致之处。



![相关矩阵比较](plots/5649-0.json)



> 真实数据和合成数据相关矩阵的并排热图。相似的颜色模式表明线性关系具有良好的保真度。

#### 定量比较指标

视觉检查有其用处但存在主观性。我们需要定量指标来总结$R_{corr}$和$S_{corr}$之间的差异。

1. **逐元素差异分析：** 计算差异矩阵 $D = R_{corr} - S_{corr}$。分析$D$中元素的分布。常见的统计量包括：

   - 平均绝对差 (MAD)：$D$的上（或下）三角中元素绝对值的平均值。值越小越好。
     $MAD = \frac{2}{n(n-1)} \sum_{i=1}^{n} \sum_{j=i+1}^{n} |r_{ij} - s_{ij}|$
   - 最大绝对差：对应相关系数之间的最大绝对差。有助于找出复制效果最差的关系。
   - 均方根误差 (RMSE)：与MAD相似，但对更大的误差给予更重的惩罚。
2. **矩阵距离指标：** 将相关矩阵视为高维空间 (high-dimensional space)中的点，并计算它们之间的距离。差值的Frobenius范数是常见的选择：
   $||R_{corr} - S_{corr}||_F = \sqrt{\sum_{i=1}^{n}\sum_{j=1}^{n} (r_{ij} - s_{ij})^2}$
   较小的Frobenius范数表示相关结构之间有更高的相似度。

### 比较协方差矩阵

协方差与相关性相似，但未经标准化。它衡量线性关系的方向，但其大小取决于各个特征的方差。协方差矩阵的元素$(i, j)$表示特征$i$和特征$j$之间的协方差。对角线元素表示每个特征的方差（$Var(X_i)$）。

$Cov(X, Y) = E[(X - E[X])(Y - E[Y])]$

您可以使用应用于相关矩阵的相同技术来比较真实数据集（$R_{cov}$）和合成数据集（$S_{cov}$）的协方差矩阵：视觉热图、逐元素差异分析和矩阵范数（如Frobenius范数）。

然而，如果真实数据和合成数据之间的边际方差差异很大，即使底层关系*结构*相似，直接比较协方差矩阵有时也可能产生误导。如果合成数据生成过程单独旨在匹配边际分布（包括方差），那么协方差比较是有用的。如果主要关注的是线性依赖关系的尺度不变结构，通常更推荐相关矩阵比较。

### 局限性与考量

- **线性：** 相关性和协方差都只捕获*线性*依赖关系。特征可能以非线性方式（例如，U形关系）强相关，但这会导致相关系数较低。这些方法无法捕获此类非线性结构。对于非线性依赖，需要其他技术，例如基于互信息的方法（将在“信息论度量”部分讨论）。
- **分类数据：** 标准的相关性/协方差是为数值特征定义的。对于包含分类特征的数据集，您可能需要使用不同的关联度量（如Cram\u00e9r's V）或特定的编码策略，尽管解释混合类型“相关”矩阵需要谨慎。
- **异常值：** 相关性和协方差计算可能对数据中的异常值敏感。可能需要进行预处理步骤或使用抗干扰方法。
- **解释：** 相关矩阵中的微小差异通常是好的，但可接受的阈值很大程度上取决于后续应用。相关系数0.05的差异对于某个任务可能微不足道，但对于严重依赖该特定特征相互影响的另一个任务则可能很重要。

通过比较相关性和协方差结构，您可以获得重要信息，以了解合成数据是否保留了原始数据中存在的成对线性关系，从而向更全面的统计保真度评估迈进。

## 参考资料

- [SDGym: A Benchmark for Evaluating Synthetic Data Generators](https://dl.acm.org/doi/10.1145/3447548.3467261) — João Miguel Santos, António G. C. F. F. N. De Azevedo, Sara H. Gutfreund, Afonso Veiga, Tomás Lopes (2021)
  Journal: Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining; Publisher: ACM; Pages: 3700-3708; DOI: [10.1145/3447548.3467261](https://doi.org/10.1145/3447548.3467261)
  提供了一个评估合成数据生成器的基准框架，直接展示了相关性比较等保真度指标的实际应用。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, Jerome Friedman (2009)
  Publisher: Springer
  统计学习的奠基性著作，为理解数据中的线性和非线性依赖关系及其对统计保真度的影响提供了更广阔的视角。
