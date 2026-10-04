---
course: "bayesian-machine-learning"
chapter: "advanced-probabilistic-graphical-models"
lesson: "lda-bayesian-formulation"
sourceId: 3618
sourceUrl: "https://apxml.com/zh/courses/bayesian-machine-learning/chapter-5-advanced-probabilistic-graphical-models/lda-bayesian-formulation"
title: "隐狄利克雷分配 (LDA)：贝叶斯表述"
description: "理解LDA的主题建模生成过程和贝叶斯表述。"
order: 5
plots: []
sourceHash: "57bdbe88fd4d3fa288e83a8637aef237a1bf6b1439534ba7119bc20da7675e7f"
sourceCorrections: []
---

隐狄利克雷分配 (LDA) 是一个突出的概率图模型例子，应用于文本分析，特别用于识别文档集合（语料库）中的主题结构，常称作“主题”。它基于生成过程的原则运作，即描述了文档可能如何生成的概率机制。理解这一生成过程，对把握LDA的贝叶斯表述有重要作用。

LDA 的主要思想有两个方面：

1. 每个文档被建模为固定数量主题的混合。例如，一篇关于技术的新闻文章可能由70%的“技术”、20%的“商业”和10%的“政治”主题组成。
2. 每个主题被建模为词汇表 (vocabulary)中词语的分布。“技术”主题可能对“软件”、“云计算”、“AI”和“网络”等词语赋予高概率，而“商业”主题则倾向于“股票”、“市场”、“利润”和“公司”等词语。

LDA 将每个文档的主题混合和每个主题的词语分布都视为隐性（未观测）随机变量。此外，它为每个文档中的*每个词语实例*分配一个特定主题。唯一观测到的数据是词语本身。推断的目标（我们将在后续章节中讨论）是在给定观测词语的情况下，推断这些隐性结构（主题混合、主题-词语分布和词语-主题分配）。

### 生成过程

让我们概述LDA假设的逐步生成过程，针对包含$M$个文档、预定义数量的$K$个主题以及包含$V$个独特词语的词汇表 (vocabulary)的语料库。

1. **定义先验：**

   - 为文档-主题分布的狄利克雷先验选择参数 (parameter)$\alpha$。通常，$\alpha$是一个对称的$K$维向量 (vector)，$\alpha = (\alpha_1, ..., \alpha_K)$，常设为$\alpha_i = \alpha_0 / K$，其中$\alpha_0$为某个标量。
   - 为主题-词语分布的狄利克雷先验选择参数$\beta$。通常，$\beta$是一个对称的$V$维向量，$\beta = (\beta_1, ..., \beta_V)$，常设为$\beta_j = \beta_0 / V$，其中$\beta_0$为某个标量。
2. **生成主题-词语分布：**

   - 对于每个主题 $k \in \{1, ..., K\}$：
     - 抽取一个词语分布 $\phi_k \sim Dir(\beta)$。$\phi_k$是一个$V$维向量，其中$\phi_{kv}$是词语$v$在主题$k$下出现的概率，且$\sum_{v=1}^V \phi_{kv} = 1$。
3. **生成文档特定变量：**

   - 对于每个文档 $d \in \{1, ..., M\}$：
     - 抽取一个主题混合 $\theta_d \sim Dir(\alpha)$。$\theta_d$是一个$K$维向量，其中$\theta_{dk}$是主题$k$在文档$d$中的比例，且$\sum_{k=1}^K \theta_{dk} = 1$。
     - 确定文档中的词语数量，$N_d$。
     - 对于每个词语位置 $n \in \{1, ..., N_d\}$：
       - 抽取一个主题分配 $z_{dn} \sim Cat(\theta_d)$。$z_{dn}$指示了文档$d$中第$n$个词语是由哪个主题生成的。
       - 抽取观测词语 $w_{dn} \sim Cat(\phi_{z_{dn}})$。$w_{dn}$是文档$d$中位置$n$处观测到的实际词语，它从对应于分配主题$z_{dn}$的词语分布中抽取。

超参数 (hyperparameter)$\alpha$和$\beta$分别控制主题混合和主题-词语分布的特征。较低的值通常导致更稀疏的分布（文档包含较少主题，主题集中于较少词语）。

### 图模型表示

此生成过程直接对应于一个贝叶斯网络结构。我们可以使用盘式记法（plate notation）来可视化这些依赖关系，其中盘（矩形）表示变量的重复。

> 隐狄利克雷分配模型的盘式记法。圆圈代表随机变量（阴影部分是隐性的，双圆圈是已观测的）。矩形（盘）表示重复。箭头表示条件依赖。从$\phi$到$w$的虚线表示所使用的特定$\phi_k$取决于$z$的值。

### 贝叶斯表述总结

在此PGM中，隐变量是文档-主题分布 $\boldsymbol{\theta} = \{\theta_d\}_{d=1}^M$、主题-词语分布 $\boldsymbol{\phi} = \{\phi_k\}_{k=1}^K$以及每个词语的主题分配 $\mathbf{Z} = \{z_{dn}\}_{d=1, n=1}^{M, N_d}$。观测变量是词语本身，$\mathbf{W} = \{w_{dn}\}_{d=1, n=1}^{M, N_d}$。参数 (parameter)$\alpha$和$\beta$通常被视为固定超参数 (hyperparameter)，尽管它们也可以被学习（例如，使用经验贝叶斯或设置超先验）。

在给定超参数的情况下，关于所有变量的完整联合概率分布根据图模型进行因子分解：


$$
p(\mathbf{W}, \mathbf{Z}, \boldsymbol{\theta}, \boldsymbol{\phi} | \alpha, \beta) = \left( \prod_{k=1}^K p(\phi_k | \beta) \right) \left( \prod_{d=1}^M p(\theta_d | \alpha) \left( \prod_{n=1}^{N_d} p(z_{dn} | \theta_d) p(w_{dn} | z_{dn}, \boldsymbol{\phi}) \right) \right)
$$


此处：

- $p(\phi_k | \beta)$ 是主题$k$词语分布的狄利克雷概率密度。
- $p(\theta_d | \alpha)$ 是文档$d$主题分布的狄利克雷概率密度。
- $p(z_{dn} | \theta_d)$ 是基于$\theta_d$分配主题$z_{dn}$的分类概率质量。
- $p(w_{dn} | z_{dn}, \boldsymbol{\phi})$ 是在给定其分配主题$z_{dn}$和相关主题-词语分布$\phi_{z_{dn}}$的情况下，观测词语$w_{dn}$的分类概率质量。

贝叶斯表述为推断奠定基础。我们的目标通常是计算在给定观测文档的情况下，隐变量的后验分布：$p(\mathbf{Z}, \boldsymbol{\theta}, \boldsymbol{\phi} | \mathbf{W}, \alpha, \beta)$。这种后验分布显现了隐藏的主题结构。然而，由于复杂的依赖关系和高维度，直接计算此后验是不可行的。这需要使用近似推断技术，例如折叠吉布斯采样或变分贝叶斯，这些是后续章节的主题。

## 参考资料

- [Latent Dirichlet Allocation](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf) — David M. Blei, Andrew Y. Ng, Michael I. Jordan (2003)
  Journal: Journal of Machine Learning Research; Publisher: MIT Press; Volume: 3; Pages: 993-1022; DOI: [10.1162/jmlr.2003.3.6.993](https://doi.org/10.1162/jmlr.2003.3.6.993)
  提出了Latent Dirichlet Allocation的开创性公式及其生成模型。
- [Machine Learning: A Probabilistic Perspective](https://www.cs.ubc.ca/~murphyk/MLbook/) — Kevin P. Murphy (2012)
  Publisher: MIT Press
  在概率机器学习背景下详细介绍了LDA，包含其生成过程和推断方法。
- [Finding statistical structure in natural language](https://www.pnas.org/doi/10.1073/pnas.0307797101) — Thomas L. Griffiths, Mark Steyvers (2004)
  Journal: Proceedings of the National Academy of Sciences; Publisher: National Academy of Sciences; Volume: 101; Pages: 5228-5235; DOI: [10.1073/pnas.0307797101](https://doi.org/10.1073/pnas.0307797101)
  介绍了用于LDA模型推断的广泛应用的Collapsed Gibbs抽样算法。
- [Probabilistic topic models](https://doi.org/10.1145/2133806.2133826) — David M. Blei (2012)
  Journal: Communications of the ACM; Publisher: ACM; Volume: 55; Pages: 77-84; DOI: [10.1145/2133806.2133826](https://doi.org/10.1145/2133806.2133826)
  从创作者视角对包含LDA在内的概率主题模型进行了易于理解的概述。
