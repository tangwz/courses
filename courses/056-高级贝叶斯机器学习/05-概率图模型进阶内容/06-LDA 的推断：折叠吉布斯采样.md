# LDA 的推断：折叠吉布斯采样

来源：[原文](https://apxml.com/zh/courses/bayesian-machine-learning/chapter-5-advanced-probabilistic-graphical-models/lda-collapsed-gibbs-sampling)

[返回章节目录](README.md) · [返回课程目录](../README.md)

在前一节中，我们已经确立了潜在狄利克雷分配（LDA）的贝叶斯表述，现在我们来应对进行后验推断的挑战。我们的目标是计算给定观测词 $w$ 和超参数 (parameter) (hyperparameter) $\alpha$ 与 $\beta$ 的潜在变量（主要是主题分配 $z$）的后验分布。也就是说，我们想得到 $p(z, \theta, \phi | w, \alpha, \beta)$。

尽管可以通过从每个潜在变量（$z$, $\theta$, $\phi$）的条件分布中迭代采样来应用标准吉布斯采样，但连续参数 $\theta$（文档-主题分布）和 $\phi$（主题-词语分布）的高维度使其效率不高。此外，我们通常最关心的是主题分配 $z$ 本身，或者是 $\theta$ 和 $\phi$ 的期望值。

折叠吉布斯采样为LDA提供了一种优雅且通常有效的方法。其核心思想是利用LDA模型中狄利克雷先验与多项式似然之间的共轭性，分析性地积分掉，或“折叠”连续参数 $\theta$ 和 $\phi$。这使得我们得到一个吉布斯采样器，它只需要对每个文档 $d$ 中的每个词 $n$ 遍历离散的主题分配 $z_{d,n}$。

### 折叠吉布斯采样更新规则

LDA折叠吉布斯采样的核心是，在给定所有*其他*主题分配 $z_{\neg(d,n)}$、观测词 $w$ 以及超参数 (parameter) (hyperparameter) $\alpha, \beta$ 的情况下，将特定词元 (token) $(d,n)$（文档 $d$ 中的第 $n$ 个词）分配给特定主题 $k$ 的条件概率。我们将 $w_{d,n}$ 对应的词汇表 (vocabulary)词表示为 $v$。

利用贝叶斯定理并运用条件独立性和狄利克雷-多项式共轭性，我们可以推导这个条件概率。我们积分掉 $\theta_d$（文档 $d$ 的主题比例）和 $\phi_k$（主题 $k$ 的词语分布）：


$$
p(z_{d,n}=k | z_{\neg(d,n)}, w, \alpha, \beta) \propto p(w_{d,n}=v | z_{d,n}=k, z_{\neg(d,n)}, \beta) \times p(z_{d,n}=k | z_{d, \neg n}, \alpha)
$$


让我们分析右侧的这两个项：

1. **文档-主题项 $p(z_{d,n}=k | z_{d, \neg n}, \alpha)$：** 该项反映了主题 $k$ 在文档 $d$ 中出现的可能性，考虑了*同一文档*中其他词的分配 ($z_{d, \neg n}$) 和文档-主题先验 $\alpha$。积分掉 $\theta_d$ 会得到一个与文档 $d$ 中已分配给主题 $k$ 的词的数量（不包括当前词 $n$）加上先验参数 $\alpha_k$ 成比例的概率。设 $N_{d,k}^{\neg n}$ 是文档 $d$ 中（不包括词 $n$）分配给主题 $k$ 的词的数量。为简便起见，假设 $\alpha$ 是对称先验（即对所有 $k$ 都有 $\alpha_k = \alpha$）：

   
   $$
   p(z_{d,n}=k | z_{d, \neg n}, \alpha) \propto N_{d,k}^{\neg n} + \alpha
   $$
   

   该项倾向于将词分配给文档中已经常用的主题。
2. **主题-词语项 $p(w_{d,n}=v | z_{d,n}=k, z_{\neg(d,n)}, \beta)$：** 该项反映了特定词 $v$ 在主题 $k$ 下出现的可能性，考虑了语料库中所有其他词的分配 ($z_{\neg(d,n)}$) 和主题-词语先验 $\beta$。积分掉 $\phi$ 会得到一个与词 $v$ 在语料库中其他地方被分配给主题 $k$ 的次数加上先验参数 $\beta_v$ 成比例的概率。设 $N_{k,v}^{\neg (d,n)}$ 是词 $v$ 在所有文档中分配给主题 $k$ 的计数（不包括当前实例 $(d,n)$）。设 $N_k^{\neg (d,n)}$ 是分配给主题 $k$ 的词的总数（不包括当前实例）。假设 $\beta$ 是对称先验（即对所有 $v$ 都有 $\beta_v = \beta$），且 $V$ 是词汇表大小：

   
   $$
   p(w_{d,n}=v | z_{d,n}=k, z_{\neg(d,n)}, \beta) \propto \frac{N_{k,v}^{\neg (d,n)} + \beta}{N_k^{\neg (d,n)} + V\beta}
   $$
   

   该项倾向于将词分配给经常生成这种特定词类型 $v$ 的主题。

结合这些，采样主题分配 $z_{d,n}$ 的完整条件概率是：


$$
p(z_{d,n}=k | z_{\neg(d,n)}, w_{d,n}=v, \alpha, \beta) \propto (N_{d,k}^{\neg n} + \alpha) \times \frac{N_{k,v}^{\neg (d,n)} + \beta}{N_k^{\neg (d,n)} + V\beta}
$$


这个公式给出了将当前词元 $w_{d,n}$ 分配给每个主题 $k$ 的非归一化 (normalization)概率。我们计算所有 $K$ 个主题的这个值，然后对它们进行归一化，形成一个有效的概率分布，从中我们采样新的主题分配。

> 折叠吉布斯采样更新中，主题分配 $z_{d,n}=k$ 的依赖关系。该概率取决于与文档相关的计数 ($N_{d,k}$) 以及与主题和词类型相关的计数 ($N_{k,v}$, $N_k$)，并受先验 ($\alpha$, $\beta$) 调节。

### LDA 的折叠吉布斯采样算法

该算法步骤如下：

1. **初始化：**

   - 选择主题数量 $K$。
   - 设置超参数 (parameter) (hyperparameter) $\alpha$ 和 $\beta$（通常是对称的，例如 $\alpha = 50/K$, $\beta = 0.01$）。
   - 对于语料库中的每个词元 (token) $w_{d,n}$，随机分配一个主题 $z_{d,n} \in \{1, ..., K\}$。
   - 根据这些随机分配初始化计数矩阵：
     - $N_{d,k}$：文档 $d$ 中分配给主题 $k$ 的词的数量。
     - $N_{k,v}$：词 $v$ 在所有文档中分配给主题 $k$ 的次数。
     - $N_k$：所有文档中分配给主题 $k$ 的词的总数 ($N_k = \sum_v N_{k,v}$)。
2. **迭代（MCMC 采样）：**

   - 重复所需迭代次数（包括预热期和采样期）：
     - 对于每个文档 $d = 1 \dots M$：
       - 对于每个词位置 $n = 1 \dots N_d$（其中 $N_d$ 是文档 $d$ 的长度）：
         - 设 $k_{old} = z_{d,n}$ 是当前主题分配，$v = w_{d,n}$ 是词类型。
         - **减少计数：** 减少与*当前*分配 $(d, n, k_{old}, v)$ 相关的计数：
           $N_{d,k_{old}} \leftarrow N_{d,k_{old}} - 1$
           $N_{k_{old},v} \leftarrow N_{k_{old},v} - 1$
           $N_{k_{old}} \leftarrow N_{k_{old}} - 1$
           *（这些计数现在表示公式所需的 $N^{\neg(d,n)}$）。*
         - **计算采样概率：** 对于每个主题 $k' = 1 \dots K$，使用以下公式计算非归一化 (normalization)概率 $p'_{k'}$：
           
           $$
           p'_{k'} = (N_{d,k'}^{\neg n} + \alpha) \times \frac{N_{k',v}^{\neg (d,n)} + \beta}{N_{k'}^{\neg (d,n)} + V\beta}
           $$
           
           *（注意：此处使用的计数是减少后的计数）。*
         - **归一化：** 创建一个归一化概率分布 $P = [p_1, ..., p_K]$，其中 $p_{k'} = p'_{k'} / \sum_{j=1}^K p'_j$。
         - **采样新主题：** 从多项式分布 $P$ 中为词 $(d,n)$ 抽取一个新主题 $k_{new}$。
           $k_{new} \sim \text{Multinomial}(1, P)$
         - **更新分配：** 设置 $z_{d,n} = k_{new}$。
         - **增加计数：** 增加与*新*分配 $(d, n, k_{new}, v)$ 相关的计数：
           $N_{d,k_{new}} \leftarrow N_{d,k_{new}} + 1$
           $N_{k_{new},v} \leftarrow N_{k_{new},v} + 1$
           \$N\_{k\_{new}} \leftarrow N\_{k\_{new}} + 1
3. **输出：**

   - 丢弃初始“预热期”的样本。
   - 使用预热期后一个或多个样本的计数来估计模型参数。

### 估计模型参数 (parameter)（$\theta$ 和 $\phi$）

尽管 $\theta$ 和 $\phi$ 在采样过程中被积分掉，但我们可以使用采样器的最终计数（通常是充分预热后最后一次迭代的计数，或多个预热后样本的平均值）来估计它们的后验期望。

文档 $d$ 的预期文档-主题分布是：


$$
\hat{\theta}_{d,k} = \frac{N_{d,k} + \alpha_k}{\sum_{k'=1}^K (N_{d,k'} + \alpha_{k'})}
$$


主题 $k$ 的预期主题-词语分布是：


$$
\hat{\phi}_{k,v} = \frac{N_{k,v} + \beta_v}{\sum_{v'=1}^V (N_{k,v'} + \beta_{v'})}
$$


这些估计的分布 $\hat{\theta}$ 和 $\hat{\phi}$ 分别表示每个文档的学习主题混合，以及定义每个主题的词语概率。

### 讨论

折叠吉布斯采样是一种广泛使用的 LDA 推断方法，因为它相对简单，并且能有效地使用模型的共轭特性。通过避免直接采样连续参数 (parameter)，它有时可以比标准吉布斯采样器更有效地对主题分配 $z$ 的后验分布进行采样。

然而，它仍然是一种 MCMC 方法。需要评估收敛性（例如，通过监测数据的对数似然或主题一致性指标），并且需要一个合适的预热期。采样器按顺序处理词语，这可能导致混合缓慢，尤其是在大型数据集或主题数量较多的情况下。分配之间的关联可能意味着需要多次迭代才能从后验中获得独立的样本。

尽管存在这些潜在限制，折叠吉布斯采样为 LDA 的贝叶斯推断提供了一个可靠的基线和明确的方法。它与基于优化的方法（如变分贝叶斯，我们接下来会研究）形成对比，通过确定性近似权衡采样变异性以实现可能更快的收敛。

## 参考资料

- [Latent Dirichlet Allocation](http://www.jmlr.org/papers/v3/blei03a.html) — David M. Blei, Andrew Y. Ng, Michael I. Jordan (2003)
  Journal: Journal of Machine Learning Research; Volume: 3; Pages: 993-1022
  引入潜在狄利克雷分配模型的基础论文。
- [Finding Scientific Topics](https://doi.org/10.1073/pnas.0307752101) — Thomas L. Griffiths, Mark Steyvers (2004)
  Journal: Proceedings of the National Academy of Sciences; Publisher: National Academy of Sciences; Volume: 101; Pages: 5228-5235; DOI: [10.1073/pnas.0307752101](https://doi.org/10.1073/pnas.0307752101)
  介绍并详细阐述了用于潜在狄利克雷分配的坍缩吉布斯采样算法。
- [Bayesian Data Analysis](https://www.crcpress.com/Bayesian-Data-Analysis/Gelman-Carlin-Stern-Dunson-Vehtari-Rubin/p/book/9781439840955) — Andrew Gelman, John B. Carlin, Hal S. Stern, David B. Dunson, Aki Vehtari, Donald B. Rubin (2013)
  Publisher: Chapman and Hall/CRC
  关于贝叶斯统计建模和推断的综合教材，包括吉布斯采样等MCMC方法。
- [Probabilistic Graphical Models: Principles and Techniques](https://mitpress.mit.edu/books/probabilistic-graphical-models) — Daphne Koller, Nir Friedman (2009)
  Publisher: The MIT Press
  涵盖概率图模型（包括LDA和各种推断算法）的权威教材。

---

[上一节](05-%E9%9A%90%E7%8B%84%E5%88%A9%E5%85%8B%E9%9B%B7%E5%88%86%E9%85%8D%20%28LDA%29%EF%BC%9A%E8%B4%9D%E5%8F%B6%E6%96%AF%E8%A1%A8%E8%BF%B0.md) · [下一节](07-LDA%E7%9A%84%E6%8E%A8%E6%96%AD%EF%BC%9A%E5%8F%98%E5%88%86%E8%B4%9D%E5%8F%B6%E6%96%AF.md)
