---
course: "bayesian-machine-learning"
chapter: "variational-inference-techniques"
lesson: "deriving-elbo"
sourceId: 3557
sourceUrl: "https://apxml.com/zh/courses/bayesian-machine-learning/chapter-3-variational-inference-techniques/deriving-elbo"
title: "推导证据下界 (ELBO)"
description: "理解变分推断中 ELBO 的数学推导与解释。"
order: 2
plots: []
sourceHash: "e83d35f3d96cb87184a11941556d6aa1619b0fbd77471e6920b9985b087b788a"
sourceCorrections: []
---

变分推断（VI）通过将推断问题转化为优化问题，寻求对真实后验 $p(\mathbf{z}|\mathbf{x})$ 的近似 $q(\mathbf{z})$。我们优化的核心量是证据下界（ELBO）。下面我们来推导这个主要的目标函数。

我们的目标是使 $q(\mathbf{z})$ 尽可能接近真实后验 $p(\mathbf{z}|\mathbf{x})$。衡量两种概率分布之间差异的标准方式是 Kullback-Leibler (KL) 散度。我们希望使近似 $q(\mathbf{z})$ 到真实后验 $p(\mathbf{z}|\mathbf{x})$ 的 KL 散度最小化：


$$
KL(q(\mathbf{z}) || p(\mathbf{z}|\mathbf{x})) = \int q(\mathbf{z}) \log \frac{q(\mathbf{z})}{p(\mathbf{z}|\mathbf{x})} d\mathbf{z}
$$


直接最小化这个 KL 散度很困难，因为它涉及真实后验 $p(\mathbf{z}|\mathbf{x})$，而真实后验本身包含难以处理的证据项 $p(\mathbf{x})$，因为 $p(\mathbf{z}|\mathbf{x}) = p(\mathbf{x}, \mathbf{z}) / p(\mathbf{x})$。

让我们展开 KL 散度公式中后验的定义：


$$
KL(q(\mathbf{z}) || p(\mathbf{z}|\mathbf{x})) = \int q(\mathbf{z}) \log \frac{q(\mathbf{z}) p(\mathbf{x})}{p(\mathbf{x}, \mathbf{z})} d\mathbf{z}
$$


我们可以拆分对数项：


$$
KL(q(\mathbf{z}) || p(\mathbf{z}|\mathbf{x})) = \int q(\mathbf{z}) \log q(\mathbf{z}) d\mathbf{z} - \int q(\mathbf{z}) \log p(\mathbf{x}, \mathbf{z}) d\mathbf{z} + \int q(\mathbf{z}) \log p(\mathbf{x}) d\mathbf{z}
$$


现在，让我们分析每个项：

1. $\int q(\mathbf{z}) \log q(\mathbf{z}) d\mathbf{z}$: 这是分布 $q(\mathbf{z})$ 的负熵的定义，通常写作 $-\mathbb{H}(q)$。用期望符号表示，它是 $\mathbb{E}_{q(\mathbf{z})} [\log q(\mathbf{z})]$。
2. $\int q(\mathbf{z}) \log p(\mathbf{x}, \mathbf{z}) d\mathbf{z}$: 这是在分布 $q(\mathbf{z})$ 下对数联合概率 $p(\mathbf{x}, \mathbf{z})$ 的期望，写作 $\mathbb{E}_{q(\mathbf{z})} [\log p(\mathbf{x}, \mathbf{z})]$。
3. $\int q(\mathbf{z}) \log p(\mathbf{x}) d\mathbf{z}$: 由于 $\log p(\mathbf{x})$ 对于积分变量 $\mathbf{z}$ 是常数，我们可以将其提出积分号外。剩余的积分 $\int q(\mathbf{z}) d\mathbf{z}$ 简单地等于 1，因为 $q(\mathbf{z})$ 是一个概率分布。因此，这一项简化为 $\log p(\mathbf{x})$。

将这些代回 KL 散度公式：


$$
KL(q(\mathbf{z}) || p(\mathbf{z}|\mathbf{x})) = \mathbb{E}_{q(\mathbf{z})} [\log q(\mathbf{z})] - \mathbb{E}_{q(\mathbf{z})} [\log p(\mathbf{x}, \mathbf{z})] + \log p(\mathbf{x})
$$


将此方程重新排列以分离对数证据 $\log p(\mathbf{x})$，得到变分推断中的一个基本恒等式：


$$
\log p(\mathbf{x}) = \underbrace{\mathbb{E}_{q(\mathbf{z})} [\log p(\mathbf{x}, \mathbf{z})] - \mathbb{E}_{q(\mathbf{z})} [\log q(\mathbf{z})]}_{\mathcal{L}(q)} + KL(q(\mathbf{z}) || p(\mathbf{z}|\mathbf{x}))
$$


标记 (token)为 $\mathcal{L}(q)$ 的项是证据下界（ELBO）：


$$
\mathcal{L}(q) = \mathbb{E}_{q(\mathbf{z})} [\log p(\mathbf{x}, \mathbf{z})] - \mathbb{E}_{q(\mathbf{z})} [\log q(\mathbf{z})]
$$


这也可以使用熵 $\mathbb{H}(q) = -\mathbb{E}_{q(\mathbf{z})} [\log q(\mathbf{z})]$ 表示为：


$$
\mathcal{L}(q) = \mathbb{E}_{q(\mathbf{z})} [\log p(\mathbf{x}, \mathbf{z})] + \mathbb{H}(q)
$$


### 为什么它是下界？

KL 散度具有以下性质：对于任何分布 A 和 B，$KL(A || B) \ge 0$，当且仅当 A=B 时等号成立。因此，$KL(q(\mathbf{z}) || p(\mathbf{z}|\mathbf{x})) \ge 0$。

回顾重新排列后的方程：


$$
\log p(\mathbf{x}) = \mathcal{L}(q) + KL(q(\mathbf{z}) || p(\mathbf{z}|\mathbf{x}))
$$


由于 KL 项是非负的，立刻得到：


$$
\log p(\mathbf{x}) \ge \mathcal{L}(q)
$$


这证实了 $\mathcal{L}(q)$ 确实是模型证据对数的下界。

### 优化目标

恒等式 $\log p(\mathbf{x}) = \mathcal{L}(q) + KL(q(\mathbf{z}) || p(\mathbf{z}|\mathbf{x}))$ 为变分推断提供了主要思路。我们希望在我们选择的分布族中找到使 KL 散度 $KL(q(\mathbf{z}) || p(\mathbf{z}|\mathbf{x}))$ 最小化的 $q(\mathbf{z})$。

请注意，对数证据 $\log p(\mathbf{x})$ 对于定义 $q(\mathbf{z})$ 的参数 (parameter)而言是常数。因此，最大化 ELBO $\mathcal{L}(q)$ *等价于* 最小化 KL 散度 $KL(q(\mathbf{z}) || p(\mathbf{z}|\mathbf{x}))$。

可以这样理解：总对数证据是固定的。它由 ELBO 和 KL 散度组成。如果我们增加 ELBO，KL 散度必须减少相同的量以保持总和不变。

这非常有用，因为 ELBO，即 $\mathcal{L}(q) = \mathbb{E}_{q(\mathbf{z})} [\log p(\mathbf{x}, \mathbf{z})] - \mathbb{E}_{q(\mathbf{z})} [\log q(\mathbf{z})]$，仅取决于联合分布 $p(\mathbf{x}, \mathbf{z})$（这通常是可计算的）和我们选择的变分分布 $q(\mathbf{z})$。它巧妙地避免了难以处理的证据 $p(\mathbf{x})$。

因此，变分推断按以下步骤进行：

1. 为 $q(\mathbf{z})$ 选择一个分布族，通常由一些变分参数 $\phi$ 参数化。我们将其表示为 $q_{\phi}(\mathbf{z})$。
2. 寻找使 ELBO 最大的参数 $\phi$：
   
   $$
   \phi^* = \arg \max_{\phi} \mathcal{L}(q_{\phi})
   $$
   

所得分布 $q_{\phi^*}(\mathbf{z})$ 作为对真实后验 $p(\mathbf{z}|\mathbf{x})$ 的近似。ELBO 在最优值处的取值，即 $\mathcal{L}(q_{\phi^*})$，也提供了对数边际似然的下界，这对于模型比较很有用。

理解 ELBO 及其与 KL 散度和模型证据的关系，是把握变分推断方法运作方式的基本。后续章节将讨论 $q(\mathbf{z})$ 的具体分布族以及优化 ELBO 的算法。

## 参考资料

- [Pattern Recognition and Machine Learning](https://link.springer.com/book/9780387310732) — Christopher M. Bishop (2006)
  Publisher: Springer; Pages: 461-468 (Chapter 10, Section 10.1)
  一本经典教科书，在变分推断背景下，提供了证据下界的基石性全面推导。
- [An Introduction to Variational Methods for Graphical Models](https://link.springer.com/article/10.1023/A:1007665907178) — Michael I. Jordan, Zoubin Ghahramani, Tommi Jaakkola, and Lawrence K. Saul (1999)
  Journal: Machine Learning; Volume: 37; Pages: 183-233; DOI: [10.1023/A:1007665907178](https://doi.org/10.1023/A%3A1007665907178)
  一篇基础性的综述论文，详细阐述了变分方法的理论基础，包括ELBO的推导。
- [Variational Inference: A Review for Statisticians](https://www.tandfonline.com/doi/full/10.1080/01621459.2017.1285773) — David M. Blei, Alp Kucukelbir, Jon D. McAuliffe (2017)
  Journal: Journal of the American Statistical Association; Publisher: Taylor and Francis Ltd.; Volume: 112; Pages: 859-877; DOI: [10.1080/01621459.2017.1285773](https://doi.org/10.1080/01621459.2017.1285773)
  清晰地阐述了变分推断和ELBO，并从统计学角度进行了介绍。
