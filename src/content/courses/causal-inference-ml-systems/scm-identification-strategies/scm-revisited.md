---
course: "causal-inference-ml-systems"
chapter: "scm-identification-strategies"
lesson: "scm-revisited"
sourceId: 4296
sourceUrl: "https://apxml.com/zh/courses/causal-inference-ml-systems/chapter-1-scm-identification-strategies/scm-revisited"
title: "因果结构模型回顾"
description: "复习因果结构模型（SCM）的表示法和假定，以用于高级应用。"
order: 1
plots: []
sourceHash: "e5f8eb7e1c1ba103f5c7d92a2af2ca3a634047901f77ded722eb37484e55f1e2"
sourceCorrections: []
---

因果结构模型（SCM）的框架被正式化。对SCM及其相关的有向无环图（DAG）的透彻定义，对于应用高级识别策略和估计方法是必要的。SCM提供了数学工具，可以明确表示因果假定，定义干预，并最终确定是否可以从现有数据中估计因果效应。

### 定义因果结构模型

SCM完整描述了系统变量如何获得其值。形式上，因果结构模型 $M$ 是一个元组 $M = \langle U, V, F, P(U) \rangle$，其中：

1. **$U$** 是一个**外生变量**集。这些变量由模型外部的因素决定。它们代表随机背景条件、噪声或未建模的影响。可以把它们看作系统随机性的根本来源。
2. **$V$** 是一个**内生变量**集。这些变量的值由模型*内部*的其他变量（包括内生和外生变量）决定。它们通常是我们旨在理解其关系的变量，例如处理、结果和混杂因素。
3. **$F$** 是一个**结构方程**集，其中包含针对每个内生变量 $V_i \in V$ 的一个方程。每个方程将 $V_i$ 的值指定为模型中其他变量（包括内生变量 ($PA_i \subseteq V \setminus \{V_i\}$) 和外生变量 ($U_i \subseteq U$)）的函数 $f_i$：
   
   $$
   V_i = f_i(PA_i, U_i)
   $$
   
   此处，$PA_i$ 表示 $V_i$ 的内生父变量集，即根据函数 $f_i$ 直接影响 $V_i$ 的 $V$ 中的变量。与 $V_i$ 相关联的 $U_i$ 代表该变量确定过程特有的外生影响，捕捉了任何未建模的因素或固有的随机性。
4. **$P(U)$** 是外生变量 $U$ 上的**概率分布**。此分布捕捉了影响系统的背景因素之间的潜在相关性和依赖性。

重要地，每个方程 $V_i = f_i(PA_i, U_i)$ 都代表一个自主的因果机制。它描述了 $V_i$ 如何根据其直接原因和特定的外生因素来确定，并且独立于控制其他变量的机制。

### 假定及其影响

SCM定义隐含着重要的假定：

- **隐式因果充分性：** 标准定义假定，对于任何一对内生变量 $V_i, V_j$，任何共同原因要么包含在 $V$ 中，要么其影响通过 $P(U)$ 中的相关性得到体现。如果我们进一步假定外生变量 $U_i$ 相互独立 ($P(U) = \prod P(U_i)$)，则模型假定内生变量 $V$ 之间*没有*未观察到的共同原因。这个强假定在实践中通常不切实际，第4章讨论了处理违背情况（未观察到的混杂）的方法。如果 $U_i$ *不*被假定为独立，则它们的依赖结构代表潜在混杂变量。
- **无环性（通常）：** 大多数入门教程和许多高级方法都假定 $F$ 中编码的因果关系不形成有向环。也就是说，一个变量不能是它自己的祖先。这意味着内生变量 $V$ 之间的关系可以用有向无环图（DAG）表示。我们将在本章后面（“因果图中的环和反馈处理”）具体讨论带环和反馈的模型。
- **模块化/自主性：** 这是一个根本假定。它提出，每个机制 $f_i$ 都是独立的，并且原则上可以更改（例如，通过干预）而不影响其他机制 $f_j$（对于 $j \neq i$）。这一特性使我们能够精确定义和分析干预的效果。

### 从SCM到图和分布

SCM自然地引出一个**因果图**（通常是有向无环图）。该图的节点对应于内生变量 $V$。当且仅当 $V_j$ 是函数 $f_i$ 中的一个参数 (parameter)（即 $V_j \in PA_i$）时，存在从 $V_j$ 到 $V_i$ 的有向边。从 $V_j$ 到 $V_i$ **没有**边，编码了 $V_j$ 对 $V_i$ 没有直接因果效应这一强的因果假定。

考虑一个简单的SCM：
$U = \{U_X, U_Y, U_Z\}$ 假定独立且服从标准正态分布。
$V = \{X, Y, Z\}$
$F = \{$
$X = f_X(U_X) = U_X$,
$Z = f_Z(X, U_Z) = \alpha X + U_Z$,
$Y = f_Y(X, Z, U_Y) = \beta X + \gamma Z + U_Y$
$\}$
$P(U)$ 指定了这些分布（例如，标准正态分布，独立）。

这个SCM引出以下图：

> 该图显示了内生变量之间的直接因果影响（实线箭头），并表明了外生影响（虚线，当U被假定为独立时通常省略）。

一个SCM $M$ 在内生变量上定义了一个唯一的**观测概率分布** $P(V)$。此分布是通过结构方程 $F$ 传播 $P(U)$ 中的不确定性而产生的。

### SCM中的干预：do算子

SCM的能力在于它们能够正式地建模干预。干预，由 $do$ 算子表示，代表一种外部行动，它强制变量 $X \in V$ 接受特定值 $x$，从而覆盖其自然的因果机制。

形式上，对SCM $M = \langle U, V, F, P(U) \rangle$ 执行干预 $do(X=x)$ 会创建一个**修改后的SCM**，表示为 $M_x = \langle U, V, F_x, P(U) \rangle$。唯一的区别在于函数集 $F_x$：变量 $X$ 的原始方程 $X = f_X(PA_X, U_X)$ 被替换为常数赋值 $X=x$。$F$ 中的所有其他结构方程保持不变，这反映了模块化假定。

**干预分布** $P(Y | do(X=x))$（或更一般地，干预后分布 $P_{M_x}(V)$）是由此修改后的模型 $M_x$ 产生的变量分布。

理解 $P(Y | do(X=x))$ 通常在数值上不同于观测条件分布 $P(Y | X=x)$ 是很重要的。

- $P(Y | X=x)$ 描述了在 $X$ *恰好*为 $x$ 的子群体中 $Y$ 的分布，它是从原始SCM $M$ 计算得出的。这反映了被动观察。
- $P(Y | do(X=x))$ 描述了当我们*强制* $X$ 为 $x$ 时 $Y$ 的分布，它是从修改后的SCM $M_x$ 计算得出的。这反映了主动干预。

在我们的示例图中，$P(Y | X=x)$ 受路径 $X \rightarrow Z \rightarrow Y$ 和直接路径 $X \rightarrow Y$ 的影响。如果 $X$ 和 $Y$ 共享一个由相关的 $U_X, U_Y$ 代表的未观察到的共同原因，那么 $P(Y | X=x)$ 也会反映这种非因果相关性。然而，$P(Y | do(X=x))$ 仅反映从被干预的 $X$ 沿路径 $X \rightarrow Y$ 和 $X \rightarrow Z \rightarrow Y$ 流出的因果影响。它是在 $X$ 的值被固定为 $x$ 的模型中计算的，切断了指向 $X$ 的任何传入箭头。

SCM框架中对干预的这种精确定义是因果效应识别的依据。核心问题变为：我们能否仅使用观测分布 $P(V)$ 和关于因果结构（图）的假定，而无需完全了解函数 $f_i$ 或分布 $P(U)$，来计算 $P(Y | do(X=x))$？do-演算（接下来将详细介绍）为此问题提供了正式规则。掌握SCM形式化是应用这些强大工具的必要第一步。

## 参考资料

- [Causality: Models, Reasoning, and Inference](https://www.cambridge.org/core/books/causality/326BAA41122DD5D3A20B7587F53860F5) — Judea Pearl (2009)
  Publisher: Cambridge University Press; Pages: 484; DOI: [10.1017/CBO978051181881CA](https://doi.org/10.1017/CBO978051181881CA)
  关于结构因果模型、因果图和 do-演算的基础文本，提供了因果推断的数学和哲学基础。
- [Causal Inference in Statistics: A Primer](https://www.wiley.com/en-us/Causal+Inference+in+Statistics%3A+A+Primer-p-9781119186861) — Judea Pearl, Madelyn Glymour, Nicholas P. Jewell (2016)
  Publisher: John Wiley & Sons; DOI: [10.1002/9781119186846](https://doi.org/10.1002/9781119186846)
  结构因果模型、DAG 和干预措施的易于理解的入门，适合侧重于形式化定义的读者。
- [Elements of Causal Inference: Foundations and Learning Algorithms](https://mitpress.mit.edu/books/elements-causal-inference) — Jonas Peters, Dominik Janzing, Bernhard Schölkopf (2017)
  Publisher: The MIT Press; DOI: [10.7551/mitpress/9780262037310.001.0001](https://doi.org/10.7551/mitpress/9780262037310.001.0001)
  本书将因果推断基础与机器学习相结合，涵盖了结构因果模型及其在理解数据生成过程中的作用。
