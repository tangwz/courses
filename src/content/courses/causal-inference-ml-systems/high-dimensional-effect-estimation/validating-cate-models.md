---
course: "causal-inference-ml-systems"
chapter: "high-dimensional-effect-estimation"
lesson: "validating-cate-models"
sourceId: 4359
sourceUrl: "https://apxml.com/zh/courses/causal-inference-ml-systems/chapter-3-high-dimensional-effect-estimation/validating-cate-models"
title: "CATE 估计器的验证与校准"
description: "用于验证 CATE 模型性能并校准其预测的方法。"
order: 6
plots: ["plots/4359-0.json", "plots/4359-1.json"]
sourceHash: "e12aadb387d6ba5f4e88ae54d909e6a1f27e060873b65725b13838e9e440b5ee"
sourceCorrections: []
---

估计条件平均处理效应 (CATE)，即 $E[Y(1) - Y(0) | X=x]$，使用因果森林或元学习器等模型，可以提供关于处理效应异质性的有价值信息。然而，仅仅训练这些模型是不够的。必须仔细评估它们的性能，并确保其预测结果可靠。用于预测观测结果 ($Y$) 准确性的标准机器学习 (machine learning)验证技术，不直接适用于验证因果效应估计，因为真实的个体处理效应 ($Y_i(1) - Y_i(0)$) 永远无法观测。用于验证和校准 CATE 估计器的专门方法将在文中讨论。

### 难题：验证未观测的反事实

根本的困难在于任何个体单位的 CATE 缺乏真实值。我们只观测到每个单位的一个潜在结果（如果接受处理则为 $Y_i(1)$，如果未接受处理则为 $Y_i(0)$）。因此，我们不能简单地在预测的 CATE $\hat{\tau}(x_i)$ 和真实 CATE $\tau(x_i)$ 之间，在保留集上计算诸如均方误差（MSE）之类的损失函数 (loss function)。我们需要依据因果推断问题的结构来制定替代策略。

### CATE 模型验证策略

验证旨在评估我们的 CATE 模型在协变量 $X$ 定义的总体中，捕捉真实潜在处理效应异质性的表现。

#### 1. 伪结果回归

一种方法是构建“伪结果” $\tilde{Y}_i$，其在给定 $X_i$ 条件下的期望值，在特定假设下与 CATE 对应。例如，双重机器学习 (machine learning)中使用的 Robinson 变换或 R-学习器的目标函数都涉及伪结果。在无混淆假设下，并假定结果模型 $E[Y|X=x]$ 和倾向得分模型 $P(T=1|X=x]$ 估计得好，有时可以构建一个伪结果，使得 $E[\tilde{Y}_i | X_i=x] \approx \tau(x)$。然后，我们可以训练一个回归模型，从 $X_i$ 预测 $\tilde{Y}_i$，并使用交叉验证的 R 平方或 MSE 等标准技术来评估此回归。然而，这种验证的质量在很大程度上取决于用于构建伪结果的辅助模型（结果模型和倾向得分模型）的准确性。

#### 2. 子组分析（十分位数/分位数分析）

一种更直接且更具解释性的方法是根据单位的预测效应大小对它们进行分组，从而评估 CATE 预测。

- **步骤：**

  1. 在训练集上训练您的 CATE 估计器（例如，因果森林）。
  2. 使用训练好的模型对单独验证集中的单位预测 CATE，即 $\hat{\tau}(x_i)$。
  3. 根据预测的 $\hat{\tau}(x_i)$ 将验证集划分为分位数（例如，十分位数或五分位数）。预测效应最低的单位在第一个分位数，最高的在最后一个。
  4. 在每个分位数 $q$ 内，使用双重机器学习（DML）或逆处理概率加权（IPTW）等方法，*仅*应用于该分位数内的数据，来估计实际的平均处理效应（ATE）。我们将此估计记为 $\hat{ATE}_q$。
  5. 比较各个分位数中的 $\hat{ATE}_q$ 估计值。如果 CATE 模型有效捕捉异质性，我们预期会看到 $\hat{ATE}_q$ 随分位数而增加。
  6. 我们还可以将每个分位数内平均预测 CATE，$Avg(\hat{\tau}(x_i) | i \in q)$，与估计的 $\hat{ATE}_q$ 进行比较。
- **可视化：** 绘制 $\hat{ATE}_q$ 与分位数编号的图表，可以直观评估模型根据处理效应来对个体进行排序的能力。



![CATE 验证：按预测 CATE 十分位数划分的 ATE](plots/4359-0.json)



> 表现良好的 CATE 模型应显示出明显趋势，即十分位数内实际估计的 ATE（蓝色条形）随着基于预测 CATE 的十分位数而增加。每个十分位数内的平均预测 CATE（红色线/标记 (token)）应理想地与估计的 ATE 保持一致。

#### 3. 异质性评估指标

特定指标可以量化 (quantization)模型在捕捉异质性方面的表现：

- **秩相关：** 计算预测 CATE $\hat{\tau}(x_i)$ 与伪结果 $\tilde{Y}_i$（如果可用）之间的相关性（例如，Spearman 秩相关），或使用源于子组分析结果的指标。
- **增益曲线（AUUC）：** 类似于营销中的提升图，我们可以创建曲线来显示，如果按照预测 CATE 对人口进行排序，并对部分人口进行处理时的累积估计处理效应。提升曲线下面积（AUUC）对此进行概括。较高的 AUUC 表明能更好地识别那些从处理中获益最多（或最少）的个体。
- **R-损失：** 一些 CATE 估计方法，例如 R-学习器，会优化一个与 CATE 相关的特定损失函数 (loss function)。此损失可以在保留集上进行评估。$L_R(\tau) = E[(Y - E[Y|X] - \tau(X)(T - E[T|X]))^2]$。值越低越好，但解释绝对尺度可能存在困难。

#### 4. 使用合成或半合成数据

尽管这不是对真实数据的直接验证，但模拟已知真实 CATE 函数 $\tau(x)$ 的数据集，可以进行直接比较。

- **完全合成：** 基于已知的数据生成过程（包括已定义的 $\tau(x)$）模拟 $X$、$T$ 和 $Y$。
- **半合成：** 使用真实的 $X$ 数据，可能使用真实的 $T$ 数据（或基于估计的倾向得分模拟 $T$），然后基于估计的辅助函数（$E[Y|X]$、$E[T|X]$）加上一个已知的合成处理效应函数 $\tau(x)$ 模拟 $Y$。

"此方法可以计算 MSE($\hat{\tau}(x)$, $\tau(x)$) 等指标，但由于模拟过程可能存在错误指定，合成数据上的性能可能无法完美地转化为实际性能。"

### CATE 估计器的校准

校准评估预测的 CATE 值在平均意义上是否定量准确。一个校准良好的 CATE 估计器应满足：


$$
E[\tau(X) | \hat{\tau}(X) = \hat{\tau}_0] \approx \hat{\tau}_0
$$


简而言之，如果我们查看模型预测 CATE 为（例如）0.5 的所有单位，那么该群体的实际平均处理效应是否接近 0.5？

- **校准图（可靠性图）：** 类似于概率分类器的校准图，我们可以评估 CATE 校准情况：
  1. 根据单位的预测 CATE $\hat{\tau}(x_i)$ 将其分箱。
  2. 在每个箱 $b$ 内，计算平均预测 CATE，$Avg(\hat{\tau}(x_i) | i \in b)$。
  3. 在每个箱 $b$ 内，使用某种方法（例如 DML）估计实际 ATE，即 $\hat{ATE}_b$。
  4. 绘制 $\hat{ATE}_b$（Y 轴）与 $Avg(\hat{\tau}(x_i) | i \in b)$（X 轴）的图表。
  5. 一个完美校准的模型，其点将落在 $y=x$ 对角线上。



![CATE 校准图](plots/4359-1.json)



> 校准图比较了箱内平均预测 CATE 与箱内实际估计 ATE。落在虚线对角线附近的点表明校准良好。

- **事后校准：** 如果模型校准表现不佳，可以考虑事后应用如保序回归等技术来调整 $\hat{\tau}(x)$ 预测以改进校准，尽管这本身就需要谨慎处理和验证。

### 实际考量

- **识别假设：** 所有这些验证方法都依赖于潜在的识别策略（例如，因果森林/元学习器的无混淆性）。如果识别假设被违反，验证结果本身可能具有误导性。关于这些假设的敏感性分析仍然很重要。
- **辅助模型质量：** 涉及伪结果或子组 ATE 估计的方法取决于能否准确估计辅助函数（倾向得分、条件结果期望）。劣质的辅助模型可能会扭曲验证结果。
- **指标选择：** 验证指标的选择应与 CATE 模型的预期用途保持一致。如果目标是优先对预期收益最高的个体进行处理，那么 AUUC 或前十分位数表现等指标可能最相关。如果需要精确的效应预测，校准则更具意义。

验证和校准 CATE 估计器并非易事，但却是必不可少的步骤。这需要不局限于标准的预测准确度指标，而是采用专门评估不同人群中未观测反事实差异估计情况的技术。结合使用子组分析、专门指标和校准图，能更全面地反映模型性能。

## 参考资料

- [Estimation and inference for heterogeneous treatment effects using causal forests](https://www.tandfonline.com/doi/full/10.1080/01621459.2017.1411512) — Susan Athey and Stefan Wager (2019)
  Journal: Journal of the American Statistical Association; Publisher: Taylor & Francis; Volume: 114; Pages: 1229-1242; DOI: [10.1080/01621459.2017.1411512](https://doi.org/10.1080/01621459.2017.1411512)
  介绍了因果森林这一重要的CATE估计器，并讨论了评估异质性治疗效果的方法。
- [Double/debiased machine learning for treatment and causal parameters](https://doi.org/10.1111/ectj.12097) — Victor Chernozhukov, Denis Chetverikov, Mert Demirer, Esther Duflo, Christian Hansen, Whitney Newey, and James Robins (2018)
  Journal: The Econometrics Journal; Publisher: Wiley; Volume: 21; Pages: C1-C68; DOI: [10.1111/ectj.12097](https://doi.org/10.1111/ectj.12097)
  提供了双重机器学习的理论基础，与验证中的伪结果构建和稳健ATE估计相关。
- [Identifying and Validating Optimal Treatment Regimes Using Uplift Modeling](https://link.springer.com/article/10.1057/s41270-017-0030-x) — Frederik Devriendt, Bjoern Berrevoets, and Wouter Verbeke (2018)
  Journal: Journal of Marketing Analytics; Publisher: Springer Nature; Volume: 6; Pages: 1-14; DOI: [10.1057/s41270-017-0030-x](https://doi.org/10.1057/s41270-017-0030-x)
  专注于提升模型，提供了增益曲线（AUUC）及其在异质性治疗效果模型验证中的应用见解。
