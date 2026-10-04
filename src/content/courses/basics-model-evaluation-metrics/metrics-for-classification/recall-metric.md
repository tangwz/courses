---
course: "basics-model-evaluation-metrics"
chapter: "metrics-for-classification"
lesson: "recall-metric"
sourceId: 3975
sourceUrl: "https://apxml.com/zh/courses/basics-model-evaluation-metrics/chapter-2-metrics-for-classification/recall-metric"
title: "召回率（敏感度）：衡量完整性"
description: "定义召回率（敏感度），理解它衡量什么，以及如何计算它。"
order: 7
plots: []
sourceHash: "adf0cca6190f4684c7056ecafcecad52e0db1463e7c29ff98dd05973832a0022"
sourceCorrections: []
---

**召回率**，也称作**敏感度**或**真阳性率（TPR）**，衡量模型识别正例的完整程度。精确率评估模型正向预测的准确性；召回率解决的是一个不同的问题：在所有*实际*正例中，模型正确识别出了多少？

可以这样理解：如果你需要找到一些重要的事物（比如识别患有特定疾病的患者，或检测生产线上的缺陷产品），召回率衡量了你的模型在找到*所有*这些事物方面的成功程度。它衡量了模型识别正例的“完整性”。

### 召回率的计算

召回率是利用混淆矩阵中的真阳性 ($TP$) 和假阴性 ($FN$) 计算得出的。请记住：

- **真阳性 ($TP$)**: 模型正确分类为正例的阳性实例数量。
- **假阴性 ($FN$)**: 模型错误分类为负例的阳性实例数量（即模型*遗漏*的阳性实例）。

召回率的公式是：


$$
\text{召回率} = \frac{TP}{TP + FN}
$$


分母 $TP + FN$ 代表数据集中*实际*阳性实例的总数（即被正确识别的和被遗漏的）。因此，召回率提供了您的模型成功“召回”或识别出这些实际阳性实例的比例。

### 召回率的解读

召回率的取值范围是 0 到 1（或 0% 到 100%）。

- **召回率 = 1 (或 100%)**: 模型正确识别了每一个阳性实例。没有假阴性。这是在完整性方面的理想情况。
- **召回率 = 0 (或 0%)**: 模型未能识别出*任何*阳性实例。所有实际阳性实例都被分类为阴性（所有阳性实例都导致了假阴性）。
- 介于 0 和 1 之间的值表示被正确识别的实际阳性实例的比例。例如，召回率为 0.75 意味着模型找到了所有真实阳性案例的 75%。

### 何时高召回率很重要？

在未能识别出阳性案例（假阴性）会带来严重后果的情况下，优先考虑召回率非常重要。请考虑以下例子：

1. **医疗诊断**: 在筛查严重疾病时，遗漏实际患病的患者（假阴性）可能比错误地标记 (token)健康患者进行进一步检查（假阳性）危险得多。高召回率确保大多数患病者被识别出来。
2. **欺诈检测**: 让欺诈性交易未被检测到而“溜走”（假阴性）可能会造成高昂的损失。一个为高召回率优化的系统旨在捕获尽可能多的欺诈活动，即使这意味着有时会将合法交易标记为需要审查（产生更多假阳性）。
3. **垃圾邮件过滤**: 尽管令人恼火，但让一封合法邮件进入垃圾邮件文件夹（假阳性）可能是可以接受的。然而，遗漏一封重要的垃圾邮件（例如，钓鱼尝试）并让它到达收件箱（假阴性）可能会造成损害。在注重安全性的垃圾邮件检测中，高召回率可能是期望的。

### 召回率与精确率

需要理解的是，召回率侧重于找到所有*实际阳性*实例，而精确率则侧重于确保模型*预测*为阳性的实例确实是阳性。通常，提高召回率可能会导致精确率下降，反之亦然。如果您尝试捕获*每个*可能的阳性实例（高召回率），您最终可能会错误地将更多阴性实例标记 (token)为阳性（降低精确率）。这种关系被称为精确率-召回率权衡，我们将在下一节进行讨论。

总之，召回率衡量了模型识别数据集中所有相关实例的效率。它回答了这样一个问题：“在我们*本应*找到的所有事物中，我们实际找到了多少？”当遗漏阳性实例的成本很高时，它是一个重要的指标。

## 参考资料

- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://link.springer.com/book/10.1007/978-0-387-84858-7) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer; DOI: [10.1007/978-0-387-84858-7](https://doi.org/10.1007/978-0-387-84858-7)
  统计学习的标准参考书，涵盖分类指标及其理论基础。(第2版)
- [Pattern Recognition and Machine Learning](https://www.springer.com/gp/book/9780387310732) — Christopher M. Bishop (2006)
  Publisher: Springer; DOI: [10.1007/978-0-387-31073-2](https://doi.org/10.1007/978-0-387-31073-2)
  机器学习的基础教材，详细解释包括召回率在内的分类指标。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125974/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media, Inc.
  一本广泛使用的实践指南，解释分类指标、其应用以及实际场景中的权衡。(第3版)
