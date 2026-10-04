---
course: "intro-feature-engineering"
chapter: "encoding-categorical-features"
lesson: "nominal-ordinal-categories"
sourceId: 1325
sourceUrl: "https://apxml.com/zh/courses/intro-feature-engineering/chapter-3-encoding-categorical-features/nominal-ordinal-categories"
title: "标称类别与序数类别"
description: "区分标称分类变量和序数分类变量及其编码影响。"
order: 2
plots: []
sourceHash: "819eccf3705d0fe2bfa73eb5ed1de255d85c3daa53cb01593e733775eea27c7e"
sourceCorrections: []
---

在应用任何编码技术之前，弄清分类特征本身具有不同形式是很重要的。概括来说，我们可以将它们分为两种主要类型：**标称类别**和**序数类别**。认识到这种区别是根本的，因为最佳编码策略通常取决于您正在处理的分类数据类型。使用不恰当的编码方法可能会向数据集中引入误导性信息，或者未能获取特征中包含的有价值的结构信息。

### 标称类别

标称类别表示不同的组或标签，其值之间**不存在固有的顺序或排名**。可以将它们视为定性分类。例如：

- **颜色：** 红色、绿色、蓝色、黄色
- **国家名称：** 美国、加拿大、墨西哥、英国
- **产品类型：** 笔记本电脑、平板电脑、智能手机
- **性别：** 男性、女性、非二元

对标称特征而言，对类别进行任何数值分配都纯粹是任意的。将 `1` 赋值给“红色”，将 `2` 赋值给“绿色”，将 `3` 赋值给“蓝色”，并不意味着蓝色“大于”绿色，也不意味着红色和绿色之间的差异与绿色和蓝色之间的差异相同。机器学习 (machine learning)算法，特别是线性模型或基于距离的算法（如 k-最近邻），可能会误解此类数值分配，假设原始数据中根本不存在的顺序或量级差异。这可能导致不正确的假设，并可能降低模型性能。

### 序数类别

另一方面，序数类别在其值之间具有**有意义的顺序或排名**，但连续类别之间的**差异大小**不一定已知、统一或可量化 (quantization)。顺序很重要，但对类别进行算术运算通常没有意义。例如：

- **教育水平：** 高中、学士、硕士、博士
- **客户满意度：** 非常不满意、不满意、中立、满意、非常满意
- **服装尺码：** 小、中、大、特大
- **服务等级：** 青铜、白银、黄金、白金

在这里，我们知道“硕士”高于“学士”，“满意”代表比“中立”更好的结果。然而，我们通常不能假设“高中”和“学士”之间的“差距”与“硕士”和“博士”之间的差距相同。为这些级别分配 `1, 2, 3, 4` 等数值可以捕获顺序，这对于某些模型来说是宝贵的信息。然而，仍需保持谨慎，因为模型可能会按字面意思解释数值差异（例如，假设级别 1 和 2 之间的差异与级别 3 和 4 之间的差异完全相同）。

### 为什么区别对编码很重要

弄清特征是标称的还是序数的，直接影响您选择编码策略：

1. **保存信息：** 目标是以数值方式表示分类信息，而不引入伪影或丢失现有结构。
2. **避免虚假顺序：** 对标称数据使用为序数数据设计的编码方法（如分配连续整数）可能会引入虚假的顺序感，从而混淆学习算法。例如，将 `['USA', 'Canada', 'Mexico']` 编码为 `[1, 2, 3]` 可能会导致线性模型错误地推断墨西哥具有美国所代表的某种属性的三倍。
3. **捕获现有顺序：** 对序数特征使用主要为标称数据设计的方法（如独热编码）可能会导致模型遗漏固有的排名信息。尽管这通常是一种安全的方法，但如果顺序本身是一个强预测因子，它可能不是最佳的。
4. **算法敏感性：** 不同的机器学习 (machine learning)算法反应不同。基于树的模型（如决策树、随机森林）通常可以有效处理原始序数编码（例如 1、2、3），因为它们根据阈值（`feature <= 2`）进行分割。然而，线性模型、支持向量 (vector)机和神经网络 (neural network)对数值尺度以及编码值之间隐含的关系更敏感。

> 确定分类特征是标称的还是序数的，有助于选择合适的编码技术。

确定类型通常需要检查特征的唯一值，并运用关于这些值所代表内容的领域知识。是否存在逻辑上的进展或层级关系？如果是，则很可能是序数类别；否则，就是标称类别。

考虑到这一区别，以下部分将介绍具体的编码技术，讨论它们最适合哪种数据类型，并使用常见的 Python 库演示它们的实现。我们将从通常用于标称数据的方法开始，然后介绍可以处理序数关系或解决高基数等问题的方法。

## 参考资料

- [Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  一本实用的指南，涵盖机器学习基础知识，包括数据预处理和分类特征编码，并清晰解释数据类型。
- [Feature Engineering for Machine Learning: Principles and Techniques for Data Scientists](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/) — Alice Zheng, Amanda Casari (2018)
  Publisher: O'Reilly Media
  这本书直接讨论特征工程策略，并提供专门章节介绍理解和编码不同类型的分类数据。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer
  一本基础教材，提供机器学习严谨的统计学处理方法，包括对数据类型及其对模型构建影响的讨论。(第二版)
- [Preprocessing data](https://scikit-learn.org/stable/modules/preprocessing.html) — Scikit-learn Developers (2024)
  Scikit-learn预处理模块的官方文档，为分类特征编码以及名义和序数数据的不同处理方法提供实用指导。
