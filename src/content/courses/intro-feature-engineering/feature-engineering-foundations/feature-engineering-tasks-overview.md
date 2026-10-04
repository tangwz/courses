---
course: "intro-feature-engineering"
chapter: "feature-engineering-foundations"
lesson: "feature-engineering-tasks-overview"
sourceId: 1307
sourceUrl: "https://apxml.com/zh/courses/intro-feature-engineering/chapter-1-feature-engineering-foundations/feature-engineering-tasks-overview"
title: "特征工程任务概览"
description: "介绍特征工程的主要分类：数据清洗、转换、创建和选择。"
order: 5
plots: []
sourceHash: "288223e3159482b76049c987e27e4d4a6948ab911b28b31888e3aae640b94430"
sourceCorrections: []
---

在明确了特征的重要性及其在机器学习 (machine learning)工作流程中的位置之后，让我们概述一下特征工程中包含的主要任务。可以将特征工程视为一系列用于为模型准备数据的技术，而不仅仅是一个独立的步骤。最终目标是构建能够有效捕捉与待解决问题相关的潜在规律的特征，从而提高模型性能和泛化能力。

这些任务通常分为四大类：

1. **数据准备与清洗：** 原始数据很少是完美无缺的。它通常包含缺失值或异常值，这些可能干扰模型训练或导致有偏差的结果。此阶段包括识别这些问题并应用处理策略。

   - *处理缺失数据：* 绝大多数机器学习算法无法处理含有缺失条目的数据集。技术范围从简单的填充（用均值、中位数或众数填充缺失值）到更复杂的方法，例如K近邻（KNN）填充或基于模型的迭代填充。我们还可以创建指示特征来标记 (token)数据原始缺失的位置，因为数据的缺失本身就可能提供信息。
   - *异常值处理：* 极端值，即异常值，可能对某些模型产生不成比例的影响，特别是那些对方差或距离计算敏感的模型。识别并决定如何处理异常值（例如，删除它们、转换特征、使用抗干扰算法）是一个重要的准备步骤。
2. **特征转换：** 许多算法对输入数据的格式或分布有特定要求。转换会修改特征以满足这些要求或提升模型性能。

   - *缩放：* 依赖距离计算（如KNN或支持向量 (vector)机）或梯度下降 (gradient descent)优化（如线性回归或神经网络 (neural network)）的算法，当数值特征处于相似的尺度时，通常表现更佳。标准化（Z-score缩放）和归一化 (normalization)（Min-Max缩放）等技术可以实现这一点。对于存在显著异常值的数据，也有可用的缩放方法。
   - *编码分类特征：* 模型处理的是数字，而非文本类别。将分类特征（如“颜色”或“国家”）转换为数值表示是必需的。常用方法包括用于名义特征（无固有顺序）的独热编码和用于序数特征（有序类别）的序数编码。对于高基数特征，会使用目标编码或哈希编码等更高级的技术。
   - *分布转换：* 某些模型要求特征遵循特定分布（通常是高斯分布）。对数转换、Box-Cox或Yeo-Johnson转换等技术可以帮助归一化偏斜数据，从而可能提高模型的稳定性和性能。
3. **特征创建（或生成）：** 有时，最有信息量的特征并未在原始数据集中。特征创建包括从现有特征中衍生出新特征，以捕捉更复杂的关联或融入领域知识。

   - *交互项：* 组合两个或更多特征（例如，相乘或相除）可以捕捉单独特征无法表达的共同影响。
   - *多项式特征：* 创建现有数值特征的平方或更高阶项，使得线性模型能够捕捉非线性关系。
   - *领域特定特征：* 善用问题所属领域的知识可以得到信息量丰富的特征（例如，从“出生日期”创建“年龄”，计算金融数据中的比率，或从时间戳中提取“星期几”等组成部分）。
   - *分箱：* 将连续数值特征转换为离散的分类箱有时可以简化关系或捕捉非线性效应。
4. **特征选择：** 并非特征越多越好。包含不相关或冗余的特征会增加模型复杂性，导致过拟合 (overfitting)（对新数据的泛化能力差），增加计算成本，并使模型更难解释。特征选择旨在识别并保留最相关的特征子集。

   - *过滤方法：* 这些方法根据特征的内在属性（如方差或与目标变量的相关性）独立于任何特定模型来评估特征。例子包括方差阈值法、相关性分析和统计检验（方差分析、卡方检验）。
   - *包裹方法：* 这些方法利用特定的机器学习模型，根据模型的性能评估特征子集。例子包括递归特征消除（RFE）和序列特征选择（SFS）。
   - *嵌入 (embedding)方法：* 特征选择被集成到模型训练过程本身。例子包括Lasso (L1) 正则化 (regularization)（它会惩罚大系数并可能将某些系数缩减为零，从而有效移除特征），以及从基于树的模型（如随机森林或梯度提升）中获得的特征重要性。

> 该图呈现了特征工程任务的典型流程和主要分类，将原始数据转换为适合机器学习模型的优化特征。虚线表示该过程通常是迭代的，而非严格按顺序进行。

值得注意的是，这些任务并非总是严格按线性顺序执行。特征工程通常是一个迭代过程，在一个步骤中获得的见解可能促使你重新审视之前的步骤。例如，创建交互特征可能需要重新缩放，或者特征选择可能在某些转换之前发生。

接下来的章节将详细解释这些类别中的各项技术，并附带使用Pandas和Scikit-learn等Python库的实际示例。理解这些核心任务为在任何机器学习项目中进行数据准备和特征构建提供了一个扎实的框架。

## 参考资料

- [Applied Predictive Modeling](https://link.springer.com/book/10.1007/978-1-4614-6849-3) — Max Kuhn and Kjell Johnson (2013)
  Publisher: Springer; DOI: [10.1007/978-1-4614-6849-3](https://doi.org/10.1007/978-1-4614-6849-3)
  全面介绍数据预处理和特征工程技术，注重实际应用。
- [Feature Engineering for Machine Learning: Principles and Techniques for Data Scientists](https://learning.oreilly.com/library/view/feature-engineering-for/9781491953260/) — Alice Zheng, Amanda Casari (2018)
  Publisher: O'Reilly Media
  专为数据科学家提供的资源，涵盖多种特征工程原理和方法。
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html) — Scikit-learn developers (2024)
  官方文档，提供在Python中实现许多特征工程工具的实用指南和API参考。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) — Aurélien Géron (2019)
  Publisher: O'Reilly Media
  提供使用流行Python机器学习库实现特征工程任务的实际示例和代码。（第2版）
