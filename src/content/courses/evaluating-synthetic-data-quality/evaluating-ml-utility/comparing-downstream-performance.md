---
course: "evaluating-synthetic-data-quality"
chapter: "evaluating-ml-utility"
lesson: "comparing-downstream-performance"
sourceId: 5668
sourceUrl: "https://apxml.com/zh/courses/evaluating-synthetic-data-quality/chapter-3-evaluating-ml-utility/comparing-downstream-performance"
title: "比较下游模型性能指标"
description: "比较在真实数据与合成数据上训练的模型所得指标（准确率、F1、AUC）的方法。"
order: 3
plots: ["plots/5668-0.json"]
sourceHash: "f00e86fb61a6fc2dd56b2705a4364a3327f38d127803e182bda3e4a0fe991ea3"
sourceCorrections: []
---

比较使用合成数据（如在“训练合成-测试真实”或 TSTR 框架中）训练的机器学习 (machine learning)模型与使用真实数据训练的基准模型的性能是重要的一步。目标是量化 (quantization)与使用原始真实数据相比，使用合成数据进行训练会损失或提升多少性能。这种分析构成了机器学习实用性评估的主要部分。

### 选择合适的性能指标

指标的选择完全取决于你正在评估的下游机器学习 (machine learning)任务的性质。没有单一的通用指标；你必须选择与模型要解决的问题相关的指标。

- **分类任务：** 对于预测类别的问题（例如，欺诈检测、图像分类），常用指标包括：

  - 准确率：正确预测的总体百分比。
  - 精确率：被识别为正例的样本中实际为正例的比例。在误报成本高时很重要。
  - 召回率（敏感度）：实际正例中被正确识别的比例。在漏报成本高时很重要。
  - F1分数：精确率和召回率的调和平均值，提供了一个平衡的衡量标准。
  - 受试者工作特征曲线下面积 (AUC-ROC)：衡量模型在不同阈值下区分类别的能力。
  - 精确率-召回率曲线下面积 (AUC-PR)：对于不平衡数据集，通常比 AUC-ROC 提供更多信息。
- **回归任务：** 对于涉及预测连续值的问题（例如，预测房价、销售预测），常用指标包括：

  - 平均绝对误差 (MAE)：预测值与实际值之间绝对差的平均值。
  - 均方误差 (MSE)：预测值与实际值之间平方差的平均值。对较大误差的惩罚更重。
  - 均方根误差 (RMSE)：MSE 的平方根，将误差恢复到原始单位。
  - R平方 ($R^2$)：决定系数，表示因变量中可由自变量预测的方差比例。

选择最能反映你特定应用成功标准的指标。通常，评估多个相关指标可以提供对模型性能的更全面认识。

### 比较过程

标准程序包括以下步骤：

1. **训练基准模型：** 使用*真实*训练数据集 ($D_{train\_real}$) 训练你选择的机器学习 (machine learning)模型（例如，逻辑回归、随机森林、神经网络 (neural network)）。
2. **训练合成模型：** 使用相同的超参数 (parameter) (hyperparameter)，训练一个相同的模型架构，但这次使用*合成*训练数据集 ($D_{train\_synth}$)。保持相同的模型配置对于公平比较很重要。
   "3. **评估两个模型：** 在*相同的保留真实测试数据集* ($D_{test\_real}$) 上评估基准模型和合成模型。这是 TSTR 的本质——你想要知道使用合成数据训练的模型对未见数据的泛化能力如何。"
3. **计算指标：** 根据两个模型在 $D_{test\_real}$ 上的预测，计算选定的性能指标（例如，准确率、F1、RMSE）。你将得到成对的分数，例如 $Accuracy_{real}$ 和 $Accuracy_{synth}$，$F1_{real}$ 和 $F1_{synth}$ 等。

### 结果解读

比较归结为分析从真实数据训练的模型和合成数据训练的模型获得的指标之间的差异。

- **直接比较：** 查看绝对差值：$Metric_{real} - Metric_{synth}$。差异很小表明合成数据很好地保持了对该任务的实用性。
- **性能比率：** 计算比率：$\frac{Metric_{synth}}{Metric_{real}}$。这提供了一个标准化视角。
  - 比率接近 1.0（例如，0.95 到 1.05）表明合成数据为训练此特定任务上的此特定模型提供了与真实数据几乎相同的实用性。
  - 比率明显低于 1.0（例如，0.8）表明使用合成数据时性能有明显下降。这种下降是否可接受取决于应用的容忍度和使用合成数据所获得的好处（例如，隐私、数据增强）。
  - 比率明显高于 1.0 是不寻常的，但如果合成数据生成过程以某种方式规范化了模型，或者比现有的真实训练数据更清楚地突出重要模式，则可能发生。这需要进一步查看。

可视化通常有助于同时比较多个指标。



![下游模型性能比较](plots/5668-0.json)



> 比较了在真实数据和合成数据上训练的模型，并在真实测试集上评估的分类指标。这里的比率对于准确率约为 0.95，F1分数约为 0.94，AUC 约为 0.96。

重复此过程可能需要使用不同的下游模型类型，这一点很重要，因为有些模型可能比其他模型对合成数据中的缺陷更敏感。一个合成数据集可能对线性模型产生好的结果，但对复杂的深度学习 (deep learning)模型表现不佳，反之亦然。

"最终，比较这些下游性能指标提供了一个有形的、面向任务的合成数据实用性衡量标准。它回答了这个问题："我可以使用这些合成数据来训练一个能在实际任务中表现良好的模型吗？""

## 参考资料

- [The synthetic data dilemma: an evaluation of the quality of synthetic tabular data using a use-case driven approach](https://doi.org/10.1016/j.is.2021.101899) — Inbar Taubman-Ben-Ari, Irad Ben-Gal, Moshe Last (2022)
  Journal: Information Systems; Publisher: Elsevier; Volume: 103; Pages: 101899; DOI: [10.1016/j.is.2021.101899](https://doi.org/10.1016/j.is.2021.101899)
  该论文提供了一个评估合成表格数据的框架，其中包括对机器学习效用和下游任务性能的讨论，与TSTR框架和指标比较相关。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGZNddlGENCnQ8H2WxXdRRouNXPdbmPVElpAcBx2faW_sJLub1QpI8xmJqAHuCj5aDi6DADV0QEcy1knHdB93HHKEftiGe75tF8qeRYgPDDXJOIDYcfNyvH-pp7ZWYCIeWL8g94KCs2sgB4eWsSmQjQtXVbe3JrLTBmVEIKbsXusP6Xpx0CIAGLyIag_EqcbbVeb3lGJWDRmDEnfDyZiYfmRRo=) — Aurélien Géron (2022)
  Publisher: O'Reilly Media, Inc.
  这是一本公认的机器学习书籍，涵盖了机器学习概念，包括分类和回归性能指标的解释及其在模型评估中的实际应用。
