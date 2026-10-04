---
course: "applied-autoencoders-feature-extraction"
chapter: "applying-autoencoder-features-practical-guidance"
lesson: "integrating-autoencoder-features-supervised-models"
sourceId: 6492
sourceUrl: "https://apxml.com/zh/courses/applied-autoencoders-feature-extraction/chapter-7-applying-autoencoder-features-practical-guidance/integrating-autoencoder-features-supervised-models"
title: "将自编码器特征用于监督模型"
description: "如何将自编码器学到的特征作为分类器或回归器等监督学习算法的输入。"
order: 4
plots: []
sourceHash: "3d29bb9e10825e95efd12ed9277a14992ae4e75e7aed78a70c4aa669199a883f"
sourceCorrections: []
---

你的自编码器已经训练完成，它正在认真学习如何压缩和重建数据。瓶颈层现在包含了数据的紧凑表示，即一组学到的特征。但接下来做什么呢？这些自编码器生成的特征最常见且有效的用途之一，就是将它们输入到监督机器学习 (machine learning)模型中。这种方法可以大幅提升分类器或回归器的性能，尤其是在处理复杂、高维度数据或标注样本稀少时。

本节将指导你如何使用这些学到的特征，将你的自编码器从一个自我重建工具转变为监督学习 (supervised learning)任务中高效的特征构建帮手。

### 工作流程：从无监督特征到监督预测

将自编码器特征整合到监督学习 (supervised learning)流程中通常涉及三个阶段。首先，以无监督方式训练一个自编码器。其次，使用训练好的编码器部分来提取特征。第三，使用这些新特征训练一个标准的监督模型。

以下图表说明了这个工作流程：

> 此图显示了整体流程：训练自编码器，使用其编码器提取特征，然后使用这些特征训练监督模型。

让我们细分这些步骤：

1. **训练自编码器**：

   - 使用你的可用数据集（其中大部分可以是无标注的）来训练一个自编码器。
   - 自编码器架构的选择（简单、稀疏、去噪、卷积、VAE）取决于你的数据类型和目标，如“选择合适的自编码器类型”中讨论。
   - 目标是训练自编码器，直到它实现良好的重建性能，或者潜在空间展现出期望的属性（例如，VAE的良好分离）。
2. **提取特征**：

   - 自编码器训练完成后，分离出其编码器部分。此步骤不需要解码器。
   - 将你的数据（用于监督任务的训练集和测试集）通过这个训练好的编码器。
   - 编码器瓶颈层对每个输入样本的输出成为你的新特征集。如果你的原始数据点是$X_i$，那么它的新表示是$z_i = \text{编码器}(X_i)$。
3. **训练监督模型**：

   - 使用这些提取的特征 $Z$ 作为你所选监督学习算法（例如，逻辑回归、支持向量 (vector)机、随机森林、梯度提升机，或甚至另一个神经网络 (neural network)）的输入。
   - 你的监督任务的目标标签 $Y$ 保持不变。
   - 使用 $(Z, Y)$ 对训练此监督模型。
4. **评估性能**：

   - 使用适当的指标（分类的准确率、精确率、召回率、F1分数；回归的均方误差、R平方）评估你的监督模型的性能。
   - 最好将此性能与直接使用原始特征 $X$ 训练的基线模型进行比较，以量化 (quantization)使用自编码器衍生特征带来的好处。

### 使用自编码器特征的好处

为什么要先费力训练自编码器呢？使用其学到的特征可以提供多项好处：

- **有效降维**：自编码器可以学习将高维度数据压缩到低维度潜在空间，同时保留最显著的信息。这种减少的特征集能让监督模型训练更快、所需内存更少，有时通过缓解“维度灾难”而获得更好的泛化能力。
- **自动化特征工程**：手动设计好的特征可能耗时且需要领域知识。自编码器可以自动发现数据中的复杂模式和关系，生成可能不明显或难以手动构建的特征。
- **改进模型泛化能力**：通过强制数据通过瓶颈，自编码器学习捕捉数据的主要变动。这些表示可能比原始特征更稳定，对噪声不那么敏感，有可能使监督模型更好地泛化到未见过的数据。
- **借助无标注数据**：标注数据通常昂贵或难以获得，而无标注数据可能很丰富。自编码器可以在大量无标注数据上训练，以学习好的特征表示。这些表示随后可以在小得多的标注数据集上用于训练监督模型，有效地充当半监督学习 (supervised learning) (semi-supervised learning)或预训练 (pre-training)的一种形式。

### 实用策略与注意事项

在整合自编码器特征时，请记住以下几点：

- **结合原始特征和自编码器特征**：
  有时，通过将原始特征与自编码器学习到的特征进行拼接，可以获得最好的结果：$X_{\text{组合}} = [X_{\text{原始}}, Z_{\text{自编码器}}]$。这使得监督模型能够同时访问原始的低层信息和自编码器学习到的高层抽象。进行试验以确定这是否能提升你特定问题的性能。
- **固定特征与微调 (fine-tuning)**：
  如上所述，最直接的方法是使用“固定”特征：训练自编码器，使用固定的编码器提取特征，然后训练一个独立的监督模型。
  一种更先进的技术，特别是当你的监督模型也是神经网络 (neural network)时，是微调。在这种情况下，自编码器中已训练的编码器可以作为为监督任务设计的更大神经网络的初始层。然后，整个网络（编码器权重 (weight)从自编码器初始化，加上新的监督层）在标注数据上进行端到端训练（或“微调”）。这使得特征能够进一步适应特定的监督目标。
- **监督学习 (supervised learning)器选择**：
  经过良好训练的自编码器提取的特征通常更“机器学习 (machine learning)友好”。这意味着即使是像逻辑回归或线性支持向量 (vector)机这样的简单监督模型，在这些变换后的特征上也能表现出惊人的良好性能。然而，你也可以使用更复杂的模型，如梯度提升或神经网络分类器/回归器。选择取决于特征变换后所需的决策边界的复杂性以及你的标注数据集的大小。
- **潜在维度影响**：
  自编码器瓶颈层的维度是一个重要的超参数 (parameter) (hyperparameter)。如果它太小，你可能会丢失过多信息（欠拟合 (underfitting)）。如果它太大（对于一个欠完备自编码器），它可能无法学习到非常有用的压缩。你可能需要试验不同的潜在维度，并评估它们对下游监督任务的影响，如“调优超参数以获得最佳性能”中所述。

### 示例场景：有限标注的图像分类

设想你有一个图像分类任务，但你只有一小部分标注图像。然而，你可以获得大得多且类型相似的无标注图像集合。

1. 你可以在大量无标注图像上训练一个卷积自编码器 (CAE)。CAE将学习提取分层视觉特征。
2. CAE训练完成后，取出其编码器部分。
3. 对于你的一小部分标注图像，将它们通过这个训练好的CAE编码器，以获得它们的潜在特征向量 (vector)。
4. 然后，使用这些潜在向量及其对应标签，训练一个相对简单的分类器（例如，一个小型多层感知机或甚至逻辑回归）。

这种方法通常比仅在小型标注数据集上从头开始训练一个大型卷积神经网络 (neural network) (CNN) 产生更好的分类性能，因为CAE从丰富的无标注数据中学到的特征提供了一个强有力的起点。

### 获取编码器特征：一个高层级代码思路

在大多数深度学习 (deep learning)框架中，一旦自编码器模型训练完成，其编码器部分可以被分离或直接用于进行预测。这是一个通用思路，不特定于任何库：

```
# 假设 'full_autoencoder_model' 是你训练好的自编码器
# 假设 'encoder_part' 是仅代表编码器层的模型

# 1. 获取编码器模型
# 这可能涉及创建一个与自编码器共享层的新模型
# 或者如果你的自编码器类支持，直接访问预定义的编码器属性。
# 例如，如果你的自编码器有一个输入层 'ae_input'
# 并且瓶颈层是 'bottleneck_output_layer':
# encoder_model = create_model(inputs=ae_input, outputs=bottleneck_output_layer)

# 2. 为监督任务准备数据
# X_train_supervised, X_test_supervised 是你的原始数据集

# 3. 使用编码器提取特征
# new_features_train = encoder_model.predict(X_train_supervised)
# new_features_test = encoder_model.predict(X_test_supervised)

# 4. 训练你的监督模型
# supervised_ml_model = SomeSupervisedAlgorithm()
# supervised_ml_model.fit(new_features_train, y_train_supervised)

# 5. 评估
# performance = supervised_ml_model.score(new_features_test, y_test_supervised)
```

具体语法将取决于所使用的库（TensorFlow/Keras, PyTorch）。例如，在Keras中，你可能需要定义一个新的`Model`对象，它接收自编码器的输入并输出瓶颈层的激活。

通过将原始数据转换为更有效的特征表示，自编码器作为一个有价值的预处理步骤，使监督模型能够更有效地学习并取得更好的结果。正如你在实践练习中将看到的，这种整合可以在你的机器学习 (machine learning)项目中带来显著的不同。

## 参考资料

- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  全面介绍自编码器、表征学习和神经网络架构的教科书。
- [Reducing the Dimensionality of Data with Neural Networks](https://www.science.org/doi/10.1126/science.1127647) — Geoffrey E. Hinton, Ruslan R. Salakhutdinov (2006)
  Journal: Science; Publisher: American Association for the Advancement of Science; Volume: 313; Pages: 504-507; DOI: [10.1126/science.1127647](https://doi.org/10.1126/science.1127647)
  一篇开创性论文，展示了深度自编码器如何为高维数据学习有效的低维表征，适用于预训练。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781098125968/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  一本实用指南，包含关于自编码器、无监督学习以及使用流行深度学习框架进行特征提取的章节。
