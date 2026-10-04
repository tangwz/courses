---
course: "rnns-and-sequence-modeling"
chapter: "evaluating-tuning-sequence-models"
lesson: "troubleshooting-common-training-issues"
sourceId: 2651
sourceUrl: "https://apxml.com/zh/courses/rnns-and-sequence-modeling/chapter-10-evaluating-tuning-sequence-models/troubleshooting-common-training-issues"
title: "常见训练问题的排查"
description: "诊断循环神经网络训练中收敛缓慢、发散或泛化能力差等问题。"
order: 7
plots: ["plots/2651-0.json"]
sourceHash: "b53f3d3870f4a711c289696c274053677320267ab11e8bdfbb6c52d550d01cb6"
sourceCorrections: []
---

即使架构设计得当，数据也经过细致准备，训练循环神经网络 (neural network) (RNN)有时仍会让人觉得更像一门艺术而非科学。当模型学习效果不佳时，系统地诊断问题非常重要。本节介绍了循环神经网络训练中遇到的常见问题，并提供了实际的解决方法。

## 损失停滞或下降缓慢

一个常见的情况是训练损失最初下降，但随后在高值处趋于平稳，或者下降速度非常缓慢，以至于许多训练周期后进展可以忽略不计。

**表现：**

- 训练初期损失曲线趋于平坦。
- 训练集上的性能指标（准确率，$MAE$ 等）改进甚微。

**可能原因：**

- **学习率过低：** 如果梯度下降 (gradient descent)过程中步长太小，收敛会非常缓慢。
- **梯度消失：** 正如第四章所说，梯度在时间反向传播 (backpropagation)时会呈指数级缩小，尤其是在简单循环神经网络 (neural network) (RNN)或非常深的网络中。这会阻止与早期时间步相关的权重 (weight)得到有效更新。即使是 LSTM/GRU 在极长序列上也会受到此影响。
- **权重初始化不佳：** 糟糕的初始权重可能使模型处于不良区域，使得优化器难以找到好的解决方案。
- **数据问题：** 数据可能缺少足够的信号、包含噪声或未正确缩放，从而阻碍学习过程。
- **模型容量不足：** 网络可能过于简单（单元或层数太少），无法捕捉数据中的内在规律。

**排查步骤：**

1. **调整学习率：** 尝试提高学习率（例如，增加 3 倍或 10 倍）。密切关注损失；如果它开始振荡或增加，学习率可能过高。可以考虑使用学习率调度（例如，在一定数量的训练周期后或验证损失趋于平稳时自动降低学习率）。
2. **检查梯度（在框架允许的情况下）：** 一些深度学习 (deep learning)框架允许您在训练期间查看梯度的量级。如果早期层或循环连接的权重梯度持续接近零，梯度消失是很可能的原因。
3. **切换/调整架构：** 如果简单循环神经网络用于可能存在长依赖关系的任务，请切换到 LSTM 或 GRU。如果已经在使用 LSTM/GRU，请确保它们配置得当（例如，有足够数量的单元）。
4. **改进初始化：** 尝试不同的权重初始化策略，例如 Glorot (Xavier) 或 He 初始化，它们通常比简单的随机正态或均匀分布是更好的默认选择。
5. **审查数据预处理：** 仔细检查数值特征是否已适当缩放（例如，归一化 (normalization)为零均值和单位方差，或缩放至 [0, 1] 或 [-1, 1] 范围）。确保文本分词 (tokenization)、填充和掩码正确。
6. **增加模型容量：** 尝试增加循环层中隐藏单元的数量或堆叠更多层。逐渐进行此操作并监控验证性能以避免过拟合 (overfitting)。

## 损失爆炸或变为 NaN

相反的问题是发散，即损失突然飙升至极高值，甚至更糟，变为 `NaN`（非数字），完全停止训练。

**表现：**

- 损失值在每个训练周期内迅速增加。
- 损失变为 `Infinity` 或 `NaN`。

**可能原因：**

- **学习率过高：** 过大的学习步长可能导致优化器越过最小值，并使损失不受控制地增加。
- **梯度爆炸：** 与梯度消失相对。梯度在反向传播 (backpropagation)过程中可能呈指数级增长，导致巨大的权重 (weight)更新，使网络不稳定。这在循环神经网络 (neural network) (RNN)中尤其常见。
- **数据问题：** 输入数据中的极端异常值或未正确缩放的特征可能导致非常大的激活和梯度。
- **数值不稳定：** 对非正数取对数或除以零等操作，可能由特定激活函数 (activation function)与某些输入值或中间状态结合引起。

**排查步骤：**

1. **降低学习率：** 这通常是首要且最有效的步骤。显著降低学习率（例如，除以 10 或 100）。
2. **实施梯度裁剪：** 正如第四章所介绍，梯度裁剪阻止梯度超过特定阈值。这是稳定循环神经网络训练的标准技术。设置一个裁剪值（例如，1.0，5.0），并在训练循环中或通过框架选项应用它。
3. **检查数据缩放和异常值：** 仔细检查您的输入数据。确保特征已适当缩放。识别并处理极端异常值（例如，通过裁剪值或使用更稳定的缩放方法）。
4. **验证损失函数 (loss function)和激活函数：** 确保您的损失函数数值稳定。如果使用 `tanh` 或 `sigmoid` 等激活函数，输出通常有界。如果使用 ReLU，激活值可能变得很大；如果可能，请检查中间值。在自定义代码中查找潜在的除以零或对零/负数取对数的情况。
5. **检查批量数据：** 有时，单个包含异常值的坏数据批量可能引发发散。尝试使用批量大小为 1 进行训练，或手动检查在爆炸发生前输入到模型中的数据批量。

## 训练性能高，验证/测试性能差（过拟合 (overfitting)）

机器学习 (machine learning)中的一个常见问题是过拟合，即模型对训练数据学习得非常好，包括其中的噪声和特殊性，但未能泛化到新的、未见过的数据。



![典型过拟合情景](plots/2651-0.json)



> 训练损失下降而验证损失开始增加，表明模型过拟合。

**可能原因：**

- **模型复杂度：** 模型容量过大（相对于数据量而言参数 (parameter)过多），本质上是记忆了训练样本。
- **数据不足：** 没有足够多样的训练样本来学习可泛化模式。
- **训练时间过长：** 训练周期过多导致模型拟合了训练数据中的噪声。

**排查步骤：**

1. **应用正则化 (regularization)：**
   - **Dropout：** 引入 dropout 层。对于循环神经网络 (neural network) (RNN)，使用框架的循环感知型 dropout 变体，它们对给定序列的所有时间步应用相同的 dropout 掩码。这可以防止 dropout 干扰循环状态传播。常用比率介于 0.2 和 0.5 之间。将其应用于循环层的输入和/或输出，以及可能在堆叠的循环层之间。
   - **权重 (weight)正则化 (L1/L2)：** 在损失函数 (loss function)中添加 L1 或 L2 惩罚，鼓励权重更小。与 dropout 相比，这对循环权重本身的影响通常较小，但如果适用，可以应用于循环单元内的输入/输出层或前馈连接。
2. **降低模型复杂度：** 减少 LSTM/GRU 层中隐藏单元的数量或减少堆叠层的数量。
3. **使用早期停止：** 在训练期间监控验证损失（或另一个相关的验证指标）。当验证性能停止提升或在预设的训练周期数（容忍度）内开始下降时，停止训练。保存验证性能最佳的训练周期的模型权重。
4. **获取更多数据：** 如果可行，增加训练数据集的大小和多样性。序列数据增强可能存在困难，但可能涉及文本的反向翻译或时间序列的添加噪声等技术。
5. **检查数据泄露：** 确保在预处理过程中，来自验证集或测试集的任何信息都没有无意中混入训练集。

## 模型在长序列上表现不佳

有时，循环神经网络 (neural network) (RNN)模型在较短序列上表现尚可，但随着序列长度增加，表现明显变差。

**表现：**

- 对于长序列，评估指标比短序列差很多。
- 模型未能捕捉序列中距离较远的元素之间的依赖关系。

**可能原因：**

- **梯度消失：** 即使序列极长或超参数 (parameter) (hyperparameter)不理想，LSTM 和 GRU 也并非完全不受影响。它们在长时间内传递信息的能力可能仍会下降。
- **简单循环神经网络的使用：** 如果您使用的是简单循环神经网络，它本身在捕捉长距离依赖关系方面就有限。
- **模型容量/内存不足：** 隐藏状态的大小可能太小，无法编码来自遥远过去的全部相关信息。
- **训练数据限制：** 模型在训练期间可能没有见过足够多表现出长距离依赖关系的例子。

**排查步骤：**

1. **使用 LSTM 或 GRU：** 如果尚未这样做，请从简单循环神经网络切换到 LSTM 或 GRU 单元，它们专门设计用于处理更长的依赖关系。
2. **增加模型容量：** 尝试增加隐藏单元的数量或堆叠更多循环层。
3. **实施双向循环神经网络：** 如果序列后续部分的上下文 (context)有助于解释早期部分（在情感分析或标注等自然语言处理任务中常见），使用双向 LSTM 或 GRU 可以显著提高性能，因为它以正向和反向处理序列。
4. **检查序列长度处理：** 确保训练期间使用的最大序列长度合适。如果测试序列比训练序列长得多，模型可能无法很好地泛化。如果序列极长且无法适应内存，可以考虑在更长的序列上训练，或仔细使用有状态循环神经网络或截断时间反向传播 (backpropagation) (TBPTT) 等技术。
5. **注意力机制 (attention mechanism)（进阶）：** 对于需要关注非常长距离的特定过去信息的任务（如机器翻译），注意力机制（第九章有简要介绍，通常与 Transformer 一起使用）为模型访问相关的过去状态提供更直接的方式，从而缓解与距离相关的梯度消失问题。

## 多次运行结果不一致

当多次运行相同的训练脚本产生明显不同的结果时，可能会令人沮丧，从而难以可靠地评估变化。

**表现：**

- 相同训练运行之间，最终模型性能差异很大。

**可能原因：**

- **随机性：** 典型的深度学习 (deep learning)流程中存在几种随机性来源：
  - 随机权重 (weight)初始化。
  - 每个训练周期对训练数据进行随机洗牌。
  - Dropout 层在训练期间随机将激活值设为零。
  - 某些 GPU 操作 (CUDA/cuDNN) 可能出于性能原因而具有非确定性行为。

**排查步骤：**

1. **设置随机种子：** 在脚本开始时，为所有随机性来源设置固定种子：
   - Python 的内置 `random` 模块。
   - NumPy (`numpy.random.seed`)。
   - 您的深度学习框架（TensorFlow：`tf.random.set_seed`，PyTorch：`torch.manual_seed`，如果使用 GPU 则为 `torch.cuda.manual_seed_all`）。
2. **控制数据洗牌：** 确保数据洗牌（如果执行）是确定性地完成的（例如，通过为洗牌操作设置种子或初始时使用固定种子洗牌一次）。
3. **评估时禁用 Dropout：** 确保在评估验证集或测试集上的模型性能时关闭 dropout 层。调用 `model.evaluate()` 或将模型设置为评估模式（例如，PyTorch 中的 `model.eval()`）时，框架通常会自动处理此问题。
4. **GPU 确定性（可选）：** 如果可复现性很重要，请研究特定于框架的标志或环境变量以强制执行确定性 GPU 操作（例如，TensorFlow 的 `TF_DETERMINISTIC_OPS=1`，PyTorch 的 `torch.backends.cudnn.deterministic = True`）。请注意，这有时可能会对性能产生负面影响。

诊断训练问题是一个迭代过程。使用本章前面讨论的评估指标和可视化技术来密切监控模型的行为。通过系统地识别表现，考虑可能的原因，并应用这些排查步骤，您可以显著提高训练出有效且性能稳定的序列模型的可能性。

## 参考资料

- [Long Short-Term Memory](https://www.mitpressjournals.org/doi/10.1162/neco.1997.9.8.1735) — Sepp Hochreiter, Jürgen Schmidhuber (1997)
  Journal: Neural Computation; Publisher: The MIT Press; Volume: 9; Pages: 1735-1780; DOI: [10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735)
  介绍了长短期记忆 (LSTM) 网络，一种旨在解决循环神经网络中梯度问题并学习长期依赖的模型。
- [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://aclanthology.org/D14-1179/) — Kyunghyun Cho, Bart van Merriënboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio (2014)
  Journal: Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP); Publisher: Association for Computational Linguistics; Pages: 1724–1734; DOI: [10.3115/v1/D14-1179](https://doi.org/10.3115/v1/D14-1179)
  介绍了门控循环单元 (GRU)，一种替代 LSTM 的模型，以较少参数实现相似性能。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本关于深度学习基础的综合性教科书，内容涵盖优化、正则化以及神经网络的训练方法。
- [CS231n Convolutional Neural Networks for Visual Recognition: Neural Network Training](http://cs231n.github.io/neural-networks-3/) — Stanford University (2023)
  Publisher: Stanford University
  提供了深度神经网络训练的实用指导，涵盖初始化、优化以及适用于各种模型架构的正则化技术。
