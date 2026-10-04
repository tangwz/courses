---
course: "autoencoders-representation-learning"
chapter: "foundations-representation-learning"
lesson: "mathematical-preliminaries-refresher"
sourceId: 2804
sourceUrl: "https://apxml.com/zh/courses/autoencoders-representation-learning/chapter-1-foundations-representation-learning/mathematical-preliminaries-refresher"
title: "数学基础知识回顾"
description: "回顾用于自编码器的必要概率论、信息论和优化原理。"
order: 6
plots: ["plots/2804-0.json"]
sourceHash: "4ecedb45ce31335445882bba62effa1b33a9672412d26ef0b1c0f02c84935ae0"
sourceCorrections: []
---

随着我们构建自编码器（它们是经由优化训练的神经网络 (neural network)结构），扎实掌握某些数学原理变得不可或缺。这些工具构成了我们用来描述自编码器如何学习表征、衡量重建质量以及管理网络内部信息流动的语言。本节回顾了概率论、信息论和优化方面的重要思想，并根据它们在后续章节中的应用进行专门介绍。

### 概率论基本内容

自编码器，特别是变分自编码器（VAEs），通常在概率框架下运作。理解基本概率有助于解释它们的行为和设计选择。

- **随机变量与分布：** 数据点通常可以视为随机变量的具体表现。例如，输入图像 $x$ 可以被认为是高维数据分布 $P_{data}(x)$ 中的一个样本。自编码器学习转换这些变量的函数。我们经常遇到由概率密度函数 (PDF) 描述的连续随机变量，例如高斯（正态）分布，其特点是均值 $\mu$ 和方差 $\sigma^2$。它的 PDF 由以下公式给出：

  
  $$
  p(x; \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
  $$
  

  高斯分布对 VAEs 来说是根本性的，它们常作为隐变量的先验分布，以及编码器学习到的近似后验分布的形式。离散随机变量，由概率质量函数 (PMF) 描述，也具有相关性，尤其是在处理二元数据（伯努利分布）或分类输出时。
- **期望：** 随机变量 $X$ 的期望值，或均值 $E[X]$，表示其按概率加权的平均值。对于具有 PMF $P(x)$ 的离散变量 $X$，其期望为 $E[X] = \sum_x x P(x)$。对于具有 PDF $p(x)$ 的连续变量，其期望为 $E[X] = \int x p(x) dx$。期望被广泛使用，例如在定义均方误差等损失函数 (loss function)时，或在 VAEs 内部的采样过程中。
- **贝叶斯定理：** 这个定理描述了在已知与事件相关的条件下的事件概率。它表明：

  
  $$
  P(A|B) = \frac{P(B|A)P(A)}{P(B)}
  $$
  

  在 VAEs 等概率模型中，我们通常关心在给定观测数据 $x$ 的情况下推断隐变量 $z$，即计算后验概率 $P(z|x)$。贝叶斯定理为此提供了理论依据，将其与似然 $P(x|z)$（在给定隐变量下数据出现的可能性，通常由解码器建模）、先验 $P(z)$（在看到数据之前我们对隐变量的信念）以及证据 $P(x)$（数据的整体概率）联系起来。计算 $P(x)$ 通常是难以处理的，这导致了 VAEs 中所用的变分推断技术。

### 信息论思想

信息论提供了量化 (quantization)信息和不确定性的工具，这些工具对于表征学习和损失函数 (loss function)设计非常重要。

- **熵：** 香农熵 $H(X)$ 衡量随机变量可能结果中固有的“信息量”、“意外程度”或“不确定性”的平均水平。对于具有 PMF $P(x)$ 的离散随机变量 $X$，其定义为：

  
  $$
  H(X) = -\sum_{x} P(x) \log_b P(x)
  $$
  

  对数的底数 $b$ 决定了单位（当 $b=2$ 时为比特，当 $b=e$ 时为纳特）。熵值越高意味着不确定性越大。在表征学习中，我们有时旨在创建在保持相关信息的同时最小化熵的表征。
- **交叉熵：** 熵衡量单个分布的不确定性，而交叉熵 $H(P, Q)$ 衡量使用为分布 $Q$ 优化的编码方案时，识别从分布 $P$ 中抽取的事件所需的平均比特数/纳特数。

  
  $$
  H(P, Q) = -\sum_{x} P(x) \log Q(x) \quad \text{(离散)}
  $$
  
  
  $$
  H(P, Q) = -\int P(x) \log Q(x) dx \quad \text{(连续)}
  $$
  

  在机器学习 (machine learning)中，$P$ 通常是真实的潜在分布（或训练数据的经验分布），而 $Q$ 是我们模型预测的分布。最小化模型输出分布与目标分布之间的交叉熵是训练分类模型和自编码器的标准方法，尤其当输出被概率性解释时（例如，用于像素值在 0 到 1 之间的图像重建的二元交叉熵）。
- **KL 散度（Kullback-Leibler 散度）：** KL 散度衡量一个概率分布 $P$ 与参考概率分布 $Q$ 之间的差异。它定义为概率 $P$ 和 $Q$ 之间对数差异的期望值，其中期望是根据 $P$ 计算的。

  
  $$
  D_{KL}(P || Q) = \sum_{x} P(x) \log \frac{P(x)}{Q(x)} \quad \text{(离散)}
  $$
  
  
  $$
  D_{KL}(P || Q) = \int P(x) \log \frac{P(x)}{Q(x)} dx \quad \text{(连续)}
  $$
  

  重要地是，KL 散度是非负的 ($D_{KL}(P || Q) \geq 0$)，并且当且仅当 $P$ 和 $Q$ 完全相同时才为零。它*不对称*，通常意味着 $D_{KL}(P || Q) \neq D_{KL}(Q || P)$。在 VAEs（第四章）中，KL 散度在损失函数中作为正则化 (regularization)项，扮演着重要角色。它通常衡量编码器为隐变量学习的分布 $Q(z|x)$ 与所选先验分布 $P(z)$（通常是标准高斯分布）之间的差异。最小化此项会促使编码后的分布与先验分布相似，从而得到更具结构性和规律性的隐空间。



![P (蓝色) 和 Q (橙色) 之间的 KL 散度](plots/2804-0.json)



> KL 散度 $D_{KL}(P || Q)$ 量化了从参考分布 $Q$ 到目标分布 $P$ 的差异。最小化它会促使 $Q$ 变得与 $P$ 相似。

### 优化基本原理

训练神经网络 (neural network)，包括自编码器，涉及寻找使目标函数（通常称为损失函数 (loss function)）最小化的参数 (parameter)（权重 (weight)和偏置 (bias)）。

- **目标函数（损失函数）：** 损失函数量化 (quantization)了模型输出与期望目标之间的差异。对于标准自编码器，主要目标是重建。常见的重建损失包括：

  - **均方误差 (MSE)：** 适用于连续数据（例如，归一化 (normalization)到 [-1, 1] 或 [0, 1] 范围的像素强度）。计算为输入 $x$ 和重建 $\hat{x}$ 之间的平均平方差：$L_{MSE} = \frac{1}{N} \sum_{i=1}^N (x_i - \hat{x}_i)^2$。
  - **二元交叉熵 (BCE)：** 适用于输入数据为二元或归一化到 [0, 1] 范围并被解释为概率的情况（例如，黑白图像或像素概率）。假设输出 $\hat{x}_i$ 代表每个输入分量 $x_i$ 的伯努利分布参数：$L_{BCE} = -\frac{1}{N} \sum_{i=1}^N [x_i \log(\hat{x}_i) + (1 - x_i) \log(1 - \hat{x}_i)]$。
    规范化自编码器（第三章）和 VAEs（第四章）在此重建损失中增加了额外项（如稀疏性惩罚或 KL 散度）。
- **梯度下降 (gradient descent)：** 这是深度学习 (deep learning)中主要使用的优化算法。它迭代地调整模型参数 $\theta$ 以最小化损失 $L(\theta)$。在每一步中，参数会沿着损失函数相对于参数的梯度的相反方向更新：

  
  $$
  \theta_{t+1} = \theta_t - \eta \nabla_{\theta} L(\theta_t)
  $$
  

  这里，$\eta$ 是学习率，一个控制步长的超参数 (hyperparameter)。$\nabla_{\theta} L(\theta_t)$ 是损失函数相对于 $\theta$ 中每个参数的梯度（偏导数向量 (vector)）。
- **随机梯度下降 (SGD) 和小批量：** 在整个数据集上计算梯度可能会带来庞大的计算开销。SGD 使用单个数据点，或者更常见地，使用称为小批量的一个小数据子集来近似梯度。这使得更新更快，并引入了有助于跳出较差局部最小值的噪声。大多数现代训练使用小批量梯度下降。
- **高级优化器：** 虽然基本的 SGD 有效，但更复杂的算法通常收敛更快，并且更稳定可靠。Adam、RMSprop 和 Adagrad 等优化器根据梯度历史为每个参数调整学习率。它们是训练深度网络（包括自编码器）的常规选择。我们将在实际实现中使用这些。
- **反向传播 (backpropagation)：** 这个算法是神经网络中高效计算梯度 $\nabla_{\theta} L$ 的标准方法。它使用微积分的链式法则，将误差梯度从输出层向后传播通过网络层，计算每个参数的梯度。TensorFlow 和 PyTorch 等深度学习框架自动化了这一过程。

对这些数学支柱的扎实理解——概率论用于建模不确定性和结构，信息论用于量化信息和定义散度，以及优化用于实现学习——对于我们后续设计、实现和分析各种自编码器架构来说是不可或缺的。它们为编码、解码、隐空间和表征学习等思想奠定了根基。

## 参考资料

- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本全面的教材，涵盖了理解包括自编码器在内的深度学习所需的概率论、信息论和优化数学概念。其中第3、4、5章尤其相关。
- [Pattern Recognition and Machine Learning](https://link.springer.com/book/10.1007/978-0-387-45528-0) — Christopher M. Bishop (2006)
  Publisher: Springer; DOI: [10.1007/978-0-387-45528-0](https://doi.org/10.1007/978-0-387-45528-0)
  一本经典教材，提供了机器学习的概率视角，包含与变分自编码器等生成模型相关的概率论和信息论的详细章节。
- [Machine Learning Course Notes (CS229)](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEXoiQS3mqAtn9Pl6IZRCwPgH-pTXm0aztPKhgEZJkS0Oad2T2TAm71fLCHmaJ0f_63xULa1hJR6qRjZDQcYDPyjcsxOaAQ0Er_Enwf5YeTUYIJxItKZJoxdKx-DKWIJghwCzrz) — Andrew Ng, Tengyu Ma (2023)
  Publisher: Stanford University
  提供机器学习数学基础的详细课程笔记，包括概率论、优化算法和相关的统计概念。
