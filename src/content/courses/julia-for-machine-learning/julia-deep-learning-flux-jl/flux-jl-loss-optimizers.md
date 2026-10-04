---
course: "julia-for-machine-learning"
chapter: "julia-deep-learning-flux-jl"
lesson: "flux-jl-loss-optimizers"
sourceId: 6903
sourceUrl: "https://apxml.com/zh/courses/julia-for-machine-learning/chapter-6-julia-deep-learning-flux-jl/flux-jl-loss-optimizers"
title: "定义损失函数与优化器"
description: "了解并应用 Flux.jl 中用于神经网络训练的各类损失函数和优化算法。"
order: 4
plots: []
sourceHash: "209bdb025cb484ed23cd4afba0170ec24f5c2b1ea930eecf1a9077ab26829273"
sourceCorrections: []
---

确定神经网络 (neural network)模型如何学习，需要两个主要组成部分：**损失函数 (loss function)**和**优化器**。损失函数衡量模型预测值与实际目标值的偏差程度。优化器调整模型参数 (parameter)（权重 (weight)和偏差）以使此误差最小化。在 Flux.jl 环境中定义和使用这些组件是重点。

## 了解损失函数 (loss function)

损失函数，也称为成本函数或目标函数，量化 (quantization)了模型预测输出 ($\hat{y}$) 与真实目标值 ($y$) 之间的差异。训练神经网络 (neural network)的目标是找到一组使此损失最小的参数 (parameter)。较小的损失值表明模型的预测更接近实际值。

损失函数的选择很大程度上取决于您要解决的问题类型：

- 对于**回归任务**，即预测连续值（例如，房价），均方误差是常用选项。
- 对于**分类任务**，即预测类别（例如，垃圾邮件或非垃圾邮件，数字识别），通常使用交叉熵损失函数。

Flux.jl 在其 `Flux.Losses` 模块中提供了一系列预定义的损失函数，使其易于将其纳入您的训练流程。

### Flux.jl 中常用的损失函数 (loss function)

让我们看一下 Flux 中一些最常用的损失函数以及如何使用它们。

#### 均方误差 (MSE)

均方误差衡量误差平方的平均值。因为它有平方项，所以它对异常值特别敏感。公式为：
$L_{MSE} = \frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2$
其中 $N$ 为样本数量，$y_i$ 为真实值，$\hat{y}_i$ 为预测值。

在 Flux.jl 中，您可以使用 `Flux.mse`：

```julia
using Flux

# 示例：
y_hat = [0.5, 1.8, 3.3] # 模型预测值
y_true = [0.6, 2.0, 3.0] # 真实值

loss = Flux.mse(y_hat, y_true)
println("MSE Loss: ", loss)
# 输出：均方误差：0.033333335f0 (近似值)
```

这适用于当模型最后一层直接输出连续值时，通常不带激活函数 (activation function)或带有线性激活。

#### 交叉熵损失

交叉熵损失是分类问题的标准方法。它衡量分类模型的性能，其输出是介于 0 和 1 之间的概率值。

**二元交叉熵**
对于二元分类（两个类别，例如 0 或 1），使用二元交叉熵 (BCE) 损失。如果 $y$ 是真实标签（0 或 1），$\hat{y}$ 是类别 1 的预测概率，则单个样本的 BCE 损失为：
$L_{BCE} = -[y \log(\hat{y}) + (1-y) \log(1-\hat{y})]$
Flux 提供了 `Flux.binarycrossentropy` 适用于概率输入（通常在 sigmoid 激活后），以及 `Flux.logitbinarycrossentropy` 适用于 logits 输入（sigmoid 激活前的原始得分）。使用 `logit` 版本通常在数值上更稳定。

```julia
using Flux

# logitbinarycrossentropy 示例（期望原始得分/logits）
logits = [0.5, -1.0, 2.0]  # 原始模型输出（sigmoid 前）
y_true_binary = [1.0, 0.0, 1.0] # 真实标签（0 或 1）

# 注意：Flux.logitbinarycrossentropy 期望 y_true 为 0 或 1。
loss_bce_logits = Flux.logitbinarycrossentropy(logits, y_true_binary)
println("Logit 二元交叉熵损失：", loss_bce_logits)

# binarycrossentropy 示例（期望概率）
probabilities = Flux.sigmoid.([0.5, -1.0, 2.0]) # sigmoid 后的输出
loss_bce_probs = Flux.binarycrossentropy(probabilities, y_true_binary)
println("二元交叉熵损失（带概率）：", loss_bce_probs)
```

**多类别交叉熵**
对于多类别分类（两个以上类别），使用多类别交叉熵。如果 $C$ 为类别数量，如果样本 $i$ 属于类别 $j$，则 $y_{ij}$ 为 1（否则为 0，通常为独热编码），$\hat{y}_{ij}$ 为样本 $i$ 属于类别 $j$ 的预测概率，$N$ 个样本的公式为：
$L_{CCE} = -\frac{1}{N}\sum_{i=1}^{N} \sum_{j=1}^{C} y_{ij} \log(\hat{y}_{ij})$
Flux 提供了 `Flux.crossentropy` 适用于概率输入（通常在 softmax 激活后），以及 `Flux.logitcrossentropy` 适用于 logits 输入（softmax 前的原始得分）。同样，出于数值稳定性考虑，通常更推荐使用 `logit` 版本。

```julia
using Flux
using Flux: onehotbatch, onecold # 用于独热编码

# logitcrossentropy 示例（期望多类别的原始得分/logits）
# 假设 3 个类别，2 个样本
logits_multiclass = Float32[
    0.1 0.8;  # 样本 1：类别 1 的得分，类别 2 的得分
    0.5 0.1;  # 样本 1：类别 2 的得分，类别 2 的得分
    0.4 0.1   # 样本 1：类别 3 的得分，类别 2 的得分
] # 2 个样本的预测，每个样本 3 个类别（列为样本）

# 真实标签（例如，样本 1 属于类别 2，样本 2 属于类别 1）
y_true_multiclass_indices = [2, 1]
# 转换为独热编码
y_true_onehot = Flux.onehotbatch(y_true_multiclass_indices, 1:3)

loss_cce_logits = Flux.logitcrossentropy(logits_multiclass, y_true_onehot)
println("Logit 多类别交叉熵损失：", loss_cce_logits)

# crossentropy 示例（期望 softmax 后的概率）
probabilities_multiclass = Flux.softmax(logits_multiclass, dims=1)
loss_cce_probs = Flux.crossentropy(probabilities_multiclass, y_true_onehot)
println("多类别交叉熵损失（带概率）：", loss_cce_probs)
```

务必确保您的目标数据 `y_true` 符合所选损失函数期望的格式（例如，原始标签，独热编码向量 (vector)）。

## 优化器：指导学习过程

一旦损失函数 (loss function)计算出误差，优化器的作用就是以减少此误差的方式更新模型的参数 (parameter)（权重 (weight) $W$ 和偏差 $b$）。优化器使用损失函数相对于参数的梯度。您将了解到这些梯度由 Zygote.jl 计算，它们指示损失函数最陡峭上升的方向。优化器沿相反方向（最陡峭下降）迈步以使损失最小化。

使用梯度下降 (gradient descent)法更新参数 $\theta$（可以是权重或偏差）的通用更新规则为：
$\theta_{new} = \theta_{old} - \eta \nabla L$
其中 $\eta$ (eta) 是学习率，一个控制步长的超参数 (hyperparameter)，$\nabla L$ 是损失 $L$ 相对于参数 $\theta$ 的梯度。

Flux.jl 在 `Flux.Optimise` 中提供了多种优化器。

> 该图示了模型输出、真实值、损失函数和优化器在生成参数更新方面的关系。由损失信号获得的梯度由优化器使用，以确定如何调整模型的权重。

### Flux.jl 中常用的优化器

#### 随机梯度下降 (gradient descent) (SGD)

SGD 是基础的优化算法。在它最简单的形式中，它使用固定学习率更新参数 (parameter)。

```julia
using Flux

# opt_state = Flux.setup(Flux.SGD(0.01), model) # 0.01 是学习率
# 对于模型 'm'
opt = Flux.SGD(0.01) # 学习率为 0.01
```

虽然简单，但 SGD 可能收敛速度慢，并且对学习率的选择敏感。像带动量的 SGD (`Flux.Momentum`) 这样的变体通常更有效。

#### Adam (自适应矩估计)

Adam 是一种流行且通常有效的优化器，它为每个参数单独调整学习率。它结合了动量和 RMSProp 的思想。它通常在默认超参数 (hyperparameter)设置下表现良好，使其成为一个良好的起始选择。

```julia
using Flux

# opt_state = Flux.setup(Flux.ADAM(0.001), model) # 0.001 是 Adam 常用的默认学习率
opt = Flux.ADAM(0.001) # 学习率为 0.001
# Adam 还有其他参数，如 β1, β2, ϵ，它们有默认值。
# opt = Flux.ADAM(0.001, (0.9, 0.999), 1e-8)
```

`ADAM` 的参数为 `(η, β::Tuple, ϵ)`。$\eta$ 是学习率。`β` 是一个元组 `(β1, β2)`，表示矩估计的衰减率。$\epsilon$ (`ϵ`) 是用于数值稳定性的小常数。

#### RMSProp

RMSProp (均方根传播) 也为每个参数调整学习率，通过将学习率除以平方梯度的指数衰减平均值来实现。它在循环神经网络 (neural network) (RNN)中表现良好。

```julia
using Flux

# opt_state = Flux.setup(Flux.RMSProp(0.001), model)
opt = Flux.RMSProp(0.001) # 学习率为 0.001
# RMSProp 也有一个用于衰减率的 γ 参数（默认 0.9）
# opt = Flux.RMSProp(0.001, 0.9)
```

Flux.jl 提供了多种优化器，包括 `AdaGrad`、`AdaDelta`、`AMSGrad` 等。您可以在 `Flux.Optimise` 模块文档中找到它们。

## 选择合适的组合

选择合适的损失函数 (loss function)通常很简单，由您的问题类型决定：

- **回归**：`Flux.mse` 是不错的默认选项。其他包括 `Flux.mae` (平均绝对误差)。
- **二元分类**：`Flux.logitbinarycrossentropy`（如果模型输出 logits）或 `Flux.binarycrossentropy`（如果输出通过 sigmoid 为概率）。
- **多类别分类**：`Flux.logitcrossentropy`（如果模型输出 logits）或 `Flux.crossentropy`（如果输出通过 softmax 为概率）。

选择优化器及其超参数 (parameter) (hyperparameter)（如学习率）通常需要一些实验。

- **Adam** 通常是首选，并且通常在默认设置或 `0.001` 这样的学习率下产生良好结果。
- 带动量的 **SGD** 也可以非常有效，但可能需要更多的学习率和动量参数调整。

学习率是一个特别敏感的超参数。学习率过小会导致收敛速度非常慢，而过大可能导致损失振荡或发散。学习率调度（随时间衰减学习率）等技术也可能是有益的，Flux 通过可以封装优化器的调度器函数来支持这些功能。

定义好损失函数和优化器后，您就拥有了主要组成部分，准备好指导您的模型*如何*从数据中学习。下一步是实现训练循环，它重复地将数据输入模型，计算损失，并使用优化器更新模型。

## 参考资料

- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本全面的深度学习教科书，涵盖了各种损失函数、SGD、Adam和RMSProp等优化算法，以及基于梯度学习的底层数学原理。
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) — Diederik P. Kingma and Jimmy Ba (2015)
  Journal: 3rd International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1412.6980](https://doi.org/10.48550/arXiv.1412.6980)
  介绍Adam优化器的原始论文，Adam优化器因其自适应学习率能力而在深度学习中被广泛使用。
- [Flux.jl Documentation: Losses](https://fluxml.ai/Flux.jl/stable/models/losses/) — Flux.jl Contributors (2025)
  Flux.jl中可用损失函数的官方文档，包括`mse`、`binarycrossentropy`和`crossentropy`，并附有使用示例。
- [Flux.jl Documentation: Optimizers](https://optimisers.fluxml.ai/stable/) — Flux.jl Contributors (2023)
  Flux.jl提供的优化器官方文档，详细介绍了它们在框架中的功能和用法，包括SGD、Adam和RMSProp。
- [Lecture 6a: RmsProp, Divide the gradient by a running average of its recent magnitude](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf) — Geoffrey Hinton (2012)
  Journal: Coursera Lecture Notes; Publisher: University of Toronto
  介绍RMSProp优化算法的原始讲义，这是一种自适应学习率方法。
