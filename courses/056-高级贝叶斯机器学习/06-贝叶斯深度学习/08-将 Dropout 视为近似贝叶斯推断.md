# 将 Dropout 视为近似贝叶斯推断

来源：[原文](https://apxml.com/zh/courses/bayesian-machine-learning/chapter-6-bayesian-deep-learning/dropout-bayesian-inference)

[返回章节目录](README.md) · [返回课程目录](../README.md)

Dropout 是常规深度学习 (deep learning)中广泛使用的正则化 (regularization)方法，旨在通过在每次训练更新时随机将部分神经元激活值设为零来防止过拟合 (overfitting)。它最初作为一种启发式方法提出，在实践中证明非常有效。然而，Dropout 与贝叶斯推断之间存在一种引人入胜的关联，这提供了对其机制更具原则性的理解，并促成从常规神经网络 (neural network)获取不确定性估计，且只需很少的改动。

这一见解，主要由 Gal 和 Ghahramani (2016) 提出，表明了在特定条件下，在*每个*权重 (weight)层之前应用 Dropout 训练神经网络，在数学上等同于对一个特定的深度高斯过程模型执行近似贝叶斯推断。更普遍地说，不仅在训练期间，而且在*测试时*也执行 Dropout，可以被解释为深度神经网络的一种贝叶斯近似形式，通常被称为 **MC Dropout (蒙特卡洛 Dropout)**。

### 理论联系

我们考虑一个具有 $L$ 层、权重 (weight) $W = \{W_1, ..., W_L\}$ 和偏置 (bias) $b = \{b_1, ..., b_L\}$ 的神经网络 (neural network)。在常规网络中，我们通过最小化损失函数 (loss function)（例如交叉熵或均方误差）来寻求 $W$ 和 $b$ 的点估计，可能还会附带 L2 权重衰减等正则化 (regularization)。

Dropout 为层 $l$ 中的每个单元 $i$ 引入二元随机变量 $z_{i}^{(l)}$。在训练期间，每个 $z_{i}^{(l)}$ 都从概率为 $p_l$ 的伯努利分布中采样，即 $z_{i}^{(l)} \sim \text{Bernoulli}(1 - p_l)$，其中 $p_l$ 是层 $l$ 的 Dropout 概率。层 $l$ 的输出 $y^{(l)}$ 通过与 Dropout 掩码 $z^{(l)}$ 进行元素乘法计算，然后再应用激活函数 (activation function) $\sigma$：

$y^{(l)} = \sigma( (x^{(l)} \odot z^{(l)}) W_l + b_l )$

（注意：Dropout 的具体放置位置，即在激活函数之前或之后，可以有所不同，这会影响具体的贝叶斯解释。对于这种解释，将其应用于权重乘法*之前*的输入很常见）。

核心思想是，训练带有 Dropout 和 L2 正则化的网络，可以被视为近似最小化近似分布 $q(W, b)$ 与真实贝叶斯后验 $p(W, b | \mathcal{D})$ 之间的 Kullback-Leibler (KL) 散度，这适用于对权重具有特定先验的网络。Dropout 隐式施加的先验通常与高斯或其它重尾分布的尺度混合有关，这促使稀疏性。

近似变分分布 $q(W, b)$ 由 Dropout 机制本身定义。具体来说，权重是共享的，但在每次前向传播时，会随机屏蔽掉一组不同的单元（以及与它们连接的权重）。在训练期间随机选择子网络的过程，起到了隐式变分推断的作用。

### MC Dropout：获取预测与不确定性

Dropout 的常规做法是在测试时禁用它，并将权重 (weight)按 $(1-p)$ 进行缩放，以考虑所有单元现在都已激活的情况。这会产生一个单点预测。

MC Dropout 修改了这一过程：

1. **训练：** 像往常一样使用 Dropout（例如，在权重层之前应用）和通常的 L2 正则化 (regularization)训练神经网络 (neural network)。
2. **预测：** 在测试时，*保持 Dropout 激活状态*。对于给定的输入 $x^*$，通过网络执行 $T$ 次随机前向传播。每次传播使用不同的 Dropout 掩码（随机丢弃不同的单元），产生 $T$ 个不同的预测 $\{ \hat{y}_t^* \}_{t=1}^T$。
3. **预测均值：** 最终预测 $\hat{y}_{MC}^*$ 是这些随机预测的平均值：
   
   $$
   \hat{y}_{MC}^* = \frac{1}{T} \sum_{t=1}^T \hat{y}_t^*
   $$
   
4. **不确定性估计：** 这 $T$ 个预测的样本方差（或标准差）可作为模型对其 $x^*$ 预测不确定性的估计：
   
   $$
   \text{方差}(\hat{y}^*) \approx \frac{1}{T} \sum_{t=1}^T (\hat{y}_t^* - \hat{y}_{MC}^*)^2
   $$
   
   对于分类，不确定性可以使用预测熵或基于 $T$ 次传播的预测概率的变异比率等指标进行估计。

这种不确定性估计主要捕获*认知不确定性*——源自模型参数 (parameter)的不确定性。因为不同的 Dropout 掩码有效地从近似后验 $q(W, b)$ 中采样不同的模型，所以其输出的变化反映了对最佳权重配置的不确定性。如果模型输出表示分布的参数（例如，回归的均值和方差），它也可以隐式捕获一些*偶然不确定性*。

### 实现

下面是 MC Dropout 预测在 PyTorch 或 TensorFlow 等框架中如何实现的示例。主要步骤是确保 Dropout 层在推断期间保持激活状态。

```python
# MC Dropout 预测循环代码

def get_mc_predictions(model, X_input, num_samples):
    """ 执行 T 次随机前向传播以获取 MC 预测。 """
    
    # 确保模型处于评估模式，但 Dropout 处于激活状态
    # 在 PyTorch 中，手动将 Dropout 层设置为训练模式：
    # for module in model.modules():
    #     if module.__class__.__name__.startswith('Dropout'):
    #         module.train()
    # 在 TensorFlow/Keras 中，将 training=True 传递给 Dropout 层/模型调用
    
    all_outputs = []
    
    # 在推断期间禁用梯度计算以提高效率
    # PyTorch 上下文管理器的示例：
    # with torch.no_grad(): 
    for _ in range(num_samples):
        # 假设模型前向传播需要为 Dropout 启用训练模式
        # 在 Keras 中：output = model(X_input, training=True)
        # 在 PyTorch 中：参阅上面关于设置 Dropout 模块为 train() 的注释
        
        # 执行前向传播（具体调用取决于框架）
        current_output = model(X_input) # 确保内部 Dropout 处于激活状态
        all_outputs.append(current_output)
        
    # 堆叠输出（根据需要调整维度，例如使用 torch.stack 或 tf.stack）
    # 示例：stacked_outputs 的形状可能是 [num_samples, batch_size, output_dim]
    stacked_outputs = stack_operation(all_outputs) 
    
    # 沿着样本维度（axis=0）计算均值和方差
    prediction_mean = calculate_mean(stacked_outputs, axis=0)
    prediction_variance = calculate_variance(stacked_outputs, axis=0)
    
    return prediction_mean, prediction_variance

# --- 辅助函数（替换为实际框架函数） ---
import numpy as np 
# 假设 stack_operation, calculate_mean, calculate_variance 是框架特定的
def stack_operation(outputs): return np.stack(outputs)
def calculate_mean(data, axis): return np.mean(data, axis=axis)
def calculate_variance(data, axis): return np.var(data, axis=axis)

# --- 使用示例 ---
# trained_model = load_my_model() # 加载您使用 Dropout 训练的模型
# X_new = load_new_data()
# num_mc_samples = 100 
# mean_preds, uncertainty_variance = get_mc_predictions(trained_model, X_new, num_mc_samples)

# print("平均预测：", mean_preds)
# print("预测方差（不确定性）：", uncertainty_variance)
```

> 呈现蒙特卡洛 Dropout 预测过程的 Python 代码。请注意，在推断时的多次前向传播中，确保 Dropout 层处于激活状态是重要步骤。具体机制取决于所使用的深度学习 (deep learning)框架。

### 优点与局限

**优点：**

- **简洁：** MC Dropout 的实现非常简单。通常，它只需要在现有代码库中，于预测时保持 Dropout 启用即可。
- **可扩展性：** 它使用常规深度学习 (deep learning)训练流程，并在测试时只增加极小的计算开销（与样本数量 $T$ 成比例）。
- **无需架构改动：** 它适用于常规网络架构，不需要显式的贝叶斯层或在测试时修改优化过程。
- **不确定性估计：** 提供了一种从深度模型获取不确定性估计的实用方式，这对可靠性和决策制定很有价值。

**局限：**

- **近似质量：** 贝叶斯近似的质量取决于网络架构、激活函数 (activation function)、Dropout 概率以及基本假设（如隐式先验）。它可能不如 HMC 或复杂变分推断等更显式的贝叶斯方法精确。
- **$T$ 的选择：** 蒙特卡洛样本数量 $T$ 影响均值和方差估计的稳定性。更高的 $T$ 值能提供更好的估计，但会增加计算时间。
- **隐式先验：** Dropout 施加的权重 (weight)先验分布是固定且隐式的，与显式定义先验的方法相比，其建模灵活性较低。
- **理论细节：** 精确的理论证明可能不明显，并且依赖于特定假设（例如，关于权重衰减对应高斯先验，与激活函数的互相影响）。

### 总结

Dropout 与近似贝叶斯推断之间的联系，为一种常见的正则化 (regularization)方法提供了有力的理论依据。MC Dropout 通过在测试时使用随机前向传播来估计预测不确定性，从而扩展了这一点。虽然它是一种近似方法，但其简洁性和可扩展性使其成为将不确定性感知引入深度学习 (deep learning)模型的一种极具吸引力且广泛使用的方法，而无需借助像 MCMC 或前面讨论的显式变分推断公式那样复杂的贝叶斯机制。它代表了传统深度学习与贝叶斯视角之间的一座实用桥梁。

## 参考资料

- [Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning](https://proceedings.mlr.press/v48/gal16.pdf) — Yarin Gal and Zoubin Ghahramani (2016)
  Journal: Proceedings of the 33rd International Conference on Machine Learning (ICML); Publisher: PMLR (Proceedings of Machine Learning Research); Volume: 48; Pages: 2020-2029; DOI: [10.5555/3045118.3045161](https://doi.org/10.5555/3045118.3045161)
  这篇基础论文阐述了 dropout 与近似贝叶斯推理的联系，并介绍了用于不确定性估计的蒙特卡洛 Dropout。
- [Dropout: A Simple Way to Prevent Overfitting Neural Networks](https://www.jmlr.org/papers/v15/srivastava14a.html) — Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, Ruslan Salakhutdinov (2014)
  Journal: JMLR; Publisher: JMLR; Volume: 15; Pages: 1929-1958; DOI: [10.5555/2627435.2670313](https://doi.org/10.5555/2627435.2670313)
  这篇原始论文介绍了 dropout 作为神经网络的一种正则化技术，对于理解其最初的设计至关重要。
- [Probabilistic Machine Learning: Advanced Topics](https://probml.github.io/pml-book/book2.html) — Kevin Patrick Murphy (2023)
  Publisher: MIT Press
  一本关于高级概率机器学习的现代教科书，其中包含贝叶斯深度学习和蒙特卡洛 Dropout 的章节。

---

[上一节](07-%E5%8F%98%E5%88%86%E8%87%AA%E7%BC%96%E7%A0%81%E5%99%A8%20%28VAEs%29%20%E4%BD%9C%E4%B8%BA%E6%A6%82%E7%8E%87%E6%A8%A1%E5%9E%8B.md) · [下一节](09-BNN%E7%9A%84%E5%AE%9E%E9%99%85%E8%AE%AD%E7%BB%83%E4%B8%8E%E8%AF%84%E4%BC%B0.md)
