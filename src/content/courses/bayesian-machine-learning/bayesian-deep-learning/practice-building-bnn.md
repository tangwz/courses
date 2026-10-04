---
course: "bayesian-machine-learning"
chapter: "bayesian-deep-learning"
lesson: "practice-building-bnn"
sourceId: 3632
sourceUrl: "https://apxml.com/zh/courses/bayesian-machine-learning/chapter-6-bayesian-deep-learning/practice-building-bnn"
title: "动手实践：构建贝叶斯神经网络"
description: "实现并训练贝叶斯神经网络，使用变分推断（VI）或蒙特卡洛Dropout（MC Dropout）完成回归或分类任务。"
order: 10
plots: ["plots/3632-0.json", "plots/3632-1.json"]
sourceHash: "123976bbf0f15758c58eb7b01331354e3839e891a710557a988dc121969a2f09"
sourceCorrections: []
---

一个简单的贝叶斯神经网络 (neural network)（BNN）使用变分推断（VI）方法进行构建、训练和评估。重点是一个回归任务，它允许直观地展现模型的预测结果及其相关的不确定性。TensorFlow Probability（TFP），一个将概率推断和统计分析与TensorFlow结合起来的库，用于此目的。

你应该安装了TensorFlow和TensorFlow Probability。如果没有，通常可以用pip安装它们：

```bash
pip install tensorflow tensorflow-probability matplotlib numpy
```

### 环境和数据准备

首先，我们导入所需的库，并生成一些合成数据用于我们的回归问题。我们将创建这样的数据：输入 $x$ 和输出 $y$ 之间的关系是非线性的，并加入了一些噪声。这种噪声代表了*偶然不确定性*。

```python
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# 为了结果的可复现性
np.random.seed(42)
tf.random.set_seed(42)

tfd = tfp.distributions
tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers

# 生成合成数据
def generate_data(n_samples=100, noise_std=0.1):
    X = np.linspace(-3, 3, n_samples).astype(np.float32).reshape(-1, 1)
    # 带有噪声的非线性函数
    y = X * np.sin(X * 2) + np.random.normal(0, noise_std, size=(n_samples, 1)).astype(np.float32)
    return X, y

X_train, y_train = generate_data(n_samples=150, noise_std=0.2)
X_test = np.linspace(-4, 4, 200).astype(np.float32).reshape(-1, 1)

# 可视化训练数据
fig = go.Figure()
fig.add_trace(go.Scatter(x=X_train.flatten(), y=y_train.flatten(), mode='markers', name='训练数据', marker=dict(color='#1f77b4', size=6)))
fig.update_layout(
    title='合成回归数据',
    xaxis_title='输入 (x)',
    yaxis_title='输出 (y)',
    template='plotly_white',
    legend_title_text='数据'
)
# fig.show() # 在Python环境中运行此行以显示图表
```



![合成回归数据](plots/3632-0.json)



> 训练数据遵循模式 $y \approx x \sin(2x)$ 并添加了高斯噪声。

### 定义贝叶斯神经网络 (neural network)

现在，我们将使用Keras函数式API和TFP层来定义我们的贝叶斯神经网络。具体来说，我们使用`tfp.layers.DenseVariational`。该层表示一个全连接神经网络层，其权重 (weight)和偏差是分布（我们的近似后验 $q(w)$），而不是点估计。

在训练期间，该层会将一个KL散度项添加到模型的损失函数 (loss function)中。该项衡量了学到的近似后验 $q(w)$ 与先验 $p(w)$ 之间的差异。该层会自动处理前向传播所需的采样，并将该KL项的计算作为VI目标的一部分（ELBO最大化，或等效地，负ELBO最小化）。

我们需要指定：

1. **先验分布：** 表示我们对权重信念的分布 $p(w)$，*在*看到数据*之前*。一个常见的选择是，以零为中心的各向同性高斯（正态）分布。
2. **后验近似：** 用于近似真实后验 $p(w|\mathcal{D})$ 的分布族 $q(w)$。一个常见选择是因子化的（均值场）高斯分布。
3. **KL散度计算函数：** 如何计算 $KL[q(w) || p(w)]$。TFP为此提供了工具。

```python
# 定义权重和偏差的先验分布
def prior_fn(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    prior_model = tfk.Sequential([
        tfpl.VariableLayer(tfpl.IndependentNormal.params_size(n), dtype=dtype),
        tfpl.IndependentNormal(n, convert_to_tensor_fn=tfd.Distribution.sample)
    ])
    return prior_model

# 定义后验近似策略（均值场高斯）
def posterior_fn(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    posterior_model = tfk.Sequential([
        tfpl.VariableLayer(tfpl.IndependentNormal.params_size(n), dtype=dtype),
        tfpl.IndependentNormal(n, convert_to_tensor_fn=tfd.Distribution.sample)
    ])
    return posterior_model

# 构建贝叶斯神经网络模型
def create_bnn_model(train_size):
    inputs = tfkl.Input(shape=(1,))
    hidden = tfpl.DenseVariational(
        units=32,
        make_prior_fn=prior_fn,
        make_posterior_fn=posterior_fn,
        kl_weight=1/train_size, # 按数据集大小缩放KL散度
        activation='relu'
    )(inputs)
    hidden = tfpl.DenseVariational(
        units=16,
        make_prior_fn=prior_fn,
        make_posterior_fn=posterior_fn,
        kl_weight=1/train_size,
        activation='relu'
    )(hidden)
    # 输出层：预测正态分布的均值
    # 我们将输出 y 模型化为 y ~ Normal(loc=f(x), scale=sigma)
    # 这里，f(x) 是 DenseVariational 层的输出
    # 为了简单起见，我们使用固定的标准差 (sigma)，
    # 有效地使用均方误差作为负对数似然。
    # 或者，另一个输出头可以预测sigma（偶然不确定性）。
    output_mean = tfpl.DenseVariational(
        units=1, # 预测均值参数
        make_prior_fn=prior_fn,
        make_posterior_fn=posterior_fn,
        kl_weight=1/train_size
        # 回归输出均值不需要激活函数
    )(hidden)

    # 为了简单起见，我们使用MSE损失，这对应于一个固定的高斯似然标准差。
    # 一个更完整的贝叶斯神经网络也可能预测标准差（尺度参数）。
    # 例子：output_scale = tfpl.DenseVariational(...) -> tf.exp(output_scale_raw)
    # 然后使用 tfp.layers.IndependentNormal(1) 作为最终层。

    model = tfk.Model(inputs=inputs, outputs=output_mean)
    return model

bnn_model = create_bnn_model(train_size=len(X_train))
bnn_model.summary()
```

我们将KL散度项乘以 `1 / train_size`。这是贝叶斯神经网络中变分推断的常见做法，用于平衡目标函数中的数据拟合（似然）项和正则化 (regularization)（KL散度）项。

### 定义损失函数 (loss function)和训练

对于变分推断，目标是最大化证据下界（ELBO），这等效于最小化负ELBO。

负ELBO可以表示为：


$$
-\text{ELBO} = -\mathbb{E}_{q(w)}[\log p(\mathcal{D}|w)] + KL[q(w) || p(w)]
$$


第一项是给定从近似后验中采样的参数 (parameter)时，数据的预期负对数似然。第二项是近似后验与先验之间的KL散度。

当Keras使用`DenseVariational`时，KL散度项会自动添加到模型的损失函数中。我们只需指定负对数似然项作为我们的主要损失函数。对于假定高斯噪声（恒定方差）的回归任务，负对数似然与均方误差（MSE）成正比。

```python
# 定义负对数似然损失函数（高斯似然的均方误差）
def nll_loss(y_true, y_pred_distribution):
    # 对于 DenseVariational，y_pred_distribution 在这里只是预测均值。
    # 一个更完整的模型会输出一个 tfd.Distribution。
    # return -y_pred_distribution.log_prob(y_true) # 如果输出层是 tfp.layers.IndependentNormal
    return tf.reduce_mean(tf.square(y_true - y_pred_distribution))

# 编译模型
optimizer = tfk.optimizers.Adam(learning_rate=0.01)
bnn_model.compile(optimizer=optimizer, loss=nll_loss) # Keras 会自动添加 KL 散度

# 训练模型
print("开始训练...")
history = bnn_model.fit(X_train, y_train, epochs=500, batch_size=32, verbose=0)
print("训练完成。")

# 你可以绘制损失曲线（总损失 = 负对数似然 + KL散度）
# plt.plot(history.history['loss'])
# plt.title('模型训练期间的损失')
# plt.xlabel('周期')
# plt.ylabel('总损失 (-ELBO)')
# plt.show()
```

### 进行预测和不确定性可视化

贝叶斯神经网络 (neural network)的主要优势是它们量化 (quantization)不确定性的能力。使用变分推断，我们用 $q(w)$ 近似后验 $p(w|\mathcal{D})$。为了获得预测不确定性，我们通过网络执行多次前向传播，每次采样一组不同的权重 (weight) $w_i \sim q(w)$。输出中的变化反映了模型的*认知不确定性*（关于模型参数 (parameter)的不确定性）。

```python
# 通过多次采样进行预测
n_samples = 100
predictions_mc = np.stack([bnn_model(X_test).numpy() for _ in range(n_samples)], axis=0)

# 移除不必要的维度
predictions_mc = np.squeeze(predictions_mc) # Shape: (n_samples, n_test_points)

# 计算预测均值和标准差
pred_mean = np.mean(predictions_mc, axis=0)
pred_std = np.std(predictions_mc, axis=0)

# 可视化结果：均值预测和不确定性边界
fig = go.Figure()

# 不确定性边界（例如，+/- 2个标准差）
fig.add_trace(go.Scatter(
    x=np.concatenate([X_test.flatten(), X_test.flatten()[::-1]]),
    y=np.concatenate([pred_mean - 2 * pred_std, (pred_mean + 2 * pred_std)[::-1]]),
    fill='toself',
    fillcolor='rgba(250, 82, 82, 0.2)', # 淡红色 #fa5252
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip",
    showlegend=False,
    name='认知不确定性 (±2 std)'
))

# 均值预测
fig.add_trace(go.Scatter(
    x=X_test.flatten(), y=pred_mean,
    mode='lines', name='预测均值',
    line=dict(color='#f03e3e') # 红色 #f03e3e
))

# 原始训练数据
fig.add_trace(go.Scatter(
    x=X_train.flatten(), y=y_train.flatten(),
    mode='markers', name='训练数据',
    marker=dict(color='#1c7ed6', size=6) # 蓝色 #1c7ed6
))

fig.update_layout(
    title='带不确定性的贝叶斯神经网络回归',
    xaxis_title='输入 (x)',
    yaxis_title='输出 (y)',
    template='plotly_white',
    legend_title_text='组成部分'
)
# fig.show() # 在Python环境中运行此行以显示图表
```



![带不确定性的贝叶斯神经网络回归](plots/3632-1.json)



> 贝叶斯神经网络的预测均值（红线）反映了底层趋势，而阴影区域（距均值 $\pm 2$ 个标准差）代表了认知不确定性。请注意，在没有训练数据的区域（例如 $x < -3$ 或 $x > 3$）以及函数变化迅速的地方，不确定性会增加。

### 替代方法：蒙特卡洛Dropout

如前所述，蒙特卡洛（MC）Dropout提供了一种更简单的方法来近似现有标准神经网络 (neural network)中的贝叶斯推断。它包括：

1. 训练一个带有Dropout层的标准神经网络。
2. 在预测时，保持Dropout活跃并对相同输入执行多次前向传播。
3. 计算这些多次预测的均值和方差/标准差，以估计预测均值和不确定性。

虽然在计算上更便宜且在标准框架中更容易实现，但蒙特卡洛Dropout是特定类型贝叶斯神经网络的一种近似（与高斯过程相关）。我们实现的变分推断方法通常被认为是一种更合理的方法来构建具有明确先验和后验的贝叶斯神经网络。

### 总结与后续步骤

在本实践部分，我们使用TensorFlow Probability的`DenseVariational`层构建了一个贝叶斯神经网络 (neural network)。我们使用变分推断对其进行训练，目标函数平衡了数据拟合（通过负对数似然/均方误差）和遵循先验信念（通过KL散度）。通过从学到的权重 (weight)近似后验分布中采样，我们生成了预测结果以及可量化 (quantization)的认知不确定性估计。

这个例子为应用贝叶斯神经网络提供了起始点。你 E可以通过以下方式扩展此内容：

- 通过让网络预测输出分布的方差（尺度参数 (parameter)），明确地建模偶然不确定性。
- 尝试不同的网络架构、先验或变分族。
- 将贝叶斯神经网络应用于分类任务（需要不同的似然函数，例如Categorical）。
- 研究MCMC方法，例如SGHMC，以获得可能更准确（但通常更慢）的后验采样。
- 比较贝叶斯神经网络的性能和校准，并与标准神经网络和蒙特卡洛Dropout进行对比。

构建贝叶斯神经网络提供了一个强大的框架，用于创建深度学习 (deep learning)模型，这些模型不仅能预测，还能理解自身的置信度。

## 参考资料

- [Weight Uncertainty in Neural Networks](https://arxiv.org/abs/1505.05424) — Charles Blundell, Julien Cornebise, Koray Kavukcuoglu, Daan Wierstra (2015)
  Journal: Proceedings of the 32nd International Conference on Machine Learning (ICML); DOI: [10.48550/arXiv.1505.05424](https://doi.org/10.48550/arXiv.1505.05424)
  介绍了“通过反向传播实现贝叶斯推断”，这是一种训练贝叶斯神经网络的实用变分推断方法，与`tfp.layers.DenseVariational`的实现直接相关。
- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — Diederik P Kingma, Max Welling (2014)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1312.6114](https://doi.org/10.48550/arXiv.1312.6114)
  介绍了变分自编码器（VAE）并提出了重参数化技巧，这是在变分推断中估计证据下界（ELBO）梯度的重要技术。
- [Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning](https://arxiv.org/abs/1506.02142) — Yarin Gal, Zoubin Ghahramani (2016)
  Journal: Proceedings of the 33rd International Conference on Machine Learning (ICML); DOI: [10.48550/arXiv.1506.02142](https://doi.org/10.48550/arXiv.1506.02142)
  建立了神经网络中的Dropout与贝叶斯推断之间的理论联系，提供了一种通过蒙特卡洛Dropout量化模型不确定性的方法。
