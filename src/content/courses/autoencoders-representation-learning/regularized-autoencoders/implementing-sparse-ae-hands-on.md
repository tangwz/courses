---
course: "autoencoders-representation-learning"
chapter: "regularized-autoencoders"
lesson: "implementing-sparse-ae-hands-on"
sourceId: 2841
sourceUrl: "https://apxml.com/zh/courses/autoencoders-representation-learning/chapter-3-regularized-autoencoders/implementing-sparse-ae-hands-on"
title: "实现稀疏自编码器：实践操作"
description: "编码并训练稀疏自编码器，在瓶颈层强制实现稀疏性。"
order: 7
plots: ["plots/2841-0.json"]
sourceHash: "92d924a8b07a1a3ab39faf507f78878cbc15b89c6940a482ae3563e3d5876d20"
sourceCorrections: []
---

稀疏自编码器的目标是学习压缩表示，通过鼓励隐藏（瓶颈）层激活的稀疏性。这使得网络对于任何给定输入都只使用一小部分隐藏单元，可能有助于形成更专用的特征检测器。稀疏自编码器可以通过使用两种常见技术：L1正则化 (regularization)和KL散度惩罚来实现。

这些例子我们将使用带有Keras API的TensorFlow。请确保您已安装TensorFlow（`pip install tensorflow`）。我们将使用Fashion-MNIST数据集，它是MNIST的一个稍微更具挑战性的替代选择。

### 环境与数据设置

首先，让我们导入所需的库并加载数据集。

```python
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, losses, backend as K
import numpy as np
import matplotlib.pyplot as plt

# 加载Fashion-MNIST数据集
(x_train, _), (x_test, _) = tf.keras.datasets.fashion_mnist.load_data()

# 归一化并重塑数据（展平图像）
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

print(f"Training data shape: {x_train.shape}")
print(f"Test data shape: {x_test.shape}")

# 定义输入形状和编码维度
input_dim = x_train.shape[1] # Fashion-MNIST为784
encoding_dim = 64 # 瓶颈层的大小
```

### 带有L1激活正则化 (regularization)的稀疏自编码器

鼓励稀疏性最直接的方法是在损失函数 (loss function)中添加一个惩罚项，该惩罚项与瓶颈层激活的L1范数（绝对值之和）成比例。Keras提供了一种便捷的方法，通过在瓶颈层使用`activity_regularizer`来完成此操作。

总损失变为：


$$
\text{损失} = \text{重构损失} + \lambda \sum_{i} |h_i|
$$


其中 $h_i$ 是瓶颈层单元的激活，$\lambda$ 是正则化强度参数 (parameter)。

让我们定义模型架构。

```python
# L1正则化强度
l1_lambda = 1e-5 # 这是一个需要调整的超参数

# 定义输入层
input_img = layers.Input(shape=(input_dim,))

# 定义带有瓶颈层L1激活正则化的编码器
encoded = layers.Dense(128, activation='relu')(input_img)
encoded = layers.Dense(encoding_dim, activation='relu',
                       activity_regularizer=regularizers.l1(l1_lambda))(encoded) # 在此处应用L1

# 定义解码器
decoded = layers.Dense(128, activation='relu')(encoded)
decoded = layers.Dense(input_dim, activation='sigmoid')(decoded) # 用于像素值 [0, 1] 的Sigmoid激活函数

# 构建自编码器模型
autoencoder_l1 = models.Model(input_img, decoded)

# 编译模型
autoencoder_l1.compile(optimizer='adam', loss='binary_crossentropy') # BCE适用于 [0,1] 像素值

autoencoder_l1.summary()
```

现在，训练模型。我们不需要标签（`y_train`，`y_test`），因为自编码器是无监督的。

```python
# 训练参数
epochs = 30
batch_size = 256

# 训练自编码器
history_l1 = autoencoder_l1.fit(x_train, x_train, # 输入和目标相同
                                epochs=epochs,
                                batch_size=batch_size,
                                shuffle=True,
                                validation_data=(x_test, x_test),
                                verbose=1) # 将verbose设置为2以减少每轮输出

print("训练完成。")
```

训练结束后，您可以检查重构结果，更重要的是，为了稀疏性，检查瓶颈层中的激活。

```python
# 单独构建编码器模型以获取瓶颈层激活
encoder_l1 = models.Model(input_img, encoded)

# 获取测试数据的瓶颈层激活
encoded_imgs_l1 = encoder_l1.predict(x_test)

# 计算并打印平均激活值
print(f"L1瓶颈层中的平均激活：{np.mean(encoded_imgs_l1):.4f}")

# 可视化每个神经元的平均激活
avg_activations_l1 = np.mean(encoded_imgs_l1, axis=0)

plt.figure(figsize=(10, 4))
plt.bar(range(encoding_dim), avg_activations_l1)
plt.title('每个神经元的平均激活（L1正则化）')
plt.xlabel('神经元索引')
plt.ylabel('平均激活')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
```

您应该会观察到许多神经元的平均激活非常低，这表明L1惩罚成功地引入了稀疏性。`l1_lambda`的值影响稀疏程度；值越高会导致表示越稀疏，但如果设置过高可能会损害重构质量。

### 带有KL散度正则化 (regularization)的稀疏自编码器

另一种方法是通过在损失函数 (loss function)中添加一个KL散度项来强制稀疏性。该项衡量了隐藏单元的期望平均激活（一个小的 $\rho$ 值，例如 0.05）与训练批次中观察到的实际平均激活（神经元 $j$ 的 $\hat{\rho}_j$）之间的差异。

单个神经元 $j$ 的KL散度惩罚为：


$$
\text{KL}(\rho || \hat{\rho}_j) = \rho \log \frac{\rho}{\hat{\rho}_j} + (1-\rho) \log \frac{1-\rho}{1-\hat{\rho}_j}
$$


添加到损失中的总稀疏性惩罚是所有瓶颈神经元上的和，由参数 (parameter) $\beta$ 加权：


$$
\text{损失} = \text{重构损失} + \beta \sum_{j=1}^{\text{编码维度}} \text{KL}(\rho || \hat{\rho}_j)
$$


实现这一点通常需要自定义层或修改训练循环以计算 $\hat{\rho}_j$ 并添加KL项。以下是在Keras中定义自定义正则化器的方法。

```python
# 稀疏性参数
rho = 0.05  # 目标稀疏性
beta = 3    # 稀疏性权重

# 自定义KL散度正则化器
class KLDivergenceRegularizer(regularizers.Regularizer):
    def __init__(self, rho, beta):
        self.rho = rho
        self.beta = beta

    def __call__(self, activations):
        # 计算批次中的平均激活
        # K.mean 沿着轴0（批次维度）计算平均值
        rho_hat = K.mean(activations, axis=0)
        # 计算KL散度
        kl_divergence = self.rho * K.log(self.rho / rho_hat + K.epsilon()) + \
                        (1 - self.rho) * K.log((1 - self.rho) / (1 - rho_hat) + K.epsilon())
        # 返回瓶颈神经元上的缩放和
        return self.beta * K.sum(kl_divergence)

    def get_config(self):
        return {'rho': float(self.rho), 'beta': float(self.beta)}

# 使用KL正则化器定义模型架构
input_img_kl = layers.Input(shape=(input_dim,))

encoded_kl = layers.Dense(128, activation='relu')(input_img_kl)
# 将KL正则化器应用于瓶颈层激活
encoded_kl = layers.Dense(encoding_dim, activation='sigmoid', # Sigmoid常用于KL的 [0,1] 范围
                          activity_regularizer=KLDivergenceRegularizer(rho, beta))(encoded_kl)

decoded_kl = layers.Dense(128, activation='relu')(encoded_kl)
decoded_kl = layers.Dense(input_dim, activation='sigmoid')(decoded_kl)

autoencoder_kl = models.Model(input_img_kl, decoded_kl)

# 编译模型（确保损失函数合适，例如BCE）
autoencoder_kl.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder_kl.summary()

print("\n训练KL散度稀疏自编码器...")
history_kl = autoencoder_kl.fit(x_train, x_train,
                                epochs=epochs,
                                batch_size=batch_size,
                                shuffle=True,
                                validation_data=(x_test, x_test),
                                verbose=1)

print("训练完成。")
```

请注意，在KL正则化的瓶颈层中使用了 `activation='sigmoid'`。这很常见，因为KL散度公式假设激活 $\hat{\rho}_j$ 在0到1之间，Sigmoid函数能够保证这一点。如果使用ReLU，激活可能超过1，可能导致KL公式中对数项出现问题。

现在，让我们评估KL散度实现的稀疏性。

```python
# 构建相应的编码器
encoder_kl = models.Model(input_img_kl, encoded_kl)

# 获取瓶颈层激活
encoded_imgs_kl = encoder_kl.predict(x_test)

# 计算并打印平均激活值
print(f"KL瓶颈层中的平均激活：{np.mean(encoded_imgs_kl):.4f}")

# 可视化每个神经元的平均激活
avg_activations_kl = np.mean(encoded_imgs_kl, axis=0)

plt.figure(figsize=(10, 4))
plt.bar(range(encoding_dim), avg_activations_kl)
plt.axhline(rho, color='r', linestyle='--', label=f'目标稀疏性 rho={rho}')
plt.title('每个神经元的平均激活（KL散度）')
plt.xlabel('神经元索引')
plt.ylabel('平均激活')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# 可视化测试集中所有瓶颈层激活的直方图
plt.figure(figsize=(8, 5))
plt.hist(encoded_imgs_kl.flatten(), bins=50, color='#4dabf7', alpha=0.8)
plt.title('KL瓶颈层激活直方图（测试集）')
plt.xlabel('激活值')
plt.ylabel('频率')
plt.yscale('log') # 使用对数刻度以更好地查看低激活频率
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
```



![平均神经元激活：L1对比KL](plots/2841-0.json)



> L1和KL散度稀疏自编码器在Fashion-MNIST测试集上每个神经元平均激活的比较。KL散度旨在达到特定的目标平均激活（例如0.05），而L1则鼓励激活趋近于零，没有固定目标。

KL散度方法试图将批次中每个隐藏单元的*平均*激活推向目标 $\rho$。直方图通常显示一个接近零的峰值，并且可能还有一个接近一的小峰值（如果使用Sigmoid激活），大多数激活值非常小。`beta`参数控制这种稀疏性约束相对于重构损失的强度。

### 结论

本次实践练习展示了如何在TensorFlow/Keras中使用L1和KL散度正则化 (regularization)来实现稀疏自编码器。这两种方法都有效地鼓励了瓶颈层的稀疏性，使得网络能够学习到比标准自编码器更压缩且可能更有意义的特征。L1和KL散度之间的选择，以及调整它们各自的超参数 (parameter) (hyperparameter)（$\lambda$ 或 $\rho$ 和 $\beta$），取决于特定的数据集和任务要求。有必要对这些参数进行实验，以在实现良好重构质量和强制所需稀疏性之间取得平衡。这些正则化模型通常提供更具鲁棒性并更适合下游任务的表示。

## 参考资料

- [UFLDL Tutorial: Sparse Autoencoders](http://ufldl.stanford.edu/wiki/index.php/Sparse_Autoencoder) — Andrew Ng (2011)
  Publisher: Stanford University
  提供稀疏自编码器的基础解释，包括KL散度正则化背后的理论。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本权威的教科书，涵盖自编码器、L1等正则化技术以及深度学习的基本数学原理。
- [Keras Regularizers API](https://keras.io/api/layers/regularizers/) — Keras Team and TensorFlow Authors (2023)
  Publisher: Keras
  官方文档，详细介绍了如何在Keras框架中应用L1、L2和自定义正则化器进行模型训练。
