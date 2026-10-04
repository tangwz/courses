---
course: "autoencoders-representation-learning"
chapter: "variational-autoencoders-generative"
lesson: "implementing-vae-image-generation-practice"
sourceId: 2867
sourceUrl: "https://apxml.com/zh/courses/autoencoders-representation-learning/chapter-4-variational-autoencoders-generative/implementing-vae-image-generation-practice"
title: "实现图像生成变分自编码器：实践"
description: "构建并训练一个变分自编码器，以生成新颖的图像样本。"
order: 9
plots: ["plots/2867-0.json"]
sourceHash: "1832590fbe54b17a3d9560de312a724cb7f7b737cd4dd5234c705411e5125953"
sourceCorrections: []
---

变分自编码器（VAE）是生成模型，其特点是包含概率编码器/解码器结构、重参数 (parameter)化技巧和证据下界（ELBO）目标函数。此处将演示使用主流深度学习 (deep learning)框架构建和训练一个VAE以生成新图像。重点关注手写数字的MNIST数据集，通过实际应用展示VAE的生成能力。

我们假设您已配置好Python环境，并安装了TensorFlow和Keras，以及NumPy和Matplotlib等标准库。

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
```

## 定义变分自编码器组成部分

变分自编码器由三个主要部分构成：编码器网络、使用重参数 (parameter)化技巧的采样机制以及解码器网络。

### 编码器网络

编码器通常表示为$q_\phi(z|x)$，它接收输入图像$x$并将其映射到潜在空间中概率分布的参数 (parameter)。对于变分自编码器而言，这通常是具有对角协方差矩阵的多元高斯分布。因此，编码器输出两个向量 (vector)：该分布的均值$\mu$和对数方差$\log \sigma^2$。使用对数方差可以提高训练期间的数值稳定性。

让我们定义一个适用于MNIST图像（28x28灰度图）的卷积编码器。

```python
latent_dim = 2 # 使用2个维度以便于可视化

encoder_inputs = keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(16, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)

encoder = keras.Model(encoder_inputs, [z_mean, z_log_var], name="encoder")
encoder.summary()
```

编码器接收28x28x1图像，通过卷积层和全连接层对其进行处理，并输出`z_mean`和`z_log_var`向量，每个向量的大小均为`latent_dim`。

### 采样层（重参数 (parameter)化技巧）

为了以允许梯度反向传播 (backpropagation)通过采样过程的方式，从由$\mu$和$\log \sigma^2$定义的分布$q_\phi(z|x)$中采样，我们使用重参数化技巧：$z = \mu + \sigma \odot \epsilon$，其中$\epsilon$是从标准正态分布$\mathcal{N}(0, I)$中采样得到。我们可以将其实现为自定义Keras层。

```python
class Sampling(layers.Layer):
    """使用 (z_mean, z_log_var) 采样 z，即编码数字的向量。"""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        # 使用 tf.exp(0.5 * z_log_var) 得到 sigma
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
```

该层接收`z_mean`和`z_log_var`作为输入，并输出样本$z$。

### 解码器网络

解码器$p_\theta(x|z)$接收潜在空间中的点$z$并将其映射回数据空间，试图重建原始输入或生成新的相似样本。由于我们的输入是图像，解码器将使用转置卷积层将潜在向量 (vector)上采样回28x28x1图像。最终激活函数 (activation function)通常是sigmoid，用于将像素值归一化 (normalization)到0到1之间。

```python
latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
x = layers.Reshape((7, 7, 64))(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
# 最终层重建图像，使用sigmoid激活函数处理像素概率值 [0, 1]
decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)

decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()
```

## 组合组成部分形成变分自编码器模型

现在，我们将编码器、采样层和解码器连接起来，形成端到端的变分自编码器模型。我们定义一个自定义的Keras模型类，以处理其中包含ELBO损失计算的自定义训练步骤。

> 一张图，说明了训练期间通过变分自编码器架构的数据流，包括损失组成部分的计算。

```python
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.sampling = Sampling()
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            # 编码输入以获得均值和对数方差
            z_mean, z_log_var = self.encoder(data)
            # 从潜在分布中采样
            z = self.sampling([z_mean, z_log_var])
            # 解码潜在样本以重建输入
            reconstruction = self.decoder(z)

            # 计算重建损失（MNIST使用二元交叉熵）
            # 确保输入数据展平以匹配输出形状进行损失计算
            data_flat = tf.reshape(data, [-1])
            reconstruction_flat = tf.reshape(reconstruction, [-1])
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )

            # 计算KL散度损失
            # D_KL(N(mu, sigma^2) || N(0, 1)) = 0.5 * sum(sigma^2 + mu^2 - 1 - log(sigma^2))
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

            # 总损失是重建损失和KL损失之和（负ELBO）
            total_loss = reconstruction_loss + kl_loss

        # 计算梯度并更新权重
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # 更新指标
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def call(self, inputs):
        # 用于推断/预测
        z_mean, z_log_var = self.encoder(inputs)
        z = self.sampling([z_mean, z_log_var])
        return self.decoder(z)
```

在这个`VAE`类中：

- `__init__`方法存储编码器和解码器，并初始化用于总损失、重建损失和KL散度损失的指标追踪器。
- `train_step`方法覆盖默认训练逻辑。它执行前向传播，计算重建损失（使用适用于sigmoid输出和[0,1]归一化 (normalization)像素的二元交叉熵）并分析性地计算KL散度，计算总损失（负ELBO），然后应用梯度。
- `call`方法定义用于预测/推断的前向传播，其包含编码、采样和解码。

## 准备数据和训练变分自编码器

我们将使用标准MNIST数据集。像素值应归一化 (normalization)到[0, 1]范围，这与解码器最终层中的sigmoid激活函数 (activation function)匹配。

```python
# 加载并预处理MNIST数据集
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
mnist_digits = np.concatenate([x_train, x_test], axis=0)
mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255

# 实例化变分自编码器模型
vae = VAE(encoder, decoder)

# 编译模型
vae.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3))

# 训练变分自编码器
history = vae.fit(mnist_digits, epochs=30, batch_size=128) # 30个训练周期仅作示例

# 绘制训练损失
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='总损失')
plt.plot(history.history['reconstruction_loss'], label='重建损失')
plt.plot(history.history['kl_loss'], label='KL损失')
plt.title('变分自编码器训练损失')
plt.xlabel('训练周期')
plt.ylabel('损失')
plt.legend()
plt.grid(True)
plt.show()
```

在训练期间，您会观察到重建损失减少，因为变分自编码器学习重现输入数字，而KL损失促使潜在分布$q_\phi(z|x)$保持接近标准正态先验$p(z)$。这两项之间的平衡对于良好的生成性能非常重要。

## 使用训练好的变分自编码器生成新图像

变分自编码器的能力在于它能够生成新数据。我们通过从先验分布（在我们的情况下是标准高斯分布，$\mathcal{N}(0, I)$）中采样点$z$，并将其通过训练好的解码器网络$p_\theta(x|z)$来实现这一目标。

```python
def plot_latent_samples(vae, n=15, figsize=15):
    # 显示n*n的数字二维流形
    digit_size = 28
    scale = 1.0
    figure = np.zeros((digit_size * n, digit_size * n))

    # 在网格上线性间隔采样点，从N(0,I)边界
    # 使用正态分布的百分点函数(ppf)获得更平滑的覆盖
    from scipy.stats import norm
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))[::-1] # 反转y轴

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            # 从网格点采样潜在向量z
            z_sample = np.array([[xi, yi]]) * scale
            # 解码z以生成图像x_decoded
            x_decoded = vae.decoder.predict(z_sample, verbose=0)
            # 重塑并将数字放置在图中
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit

    plt.figure(figsize=(figsize, figsize))
    start_range = digit_size // 2
    end_range = n * digit_size - start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap="Greys_r")
    plt.title('由潜在空间样本生成的数字')
    plt.show()

plot_latent_samples(vae)
```

运行`plot_latent_samples`将生成一个数字网格。由于我们使用了`latent_dim=2`，我们可以在先验分布的大致范围内，在2D网格上采样点并可视化相应的生成数字。当您在潜在空间中移动时，您应该观察到不同数字风格之间的平滑过渡，这表明变分自编码器已经学习到了有意义的表示。

## 可视化学习到的潜在空间

我们还可以通过编码MNIST测试集并绘制生成的$z$向量 (vector)（根据其实际数字标签着色）来可视化训练数据在学习到的潜在空间中是如何组织的。这有助于理解编码器捕获的结构。

````python
def plot_label_clusters(vae, data, labels):
    # 显示潜在空间中数字类别的二维图
    z_mean, _ = vae.encoder.predict(data, verbose=0)
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels, cmap='tab10', alpha=0.7, s=5)
    plt.colorbar(scatter, label='数字类别')
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.title('VAE潜在空间中的MNIST测试集 (z_均值)')
    plt.grid(True)
    plt.show()

# 使用子集以加快绘图速度
num_samples_plot = 5000
plot_label_clusters(vae, x_test[:num_samples_plot], y_test[:num_samples_plot])

# Example Plotly chart (use hardcoded data for consistency if needed)
z_mean_plot, _ = vae.encoder.predict(x_test[:num_samples_plot], verbose=0)
labels_plot = y_test[:num_samples_plot]
plotly_fig = {
    "data": [
        {
            "x": z_mean_plot[:, 0].tolist(),
            "y": z_mean_plot[:, 1].tolist(),
            "mode": "markers",
            "marker": {
                "color": labels_plot.tolist(),
                "size": 5,
                "opacity": 0.7,
                "colorscale": "Viridis", # Example colorscale
                "colorbar": {"title": "数字类别"}
            },
            "type": "scatter"
        }
    ],
    "layout": {
        "title": "VAE潜在空间中的MNIST测试集 (z_均值)",
        "xaxis": {"title": "z轴0"},
        "yaxis": {"title": "z轴1"},
         "width": 700,
         "height": 600
    }
}
print("```plotly") # Start code block marker
import json
print(json.dumps(plotly_fig)) # Print the JSON string
print("```") # End code block marker
````



![VAE潜在空间中的MNIST测试集 (z_均值) (样本数据)](plots/2867-0.json)



> 二维潜在空间（显示近似后验分布$q_\phi(z|x)$的均值$\mu$），针对MNIST测试数字的样本，按类别着色。注意同一类别的数字如何倾向于聚集在一起，以及该空间如何展现出与数字相似性相关的某种结构。（使用样本数据进行说明）。

理想情况下，该图将显示对应不同数字的聚类，表明编码器已经学会将相似数字映射到潜在空间中的邻近位置。该结构可能不会完全分离，特别是在潜在维度较小且训练有限的情况下，但整体布局应该清晰可见。

这种实践性实现展现了用于图像生成的变分自编码器的核心组成部分和训练过程。您已经看到了如何定义编码器、解码器和采样层，实现ELBO损失，训练模型，并使用它来生成新数据和可视化学习到的表示空间。这形成了坚实的基础，用于进一步研究更高级的变分自编码器变体和后续讨论的运用。

## 参考资料

- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — Diederik P Kingma, Max Welling (2013)
  Journal: arXiv preprint arXiv:1312.6114; DOI: [10.48550/arXiv.1312.6114](https://doi.org/10.48550/arXiv.1312.6114)
  介绍了变分自编码器（VAE）模型，详细阐述了概率框架、重参数化技巧和ELBO目标，为本节提供了理论基础。
- [Variational AutoEncoder (VAE)](https://keras.io/examples/generative/vae/) — fchollet (2024)
  Journal: Keras Documentation
  官方Keras示例，使用Model子类化、自定义层和train_step提供了VAE的实践实现，与本节采用的方法一致。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 20
  第20章对变分自编码器进行了理论说明，包括ELBO和重参数化技巧的数学推导。
- [Customizing what happens in Model.fit()](https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit) — fchollet (2023)
  Journal: TensorFlow Documentation
  说明了如何使用tf.keras.Model子类化实现自定义训练步骤和损失计算，这对于实践示例中VAE的ELBO目标至关重要。
