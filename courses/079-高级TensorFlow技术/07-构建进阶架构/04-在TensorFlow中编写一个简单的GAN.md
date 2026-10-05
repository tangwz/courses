# 在TensorFlow中编写一个简单的GAN

来源：[原文](https://apxml.com/zh/courses/advanced-tensorflow/chapter-7-implementing-advanced-architectures/coding-simple-gan)

[返回章节目录](README.md) · [返回课程目录](../README.md)

生成对抗网络（GAN）涉及设置两个相互竞争的神经网络 (neural network)——生成器和判别器，并以对抗方式同时训练它们。一个基本的GAN将使用TensorFlow和Keras进行构建，目标是生成与MNIST数据集相似的图像。

### 生成器网络

生成器的任务是创建模仿真实数据分布的合成数据。它接收一个随机噪声向量 (vector)（通常取自高斯分布或均匀分布）作为输入，并将其转换为与真实数据具有相同结构（例如，MNIST的28x28灰度图像）的输出。

一个简单的生成器可以使用`tf.keras.Sequential`构建。我们将从一个Dense层开始，将输入噪声投影到更高维的空间，然后进行重塑，如果构建的是卷积GAN（DCGAN），则可能使用转置卷积（`Conv2DTranspose`）。为简单起见，这里我们使用Dense层进行演示，它适用于生成扁平化的MNIST图像或适应简单的图像结构。

```python
import tensorflow as tf

def build_generator(latent_dim, output_shape):
    model = tf.keras.Sequential(name='Generator')
    model.add(tf.keras.layers.Input(shape=(latent_dim,)))
    # 使用Dense层的示例 - 根据具体任务调整架构
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(output_shape, activation='tanh')) # 对于缩放到[-1, 1]的输出使用tanh激活函数
    return model

# 扁平化MNIST图像（28*28 = 784）的示例用法
latent_dim = 100
output_dim = 784
generator = build_generator(latent_dim, output_dim)
generator.summary() # 显示模型结构
```

`latent_dim`参数 (parameter)定义了输入噪声向量的大小。最终的激活函数 (activation function)（例如，`tanh`或`sigmoid`）应与真实数据的预期范围相匹配。对于归一化 (normalization)到`[-1, 1]`的MNIST图像，`tanh`是合适的。

### 判别器网络

判别器充当二分类器。它的输入可以是真实数据样本，也可以是生成器产生的虚假样本。它的目标是输出一个概率，表示输入是真实的（概率接近1）还是虚假的（概率接近0）。

与生成器类似，一个简单的判别器可以是`tf.keras.Sequential`模型。它通常由Dense层（或图像数据的卷积层）组成，然后是一个带有单个输出单元和`sigmoid`激活函数 (activation function)的最终Dense层，以产生概率分数。

```python
def build_discriminator(input_shape):
    model = tf.keras.Sequential(name='Discriminator')
    model.add(tf.keras.layers.Input(shape=(input_shape,)))
    # 使用Dense层的示例
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3)) # 正则化会有帮助
    model.add(tf.keras.layers.Dense(1, activation='sigmoid')) # 输出概率
    return model

# 扁平化MNIST图像（784）的示例用法
discriminator = build_discriminator(output_dim) # 输入与生成器输出/真实数据匹配
discriminator.summary()
```

### 定义损失函数 (loss function)

对抗训练需要为生成器和判别器设置不同的损失函数。我们通常使用二元交叉熵损失（`tf.keras.losses.BinaryCrossentropy`），因为判别器执行的是二分类（真实 vs. 虚假）。

- **判别器损失 ($L_D$)**: 此损失旨在促使判别器对真实图像输出1，对虚假图像输出0。它由两部分组成：真实图像上的损失和虚假图像上的损失。

  
  $$
  L_D = - \frac{1}{m} \sum_{i=1}^{m} [\log(D(x^{(i)})) + \log(1 - D(G(z^{(i)})))]
  $$
  

  其中$D(x)$是判别器对真实数据$x$的输出，$G(z)$是生成器对噪声$z$的输出，$m$是批量大小。
- **生成器损失 ($L_G$)**: 此损失旨在促使生成器产生判别器将其分类为真实（输出1）的输出。

  
  $$
  L_G = - \frac{1}{m} \sum_{i=1}^{m} \log(D(G(z^{(i)})))
  $$
  

我们可以使用`tf.keras.losses.BinaryCrossentropy`来实现这些。请注意，对于生成器损失，我们将判别器对虚假图像的输出与标签1（真实）进行比较。

```python
# 如果判别器的最终层没有sigmoid激活函数，则使用from_logits=True
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    # 生成器希望判别器认为虚假图像是真实的（标签1）
    return cross_entropy(tf.ones_like(fake_output), fake_output)
```

### 优化器

由于生成器和判别器有不同的目标并单独更新，我们需要为它们各自设置独立的优化器。Adam优化器是常用的一种。

```python
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
```

学习率可能需要调整；有时生成器和判别器会使用不同的学习率。

### 训练循环

GAN训练需要自定义训练循环，因为生成器和判别器的更新必须精心协调。标准的`model.fit()`不直接适用。我们将使用`tf.GradientTape`来计算每个网络的梯度。

以下是单个训练步骤的结构，通常用`tf.function`封装以优化性能：

```python
# 假设 'real_images' 是数据集（例如 MNIST）中的一个批次
# 假设 'latent_dim' 已定义

@tf.function
def train_step(real_images, generator, discriminator, gen_optimizer, disc_optimizer, batch_size, latent_dim):
    # 1. 生成噪声
    noise = tf.random.normal([batch_size, latent_dim])

    # 使用 GradientTape 进行自动微分
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # 2. 生成虚假图像
        generated_images = generator(noise, training=True)

        # 3. 获取判别器对真实和虚假图像的预测
        real_output = discriminator(real_images, training=True)
        fake_output = discriminator(generated_images, training=True)

        # 4. 计算损失
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    # 5. 计算梯度
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    # 6. 应用梯度来更新权重
    gen_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss
```

这个`train_step`函数封装了对抗训练过程的一次迭代：生成虚假数据、评估两个网络、计算损失、计算梯度并更新权重 (weight)。

### 整合（训练周期）

完整的训练过程包括在多个周期和真实数据集的多个批次上迭代执行`train_step`。

```python
# 训练循环（需要加载数据集、周期循环等）
# epochs = ...
# batch_size = ...
# dataset = load_and_prepare_mnist_dataset(...) # 归一化到 [-1, 1]

# for epoch in range(epochs):
#     print(f"周期 {epoch+1}/{epochs}")
#     epoch_gen_loss_avg = tf.keras.metrics.Mean()
#     epoch_disc_loss_avg = tf.keras.metrics.Mean()

#     for image_batch in dataset: # 假设数据集产生真实图像的批次
#         gen_loss, disc_loss = train_step(
#             image_batch,
#             generator,
#             discriminator,
#             generator_optimizer,
#             discriminator_optimizer,
#             batch_size,
#             latent_dim
#         )
#         epoch_gen_loss_avg.update_state(gen_loss)
#         epoch_disc_loss_avg.update_state(disc_loss)

#     print(f"生成器损失: {epoch_gen_loss_avg.result():.4f}, 判别器损失: {epoch_disc_loss_avg.result():.4f}")

#     # 在此处添加代码以定期保存检查点并生成样本图像
#     # 在每个周期结束时重置指标
#     epoch_gen_loss_avg.reset_states()
#     epoch_disc_loss_avg.reset_states()
```

这种结构为编写一个简单的GAN提供了基础。具体内容包括定义独立的生成器和判别器网络，根据二元交叉熵设置各自的损失函数 (loss function)，使用独立的优化器，并使用`tf.GradientTape`在自定义训练循环中协调它们的更新。监测损失值和定期可视化生成器的输出是评估训练进展的重要步骤。请记住，GAN训练可能不稳定；通常需要仔细调整超参数 (parameter) (hyperparameter)（学习率、网络架构）。

## 参考资料

- [Generative Adversarial Nets](https://proceedings.neurips.cc/paper_files/paper/2014/file/f033ed80deb0234979a61f95710dbe25-Paper.pdf) — Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio (2014)
  Journal: Advances in Neural Information Processing Systems; Publisher: Curran Associates, Inc.; Volume: 27; Pages: 2672-2680
  这篇原始论文介绍了生成对抗网络（GAN）框架，详细阐述了其概念基础和初步成果。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  这本综合性教材包含专门的生成对抗网络章节，提供了详细的理论解释和各种架构考量。
- [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434) — Alec Radford, Luke Metz, Soumith Chintala (2015)
  Journal: International Conference on Learning Representations (ICLR 2016); DOI: [10.48550/arXiv.1511.06434](https://doi.org/10.48550/arXiv.1511.06434)
  这篇论文介绍了用于稳定训练卷积生成对抗网络（DCGANs）的架构指导和技术，适用于使用卷积层将基本GAN实现扩展到图像生成。
- [Generate images with a Deep Convolutional Generative Adversarial Network (DCGAN)](https://www.tensorflow.org/tutorials/generative/dcgan) — TensorFlow Authors (2024)
  Publisher: TensorFlow
  一份官方TensorFlow教程，提供了使用TensorFlow 2和Keras实现DCGAN的实践指南，涵盖了自定义训练循环和`tf.GradientTape`。

---

[上一节](03-%E7%94%9F%E6%88%90%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9C%20%28GANs%29%20%E5%8E%9F%E7%90%86.md) · [下一节](05-TensorFlow%E5%9B%BE%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%EF%BC%88GNN%EF%BC%89%E5%9F%BA%E7%A1%80.md)
