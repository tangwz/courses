# 图像生成实践中的DCGAN实现

来源：[原文](https://apxml.com/zh/courses/cnns-for-computer-vision/chapter-7-gans-image-synthesis/implementing-dcgan-practice)

[返回章节目录](README.md) · [返回课程目录](../README.md)

使用TensorFlow或PyTorch等标准深度学习 (deep learning)框架，从头开始实现深度卷积GAN (DCGAN)，提供了对其架构和训练动态的实践理解。构建和训练GAN是巩固对生成器和判别器理解，并获得生成模型训练第一手经验的绝佳方式。

我们将关注核心组成部分：根据DCGAN原则定义生成器和判别器网络，设置对抗性损失函数 (loss function)，并构建协调两个网络之间竞争的训练循环。

### 环境与数据准备

首先，请确保您已安装必要的库。您通常需要：

- 深度学习 (deep learning)框架（TensorFlow或PyTorch）
- NumPy 用于数值计算
- Matplotlib 或类似的库用于可视化生成的图像

对于本次练习，常见的数据集选择包括MNIST、Fashion-MNIST，或者可能是CelebA（为了更快的实验，可以是裁剪/调整大小后的子集）。这些数据集提供现成的图像，适合学习图像生成的基础知识。

我们假设使用带有Keras API的TensorFlow和Fashion-MNIST数据集。第一步是加载和预处理数据。由于DCGAN生成器通常在其最后一层使用`tanh`激活函数 (activation function)，生成范围在 $[-1, 1]$ 内的输出，因此我们需要相应地标准化我们的真实图像。

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

# 加载数据集（例如，Fashion-MNIST）
(train_images, _), (_, _) = tf.keras.datasets.fashion_mnist.load_data()

# 预处理图像
# 添加通道维度，如果需要则调整大小（DCGAN通常在64x64上表现良好）
# 将图像标准化到 [-1, 1]
BUFFER_SIZE = train_images.shape[0]
BATCH_SIZE = 256 # 根据GPU内存调整
IMG_WIDTH = 28 # 如果调整大小则为64
IMG_HEIGHT = 28 # 如果调整大小则为64
CHANNELS = 1 # 彩色图像则为3

train_images = train_images.reshape(train_images.shape[0], IMG_WIDTH, IMG_HEIGHT, CHANNELS).astype('float32')
# 将图像标准化到 [-1, 1]
train_images = (train_images - 127.5) / 127.5

# 批处理并打乱数据
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# 噪声向量的潜在维度
NOISE_DIM = 100
```

### 构建生成器

生成器的作用是将随机噪声向量 (vector)（来自潜在空间，大小为`NOISE_DIM`）转换为类似于真实数据的图像。遵循DCGAN指导原则，我们使用`Conv2DTranspose`层进行上采样，`BatchNormalization`以稳定训练，以及`ReLU`（或`LeakyReLU`）激活函数 (activation function)。最后一层使用`tanh`。

典型的DCGAN生成器首先接收噪声向量，并通过`Dense`层将其投影到一个具有许多通道的小空间范围，然后进行重塑。接着，一系列`Conv2DTranspose`层逐渐增加空间维度，同时减少通道数量。

```python
def make_generator_model(noise_dim, channels, target_height, target_width):
    model = tf.keras.Sequential(name="生成器")

    # 从Dense层开始，将噪声投影成Conv2DTranspose适合的形状
    # 28x28输出的例子：从7x7开始
    # 64x64输出的例子：从4x4开始
    start_h, start_w = target_height // 4, target_width // 4 # 假设两次2倍上采样
    model.add(tf.keras.layers.Dense(start_h * start_w * 256, use_bias=False, input_shape=(noise_dim,)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2)) # 使用LeakyReLU

    model.add(tf.keras.layers.Reshape((start_h, start_w, 256)))
    assert model.output_shape == (None, start_h, start_w, 256) # 注意：None是批次大小

    # 上采样到 target_height/2 x target_width/2
    model.add(tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, target_height//2, target_width//2, 128)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))

    # 上采样到 target_height x target_width
    model.add(tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, target_height, target_width, 64)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))

    # 最后一层生成所需通道数的图像（例如，MNIST/FashionMNIST为1，彩色图像为3）
    model.add(tf.keras.layers.Conv2DTranspose(channels, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, target_height, target_width, channels)

    return model

generator = make_generator_model(NOISE_DIM, CHANNELS, IMG_HEIGHT, IMG_WIDTH)
generator.summary() # 打印模型概况
```

> 28x28输出的DCGAN生成器结构，将100维噪声向量转换为图像。

### 构建判别器

判别器是一个标准的CNN二分类器。它接收图像（无论是真实图像还是生成图像）作为输入，并输出一个指示概率值，该概率值表示输入图像是真实的（接近1）还是伪造的（接近0）。DCGAN建议使用带步长卷积的`Conv2D`层进行下采样（而非池化），使用`LeakyReLU`激活函数 (activation function)，并可能使用`BatchNormalization`（尽管有时会省略，特别是在第一层或使用其他正则化 (regularization)方法如梯度惩罚时）。最后一层是一个带有一个输出单元和`sigmoid`激活函数的`Dense`层。

```python
def make_discriminator_model(img_height, img_width, channels):
    model = tf.keras.Sequential(name="判别器")
    input_shape = (img_height, img_width, channels)

    # 下采样到14x14（对于28x28输入）
    model.add(tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=input_shape))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    model.add(tf.keras.layers.Dropout(0.3)) # 用于正则化的Dropout

    # 下采样到7x7
    model.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    model.add(tf.keras.layers.Dropout(0.3))

    # 展平并分类
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1, activation='sigmoid')) # Sigmoid用于概率输出

    return model

discriminator = make_discriminator_model(IMG_HEIGHT, IMG_WIDTH, CHANNELS)
discriminator.summary() # 打印模型概况
```

> 28x28输入的DCGAN判别器结构，将图像分类为真实或伪造。

### 定义损失函数 (loss function)与优化器

GAN训练的核心在于对抗性损失。我们对两个网络都使用`BinaryCrossentropy`。

- **判别器损失：** 这包括两部分。一部分是正确分类真实图像的损失（目标为1），另一部分是正确分类伪造图像的损失（目标为0）。判别器总损失是这两部分之和。
- **生成器损失：** 生成器旨在使判别器将其生成的伪造图像分类为真实图像。因此，我们根据判别器对伪造图像的输出计算生成器的损失，但使用目标标签1。

我们通常为生成器和判别器使用独立的Adam优化器。DCGAN论文推荐学习率为0.0002，Beta1为0.5。

```python
# 使用二元交叉熵损失
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False) # 如果最后一层没有sigmoid/tanh，则使用 from_logits=True

# 判别器损失
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

# 生成器损失
def generator_loss(fake_output):
    # 生成器希望判别器认为伪造图像是真实的（标签1）
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# 优化器（Adam在GAN中很常见）
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5) # 示例值
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5) # 示例值
```

### 训练循环

训练循环需要仔细协调。在每一步中，我们分别训练判别器和生成器。

1. 生成一批随机噪声向量 (vector)。
2. 使用生成器从噪声中创建一批伪造图像。
3. 从数据集中获取一批真实图像。
4. **训练判别器：**
   - 将真实图像输入判别器，并计算其与标签1的损失。
   - 将伪造图像输入判别器，并计算其与标签0的损失。
   - 将真实损失和伪造损失相加。
   - 计算梯度并更新判别器的权重 (weight)。
5. **训练生成器：**
   - 再次将*相同*批次的噪声向量（或新批次）输入生成器（在生成器的梯度带范围内）。
   - 将*生成的*伪造图像输入判别器。
   - 根据判别器对伪造图像的输出计算生成器损失，使用标签1（尝试欺骗判别器）。
   - 计算梯度（仅针对生成器的变量）并更新生成器的权重。

这个过程通常在TensorFlow中使用`tf.function`封装以优化性能。

```python
# 我们将重复使用这个种子（这样更容易）
# 可视化动态GIF中的进度）
seed = tf.random.normal([16, NOISE_DIM]) # 固定噪声以保持可视化一致性

# 注意`tf.function`的使用
# 这个注解会使函数被“编译”。
@tf.function
def train_step(images, generator, discriminator, gen_optimizer, disc_optimizer, noise_dim):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    # 计算梯度
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    # 应用梯度
    gen_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss

# 训练函数
def train(dataset, epochs, generator, discriminator, gen_optimizer, disc_optimizer, noise_dim, seed):
    history = {'gen_loss': [], 'disc_loss': []}
    for epoch in range(epochs):
        start = time.time()
        epoch_gen_loss = []
        epoch_disc_loss = []

        for image_batch in dataset:
            g_loss, d_loss = train_step(image_batch, generator, discriminator, gen_optimizer, disc_optimizer, noise_dim)
            epoch_gen_loss.append(g_loss.numpy())
            epoch_disc_loss.append(d_loss.numpy())

        # 随着训练进展生成用于GIF的图像
        generate_and_save_images(generator, epoch + 1, seed) # 函数定义在下方

        avg_gen_loss = np.mean(epoch_gen_loss)
        avg_disc_loss = np.mean(epoch_disc_loss)
        history['gen_loss'].append(avg_gen_loss)
        history['disc_loss'].append(avg_disc_loss)

        print(f'Time for epoch {epoch + 1} is {time.time()-start:.2f} sec')
        print(f'Generator Loss: {avg_gen_loss:.4f}, Discriminator Loss: {avg_disc_loss:.4f}')

    # 在最后一个epoch后生成
    generate_and_save_images(generator, epochs, seed)
    return history

# 生成并保存图像的辅助函数
def generate_and_save_images(model, epoch, test_input):
    # 注意`training`被设置为False。
    # 这是为了让所有层都在推理模式下运行（批量归一化）。
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        # 根据通道数显示灰度或彩色
        if predictions.shape[-1] == 1:
           plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        else:
           plt.imshow(predictions[i, :, :, :] * 0.5 + 0.5) # 将图像从[-1,1]反标准化到[0,1]
        plt.axis('off')

    # 保存或显示图像
    # plt.savefig(f'image_at_epoch_{epoch:04d}.png')
    plt.show()

# 开始训练
EPOCHS = 50 # 根据需要调整
history = train(train_dataset, EPOCHS, generator, discriminator, generator_optimizer, discriminator_optimizer, NOISE_DIM, seed)
```

### 监控训练进度

GAN的训练可能不稳定。监控生成器和判别器的损失很重要。理想情况下，损失应该达到某种平衡，尽管它们经常显著波动。任何一方的损失收敛到零通常都不是好兆头；如果判别器损失降至零，这意味着生成器学习效果不佳。反之，如果生成器损失下降过快过低，判别器可能效果不佳，可能导致模式崩溃。

通过epoch对生成样本进行目视检查，可以说是评估进展最实用的方式。图像是否变得更真实、更多样化了？



[交互图表：GAN训练损失示例](https://apxml.com/zh/courses/cnns-for-computer-vision/chapter-7-gans-image-synthesis/implementing-dcgan-practice#plot-r70qq8)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "data": [
    {
      "type": "scatter",
      "mode": "lines",
      "name": "生成器损失",
      "x": [
        1,
        5,
        10,
        15,
        20,
        25,
        30,
        35,
        40,
        45,
        50
      ],
      "y": [
        3.5,
        2.8,
        2.5,
        2.9,
        3.2,
        3.0,
        2.8,
        3.1,
        3.3,
        3.5,
        3.6
      ],
      "line": {
        "color": "#1c7ed6"
      }
    },
    {
      "type": "scatter",
      "mode": "lines",
      "name": "判别器损失",
      "x": [
        1,
        5,
        10,
        15,
        20,
        25,
        30,
        35,
        40,
        45,
        50
      ],
      "y": [
        0.4,
        0.6,
        0.8,
        0.7,
        0.6,
        0.75,
        0.85,
        0.7,
        0.65,
        0.6,
        0.55
      ],
      "line": {
        "color": "#f03e3e"
      }
    }
  ],
  "layout": {
    "title": {
      "text": "GAN训练损失示例"
    },
    "xaxis": {
      "title": {
        "text": "训练周期"
      }
    },
    "yaxis": {
      "title": {
        "text": "损失"
      }
    },
    "template": "plotly_white"
  }
}
```

</details>



> 示例图显示了训练过程中生成器和判别器损失的波动。通常期望达到平衡或受控的振荡，而不是收敛到零。

### 实验与后续步骤

此实现为DCGAN提供了实现基础。成功训练GAN通常需要实验：

- **超参数 (parameter) (hyperparameter)：** 调整学习率、优化器Beta值、批次大小和噪声维度。
- **架构：** 修改层数、滤波器大小或激活函数 (activation function)（例如，尝试`ReLU`与`LeakyReLU`）。
- **数据：** 尝试不同的数据集或更复杂的预处理。
- **正则化 (regularization)：** 尝试标签平滑或向输入/标签添加噪声等方法以稳定训练。

构建和训练此DCGAN提供了生成模型方面的宝贵实践经验，为您了解更高级的GAN变体（如条件GAN或StyleGAN，如本章前面所讨论的）做准备。请记住，GAN训练可能很敏感，因此需要坚持不懈和仔细监控。

## 参考资料

- [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) — Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio (2014)
  Journal: Advances in Neural Information Processing Systems; Volume: 27; DOI: [10.48550/arXiv.1406.2661](https://doi.org/10.48550/arXiv.1406.2661)
  这篇基础论文介绍了生成对抗网络的概念及对抗训练框架。
- [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434) — Alec Radford, Luke Metz, Soumith Chintala (2015)
  Journal: arXiv preprint arXiv:1511.06434; DOI: [10.48550/arXiv.1511.06434](https://doi.org/10.48550/arXiv.1511.06434)
  定义了DCGAN架构、构建稳定深度卷积GAN的指南，以及Conv2DTranspose、BatchNormalization、LeakyReLU和tanh激活等设计选择。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press
  这是一本涵盖深度学习理论基础的综合性教材，包含卷积神经网络、生成模型和与GAN相关的优化技术等详细章节。
- [Deep Convolutional Generative Adversarial Network (DCGAN)](https://www.tensorflow.org/tutorials/generative/dcgan) — TensorFlow Developers (2023)
  这是TensorFlow的官方教程，演示了使用Keras API实现DCGAN的实践方法，涵盖了数据预处理、模型构建、损失函数和训练循环，与本节的代码示例直接相关。

---

[上一节](06-GAN%20%E7%9A%84%E8%AF%84%E4%BC%B0%E6%8C%87%E6%A0%87.md) · [下一节](../08-%E6%A8%A1%E5%9E%8B%E5%8E%8B%E7%BC%A9%E4%B8%8E%E9%AB%98%E6%95%88%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/01-%E9%AB%98%E6%95%88%E6%A8%A1%E5%9E%8B%E7%9A%84%E5%8A%A8%E5%9B%A0.md)
