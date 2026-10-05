# 在 Julia 中使用预训练模型

来源：[原文](https://apxml.com/zh/courses/julia-deep-learning/chapter-5-advanced-topics-gpu-computing/working-pre-trained-models-julia)

[返回章节目录](README.md) · [返回课程目录](../README.md)

从头构建复杂的深度学习 (deep learning)模型需要大量数据、计算资源和时间。幸运的是，您不总是需要从零开始。预训练 (pre-training)模型是已在大型标准数据集（如图像的 ImageNet 或语言的大型文本语料库）上训练过的网络，它们提供了一个很好的起点。在 Julia 生态系统中使用这些模型可以加速项目并提升性能，尤其是在数据集有限时。

### 预训练 (pre-training)模型的优势

使用一个已经学会从数据集中识别通用特征的模型，可以提供一个显著的领先优势。例如，在 ImageNet 上训练的模型已经学会识别边缘、纹理、模式乃至复杂的物体部分。这些学到的特征对各种计算机视觉任务通常都很有用。主要好处包括：

- **减少训练时间：** 由于模型已经学习了丰富的特征集，您将需要更少的时间和数据来使其适应您的特定问题。
- **提升性能：** 预训练模型通常是非常好的特征提取器，能让您的目标任务表现更出色，尤其是在数据集较小，从头训练深度网络可能导致过拟合 (overfitting)的情况下。
- **获取复杂架构：** 您可以使用经过充分测试的、先进的架构，而无需自己实现它们。

### 使用 Metalhead.jl 在 Julia 中获取预训练 (pre-training)模型

对于 Julia 中的计算机视觉任务，`Metalhead.jl` 是一个首选包。它提供了一系列流行的架构，如 ResNet、VGG、DenseNet、MobileNet 等，这些架构的权重 (weight)已经在 ImageNet 数据集上进行了预训练。

要开始使用 `Metalhead.jl`，您首先需要将其添加到 Julia 环境中：

```julia
using Pkg
Pkg.add("Metalhead")
```

安装后，您可以轻松加载预训练模型。例如，要加载 ResNet-18 模型：

```julia
using Flux, Metalhead

# 加载带有预训练权重的 ResNet-18 模型
model = ResNet(18; pretrain=true)
```

将 `pretrain=true` 设置为真，可以确保模型使用从 ImageNet 中学习到的权重进行初始化。如果 `pretrain=false`（某些模型在未指定时的默认值），则模型架构将以随机权重加载。

### 使用预训练 (pre-training)模型进行推断

预训练模型最直接的用途是进行推断：凭借它已有的知识对新数据进行预测。例如，一个在 ImageNet 上训练的模型可以将图像分类到 1000 个 ImageNet 类别中的一个。

以下是图像分类的简化流程：

1. **加载模型：** 如上所示。
2. **准备输入数据：** 预训练模型期望特定格式的输入数据。这通常包括：
   - 将图像调整到期望的输入尺寸（例如 224x224 像素）。
   - 根据原始训练期间使用的统计数据（通常是 ImageNet 特定的均值和标准差）对像素值进行归一化 (normalization)。`Metalhead.jl` 模型通常期望输入是一个 4D 数组（`WHCN` 格式：宽度、高度、通道、批量大小），像素值缩放到 `[0, 1]`。对于具体的归一化细节，查阅 `Metalhead.jl` 文档或模型权重 (weight)来源总是一个好做法。
   - 添加批量维度。
3. **进行预测：** 将准备好的输入传递给模型。
4. **解释输出：** 输出通常是一个向量 (vector)，包含每个类别的分数（对数几率或概率）。

```julia
using Flux, Metalhead, Images, CUDA # 假设使用 Images.jl 进行图像加载

# 1. 加载模型
model = ResNet(18; pretrain=true)

# 检查 GPU 是否可用，如果可以则将模型移至 GPU
if CUDA.functional()
    model = model |> gpu
    @info "模型已移至 GPU"
else
    @info "GPU 不可用，使用 CPU"
end

# 2. 准备输入数据（简化版本 - 实际预处理可能更复杂）
# 替换为您的图像加载和预处理逻辑
function preprocess_image(img_path)
    img = Images.load(img_path)
    # 调整大小为 224x224，转换为 Float32，将维度重新排列为 WHCN，然后归一化
    # 如果使用 Metalhead 自己的工具，它默认期望 CHWN 格式， 
    # 但 Flux 层期望 WHCN 格式。我们假设输入需要是 WHCN。
    # Metalhead 模型典型的预处理序列：
    img_resized = Images.imresize(img, (224, 224))
    img_float = Float32.(channelview(img_resized)) # 变为 CHW
    img_permuted = permutedims(img_float, (2, 3, 1)) # HWC -> WHC (或直接到 CHW，然后排列为 WHCN)
                                                     # 对于 Flux 卷积层，它是 WHCN。
                                                     # Metalhead 自己的工具可能处理了这一点。
                                                     # 我们将目标设为 WHCN，以便直接用于 Flux。
    # 假设 img_float 是 channelview 得到的 CHW 格式：
    img_whc = permutedims(img_float, (3, 2, 1)) # CHW -> WHC

    # 添加批量维度
    img_batch_cpu = Flux.unsqueeze(img_whc, 4) # WHCN
    return img_batch_cpu
end

img_data_cpu = preprocess_image("path/to/your/image.jpg") # 替换为实际路径

# 如果模型在 GPU 上，则将数据移至 GPU
img_data = CUDA.functional() ? (img_data_cpu |> gpu) : img_data_cpu

# 3. 进行预测
output = model(img_data)

# 4. 解释输出
probabilities = softmax(vec(cpu(output))) # 将输出移至 CPU 进行后处理
# Metalhead.jl 提供了 imagenet_labels() 用于获取类别名称
# class_labels = Metalhead.imagenet_labels()
# top_class_idx = argmax(probabilities)
# println("预测类别: $(class_labels[top_class_idx]) 概率: $(probabilities[top_class_idx])")
```

请记住，输入预处理非常重要。输入尺寸、归一化或通道顺序不匹配将导致结果不佳或毫无意义。务必查阅您正在使用的特定模型或库的文档。

### 迁移学习 (transfer learning)：根据您的需求调整模型

虽然直接推断很有用，但预训练 (pre-training)模型的真正优势通常来自于迁移学习。这包括将预训练模型适应于一项新任务，该任务与模型最初训练的任务不同但相关。迁移学习主要有两种策略：

#### 1. 特征提取

在这种方法中，您将预训练模型（或其一部分，通常是卷积基）视为固定的特征提取器。其思路是，卷积神经网络 (neural network) (CNN)的早期层学习通用特征（边缘、纹理），而后期层学习更特定于任务的特征。对于新任务，这些通用特征仍然可以提供非常有用的信息。

**工作流程：**

1. **加载预训练模型：** 例如，`ResNet(18; pretrain=true)`。
2. **隔离特征提取层：** 大多数 `Metalhead.jl` 模型，如 `ResNet`，都构建为 `Chain` 结构，其中第一个元素包含卷积基（特征提取器），第二个元素是分类器。您可以只选择特征提取器部分。

   ```julia
   base_model = ResNet(18; pretrain=true)
   feature_extractor = base_model.layers[1] # 这是 ResNet 的卷积基
   ```
3. **冻结权重 (weight)：** 由于我们希望将其用作固定提取器，因此阻止其权重在您的新任务训练期间更新。

   ```julia
   Flux.freeze!(feature_extractor)
   ```
4. **添加新的分类器：** 在冻结的特征提取器之上，添加新的、随机初始化的层（例如 `Dense` 层）。这个新的“头部”将在您的自定义数据集上从头开始训练。

   ```julia
   # 示例：ResNet-18 的特征提取器输出 512 个特征
   # 假设我们的新任务有 10 个类别
   num_new_classes = 10
   new_classifier_head = Chain(
       AdaptiveMeanPool((1,1)), # 从卷积基中汇聚特征
       Flux.flatten,            # 将输出展平以用于全连接层
       Dense(512, num_new_classes) # 512 是 ResNet-18 基的输出
   )

   # 组合成一个新模型
   transfer_model = Chain(feature_extractor, new_classifier_head)
   ```
5. **训练新模型：** 在您的数据集上训练 `transfer_model`。只有 `new_classifier_head` 的权重会被更新。

当您的数据集很小且与模型训练的原始数据集差异很大时，特征提取特别有效。由于您只训练一个小型的新分类器，因此通常可以用有限的数据获得良好结果并避免过拟合 (overfitting)。

#### 2. 微调 (fine-tuning)

微调将迁移学习向前推进了一步。它不是完全冻结预训练模型的权重，而是允许其中一些权重（通常在后期层）在您的新数据集上训练期间进行更新，通常使用非常小的学习率。

**工作流程：**

1. **加载预训练模型并替换头部：** 从加载预训练模型并将其最终分类层替换为适合您任务的新层开始，这与特征提取类似。

   ```julia
   model_to_finetune = ResNet(18; pretrain=true)
   feature_extractor_base = model_to_finetune.layers[1]

   num_new_classes = 10 # 您的类别数量
   # 假设原始全连接层之前的 ResNet-18 输出是 512 个特征
   new_head = Chain(
       AdaptiveMeanPool((1,1)), 
       Flux.flatten,
       Dense(512, num_new_classes)
   )
   finetune_model = Chain(feature_extractor_base, new_head)
   ```
2. **解冻某些层（可选但常见）：** 决定您想要微调预训练基础模型的哪些层。通常的做法是解冻卷积基的顶部几层或几个块，因为这些层学习更专门的特征。

   ```julia
   # 示例：解冻 ResNet-18 特征提取器基础的最后一个块中的层
   # 确切结构取决于模型；请检查 `feature_extractor_base.layers`
   # 对于 ResNet，层被分组到不同的阶段。假设我们解冻最后一个阶段。
   # 对于 ResNet，model.layers[1] 是一个 Chain 阶段。
   # Flux.unfreeze!(finetune_model.layers[1][end]) # 解冻最后一个阶段
   # 或者解冻基础模型中的所有参数以进行完全微调
   Flux.unfreeze!(finetune_model.layers[1])
   ```
3. **使用小学习率训练：** 微调通常包括两个阶段：
   - 首先，在基础模型冻结的情况下（如特征提取），只训练新添加的分类头部。这会初始化头部。
   - 然后，解冻基础模型的所需层，并继续使用非常小的学习率（例如 $10^{-4}$ 或 $10^{-5}$）训练整个网络。这可以防止预训练权重被过度或过快地扭曲。

当您的数据集较大且与原始数据集有些相似时，微调通常会很有益。它使模型能够使其学习到的特征更紧密地适应您的特定任务。

**迁移学习中的 GPU 加速：**
如果 CUDA 可用，请记住使用 `model |> gpu` 和 `data |> gpu` 将您的 `transfer_model` 或 `finetune_model` 和数据移动到 GPU。这对于较大的预训练模型和微调尤为重要，因为它们可能计算密集。

```julia
if CUDA.functional()
    finetune_model = finetune_model |> gpu
    # 确保您的数据加载器也提供 GPU 上的数据
end

# 微调的优化器示例
# opt = Adam(1e-4) # 小学习率

# 继续您的训练循环（如第 4 章所述）
# train!(loss, Flux.params(finetune_model), train_dataloader, opt; cb = ...)
```

### 在其他方向：预训练 (pre-training)模型

尽管 `Metalhead.jl` 在视觉方面表现突出，但使用预训练模型的原则也适用于其他方向。对于自然语言处理（NLP），Julia 生态系统正在出现像 `Transformers.jl` 这样的库，它们提供了对 BERT 或 GPT 等预训练语言模型的支持。加载模型、准备数据以及通过特征提取或微调 (fine-tuning)来适应它们的通用工作流程仍然相似，尽管模型架构和数据预处理的具体细节会有所不同。

### 保存和加载您调整后的模型

一旦您调整了预训练 (pre-training)模型并在您的任务上对其进行了训练，您就会希望保存它。如第 3 章所述，`BSON.jl` 是序列化 Julia 对象（包括 Flux 模型）的常用选择：

```julia
using BSON

# 假设 'my_custom_model' 是您训练好的迁移学习模型
BSON.@save "my_custom_model.bson" my_custom_model

# 要重新加载：
BSON.@load "my_custom_model.bson" loaded_model # loaded_model 将是保存时使用的名称
# 然后 loaded_model 将包含您的模型架构和训练好的权重。
```

使用预训练模型是一种强大的技术，可以显著提高您在深度学习 (deep learning)中的生产力以及模型性能。通过了解如何在 Julia 中获取、应用和调整这些模型，您可以在这些大型网络中编码的集体知识基础上，更有效地应对自己的具体难题。随着您的进步，您会发现这些技能对于各种应用都非常有价值。

## 参考资料

- [How transferable are features in deep neural networks?](https://proceedings.neurips.cc/paper_files/paper/2014/file/fc490ca45c00b1249bbe3554a4fdf6fb-Paper.pdf) — Jason Yosinski, Jeff Clune, Yoshua Bengio, Hod Lipson (2014)
  Journal: Advances in Neural Information Processing Systems 27 (NIPS 2014); Publisher: Neural Information Processing Systems Foundation, Inc.; Volume: 27; Pages: 3320-3328
  论证深度卷积神经网络学习特征的可迁移性的基础性论文，这是预训练模型背后的核心原理。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  全面介绍深度学习理论和实践的教科书，包含迁移学习和预训练模型的相关章节。
- [Metalhead.jl Documentation](https://fluxml.ai/Metalhead.jl/stable/) — The Julia Flux Community (2025)
  Metalhead.jl 的官方文档，提供 Julia 中计算机视觉任务的可用预训练模型、使用方法和示例。
- [Flux.jl Documentation](https://fluxml.ai/Flux.jl/stable/) — The Julia Flux Community (2025)
  Flux.jl 的官方文档，它是 Julia 中底层的深度学习框架，对于理解模型构建、训练循环和如 `freeze!` 等实用工具至关重要。

---

[上一节](03-Flux%E6%A8%A1%E5%9E%8B%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90%E4%B8%8E%E4%BC%98%E5%8C%96.md) · [下一节](05-%E4%BD%BF%E7%94%A8%20Flux%20%E4%BB%8B%E7%BB%8D%E7%94%9F%E6%88%90%E6%A8%A1%E5%9E%8B.md)
