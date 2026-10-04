---
course: "deep-learning-fundamentals-keras"
chapter: "convolutional-neural-networks-cnns"
lesson: "working-image-data-keras"
sourceId: 4947
sourceUrl: "https://apxml.com/zh/courses/deep-learning-fundamentals-keras/chapter-4-convolutional-neural-networks-cnns/working-image-data-keras"
title: "在Keras中处理图像数据"
description: "使用Keras工具加载、预处理和增强用于CNN训练的图像数据的方法。"
order: 6
plots: []
sourceHash: "95a3c8d0029f25805d064819bb44dddf2fd5df12e347830cb36558679edfc3ad"
sourceCorrections: []
---

卷积神经网络 (neural network) (CNN)善于处理网格状数据，图像就是一个典型例子。然而，原始图像文件不能被神经网络直接使用。它们需要经过加载、解码、可能调整到统一尺寸以及归一化 (normalization)处理，才能被送入CNN的输入层。Keras提供了一系列方便的工具，使得这一整个流程更为高效，尤其是在处理大型数据集时。

### 从目录加载图像

对图像数据集进行分类任务的常见组织方式是在主数据目录中为每个类别设置一个子目录。例如：

```
data/
├── train/
│   ├── cats/
│   │   ├── cat_001.jpg
│   │   ├── cat_002.jpg
│   │   └── ...
│   └── dogs/
│       ├── dog_001.jpg
│       ├── dog_002.jpg
│       └── ...
└── validation/
    ├── cats/
    │   ├── cat_101.jpg
    │   └── ...
    └── dogs/
        ├── dog_101.jpg
        └── ...
```

Keras提供了`image_dataset_from_directory`工具，它对于这种结构非常高效。此工具会自动根据目录名推断类别标签，并生成图像批次和相应的标签。

```python
import keras
import tensorflow as tf # Often needed for dataset operations even with other backends

# 定义图像尺寸和批次大小
img_height = 180
img_width = 180
batch_size = 32

# 加载训练数据
train_ds = keras.utils.image_dataset_from_directory(
    "data/train",
    labels='inferred',         # 从目录结构推断标签
    label_mode='int',         # 使用整数标签（例如，0代表猫，1代表狗）
    # 或者'categorical'用于one-hot编码标签，'binary'用于2个类别
    image_size=(img_height, img_width), # 加载时调整图像尺寸
    interpolation='nearest',   # 调整尺寸的方法
    batch_size=batch_size,
    shuffle=True               # 打乱数据
)

# 加载验证数据（通常不需要打乱）
val_ds = keras.utils.image_dataset_from_directory(
    "data/validation",
    labels='inferred',
    label_mode='int',
    image_size=(img_height, img_width),
    interpolation='nearest',
    batch_size=batch_size,
    shuffle=False
)

# 打印找到的类别名称
print("类别名称:", train_ds.class_names)
```

该函数返回一个`tf.data.Dataset`对象（即使通过Keras 3使用PyTorch或JAX等后端，TensorFlow的`tf.data` API也常用于高效的数据加载流程）。此对象会生成`(图像, 标签)`批次，其中`图像`是形状为`(批次大小, 图像高度, 图像宽度, 通道数)`的张量，`标签`是形状为`(批次大小,)`的张量。RGB图像的`通道数`通常为3，灰度图像为1。

### 图像预处理

原始像素值（通常是0到255的整数）对于神经网络 (neural network)训练来说往往不甚理想。以下两个常见的预处理步骤是必需的：

1. **尺寸调整**：CNN要求输入具有一致的形状。虽然`image_dataset_from_directory`可以在加载时调整图像尺寸，但您也可能以不同方式加载数据，或希望将尺寸调整作为模型定义的一部分。`keras.layers.Resizing`层可以添加到您的模型中。
2. **重缩放（归一化 (normalization)）**：当输入值较小时，神经网络通常表现更佳，这些值一般缩放到0到1之间，或-1到1之间。这有助于稳定训练，并使优化器更高效地收敛。`keras.layers.Rescaling`层非常适合此目的。

您可以使用Keras层将这些预处理步骤直接整合到您的模型中。这确保了预处理在训练、评估和推理 (inference)过程中得到一致应用，甚至可以在GPU上进行，以获取更好的性能。

```python
import keras
from keras import layers

# 定义输入形状（高、宽、通道）
input_shape = (img_height, img_width, 3) # 假定为RGB图像

# 创建包含预处理的模型
preprocessing_model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Rescaling(1./255), # 将像素值从 [0, 255] 缩放到 [0, 1]
        # 另一种选择，用于 [-1, 1] 缩放：layers.Rescaling(1./127.5, offset=-1)
    ]
)

# 之后您可以在此预处理块后面添加您的CNN层
# 示例：
# model = keras.Sequential([
#     preprocessing_model,
#     layers.Conv2D(...),
#     # ... 更多CNN层 ...
# ])
```

### 数据增强

深度学习 (deep learning)模型，特别是训练在图像上的CNN，从更大的数据集中获益匪浅。当收集更多数据成本高昂或不便时，**数据增强**提供了一种有效的替代方案。它通过对现有训练图像应用随机变换来生成略有修改但合理的全新训练样本。这有助于模型对位置、方向、亮度或缩放等变化更具不变性，从而获得更好的泛化性能并减少过拟合 (overfitting)。

Keras提供了一系列专门用于数据增强的预处理层。这些层在训练期间随机应用变换，但在评估或推理 (inference)期间不活跃。常见的数据增强层包括：

- `layers.RandomFlip("horizontal")`：随机水平翻转图像。
- `layers.RandomRotation(factor=0.1)`：随机旋转图像，旋转因子为2π的一部分（例如，0.1表示+/- 360度的10%）。
- `layers.RandomZoom(height_factor=0.2)`：随机垂直和水平放大或缩小图像。
- `layers.RandomContrast(factor=0.2)`：随机调整图像对比度。
- `layers.RandomBrightness(factor=0.2)`：随机调整图像亮度。

这些增强层通常放置在尺寸调整和重缩放之后，但在模型定义中的主要卷积层之前。

```python
import keras
from keras import layers

# 构建数据增强流水线
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
        # 根据需要添加更多增强层
    ],
    name="data_augmentation",
)

# 整合到完整的模型定义中
model = keras.Sequential([
    keras.Input(shape=input_shape),
    # 首先应用重缩放
    layers.Rescaling(1./255),
    # 应用增强（仅在训练期间活跃）
    data_augmentation,
    # 现在添加CNN基础
    layers.Conv2D(filters=32, kernel_size=3, activation="relu"),
    layers.MaxPooling2D(pool_size=2),
    layers.Conv2D(filters=64, kernel_size=3, activation="relu"),
    layers.MaxPooling2D(pool_size=2),
    # ... 用于分类的Flatten和Dense层 ...
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5), # Dropout是另一种正则化形式
    layers.Dense(1, activation='sigmoid') # 二分类示例
])

# 编译模型
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 现在您可以使用train_ds和val_ds训练此模型
# model.fit(train_ds, epochs=..., validation_data=val_ds)
```

> 准备用于Keras CNN模型的图像数据的一个示例流程，包括加载、重缩放和增强。

通过使用Keras的工具，例如`image_dataset_from_directory`以及预处理/增强层，您可以为CNN构建高效的输入流水线。请记住，细致的预处理和恰当的数据增强通常是在图像任务中取得良好表现的重要方面。

## 参考资料

- [Keras documentation: Working with images](https://www.tensorflow.org/guide/preprocessing_layers) — Keras team (2024)
  Publisher: TensorFlow
  Keras官方指南，涵盖Keras中高效的图像数据加载、预处理和数据增强技术，包括`image_dataset_from_directory`和预处理层。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  深度学习的开创性教材，为卷积神经网络、数据预处理和数据增强等正则化技术提供了详细的理论基础。
- [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python-second-edition) — François Chollet (2021)
  Publisher: Manning Publications
  使用Keras构建深度学习模型的实践指南，涵盖图像分类、数据加载、预处理和数据增强，并提供具体示例。
