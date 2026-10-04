---
course: "deep-learning-fundamentals-keras"
chapter: "improving-model-performance-workflow"
lesson: "practical-applying-improvement-techniques"
sourceId: 4986
sourceUrl: "https://apxml.com/zh/courses/deep-learning-fundamentals-keras/chapter-6-improving-model-performance-workflow/practical-applying-improvement-techniques"
title: "动手实践：应用改进技术"
description: "应用正则化、回调和数据增强来提高先前构建的模型性能。"
order: 8
plots: ["plots/4986-0.json"]
sourceHash: "86be7be9b990316be1b73cd2f548a0024bcf3e33641657cfb8724014b23059bd"
sourceCorrections: []
---

多种模型改进技术将应用于一个用于图像分类的卷积神经网络 (neural network)（CNN）。将采用多种方法来提高其性能，并使训练过程更稳定。目的不仅是获得更高的准确率，更是构建一个对未见数据泛化能力更强的模型，并改进开发流程。

我们假设有一个在CIFAR-10等数据集上训练过的基线CNN。通常，此类模型，尤其是训练了许多周期后，一个常见问题是过拟合 (overfitting)：模型在训练数据上表现良好，但在验证或测试数据上表现不佳。我们的基线模型可能会表现出这种行为。

### 设定场景

首先，让我们确定起点。假设我们的基线CNN（来自第4章）在20个周期后，在CIFAR-10验证集上达到了大约70%的准确率，但验证准确率趋于平稳甚至开始下降，而训练准确率却持续上升。这种分化是过拟合 (overfitting)的典型迹象。

我们的任务是使用Dropout、数据增强和回调来重新构建和训练此模型，以解决过拟合并控制训练过程。我们将使用Keras 3，同时考虑到其后端灵活性（我们将使用与PyTorch或TensorFlow兼容的语法）。

```python
# 假设必要的导入: keras, layers, torch (或 tensorflow)
# 假设cifar10数据集已加载并预处理为:
# x_train, y_train, x_val, y_val, x_test, y_test
# (像素值已缩放，例如到[0, 1]，标签已进行one-hot编码)

import keras
from keras import layers
# import torch # 如果使用PyTorch后端

# 示例: 为CIFAR-10定义输入形状和类别数量
input_shape = (32, 32, 3)
num_classes = 10

# 基线模型结构 (为方便说明，已从第4章简化)
# baseline_model = keras.Sequential(
#     [
#         keras.Input(shape=input_shape),
#         layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
#         layers.MaxPooling2D(pool_size=(2, 2)),
#         layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
#         layers.MaxPooling2D(pool_size=(2, 2)),
#         layers.Flatten(),
#         layers.Dense(128, activation="relu"),
#         layers.Dense(num_classes, activation="softmax"),
#     ]
# )

# baseline_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# History baseline = baseline_model.fit(x_train, y_train, batch_size=128, epochs=20, validation_data=(x_val, y_val))
# 基线评估: baseline_loss, baseline_acc = baseline_model.evaluate(x_test, y_test)
# print(f"基线测试准确率: {baseline_acc:.4f}") # 示例结果: ~0.68
```

### 应用正则化 (regularization)：Dropout

为了应对过拟合 (overfitting)，我们将引入`Dropout`层。Dropout在训练期间的每次更新中随机将一部分输入单元设置为0，这有助于防止训练数据上出现复杂的协同适应。我们将在池化层之后和最终全连接层之前添加Dropout。

```python
# 使用Dropout的改进模型
improved_model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        # 池化后添加Dropout
        layers.Dropout(0.25), # Dropout比率为25%

        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        # 池化后添加Dropout
        layers.Dropout(0.25), # Dropout比率为25%

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        # 在最终全连接层之前添加Dropout
        layers.Dropout(0.5), # 全连接层使用更高的Dropout率很常见
        layers.Dense(num_classes, activation="softmax"),
    ]
)

improved_model.summary()
```

Dropout率（0.25、0.5）是常见的起始点；它们是稍后可以调整的超参数 (parameter) (hyperparameter)。

### 实现数据增强

数据增强通过创建现有图像的修改版本来人工扩展训练数据集。这使得模型对旋转、平移或翻转等变换更具不变性，从而提高泛化能力。Keras提供了可以直接包含在模型定义中或应用于数据集管道的预处理层。将它们添加到模型中很方便。

```python
# 包含数据增强层的模型定义
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal", input_shape=input_shape),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        # 根据需要添加其他增强方式（例如 RandomContrast）
    ]
)

improved_model_with_aug = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        # 首先应用增强层
        data_augmentation,
        # 模型结构的其余部分（包含Dropout）
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

improved_model_with_aug.summary()
```

现在，模型在每个周期都会看到略微不同的训练图像版本。

### 使用回调进行高效训练

回调允许我们在训练期间自动化操作。我们将使用`ModelCheckpoint`，仅在验证性能提高时保存模型权重 (weight)；使用`EarlyStopping`，如果在设定的周期数内验证性能没有提升，则停止训练。

```python
# 定义回调
callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="best_model.keras", # 模型文件的保存路径
        save_best_only=True,        # 仅在val_loss改善时保存
        monitor="val_loss",         # 监控的指标
        verbose=1                   # 保存时打印日志
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_loss", # 监控的指标
        patience=10,        # 等待的无改进周期数
        verbose=1,          # 停止时打印日志
        restore_best_weights=True # 从val_loss最佳的周期恢复权重
    )
    # 可选: 添加TensorBoard回调
    # keras.callbacks.TensorBoard(log_dir="./logs")
]

# 编译改进后的模型（包含增强和dropout）
improved_model_with_aug.compile(loss="categorical_crossentropy",
                                optimizer="adam",
                                metrics=["accuracy"])
```

### 训练和评估改进模型

现在，我们使用`fit`方法训练增强模型，并传入我们的回调列表。初始阶段可能需要训练更多周期，因为正则化 (regularization)和数据增强有时会减缓收敛速度，但`EarlyStopping`将防止过度训练。

```python
# 训练模型
print("正在训练改进模型...")
history_improved = improved_model_with_aug.fit(
    x_train, y_train,
    batch_size=128,
    epochs=50,  # 允许更多周期，EarlyStopping将进行管理
    validation_data=(x_val, y_val),
    callbacks=callbacks # 包含我们定义的回调
)

# 注意: EarlyStopping可能会在50个周期前停止训练。
# 'best_model.keras' 文件现在包含val_loss最低的周期的权重。

# 选项1: 如果restore_best_weights=True，EarlyStopping会自动恢复最佳权重
print("\n正在评估已恢复最佳权重的模型（由于EarlyStopping）。..")
improved_loss, improved_acc = improved_model_with_aug.evaluate(x_test, y_test, verbose=0)
print(f"改进模型测试准确率（来自EarlyStopping）: {improved_acc:.4f}")

# 选项2: 显式加载ModelCheckpoint保存的最佳模型
# print("\n正在加载ModelCheckpoint保存的最佳模型并评估...")
# best_model = keras.models.load_model("best_model.keras")
# improved_loss, improved_acc = best_model.evaluate(x_test, y_test, verbose=0)
# print(f"改进模型测试准确率（来自best_model.keras）: {improved_acc:.4f}")

# 与基线进行比较（示例结果）
# 基线测试准确率: ~0.68
# 改进模型测试准确率: ~0.75 (预期提高)
```

您应该会观察到，训练准确率和验证准确率之间的差距小于基线模型，这表明过拟合 (overfitting)现象有所减少。最终的测试准确率有望高于基线模型。

### 可视化改进效果

让我们可视化训练历史，以查看其影响。我们期望改进模型的验证曲线能更紧密地跟随训练曲线，并有可能达到更高的峰值。



![模型训练历史比较](plots/4986-0.json)



> 基线模型（蓝色）和改进模型（绿色）的训练与验证准确率曲线比较，其中改进模型使用了正则化 (regularization)和数据增强。请注意，改进模型表现出更少的过拟合 (overfitting)（训练/验证曲线之间的差距更小），并获得了更好的验证准确率。提前停止可能会在第25-30个周期左右终止改进模型的训练。

### 保存最终模型

评估后，如果满意，您可以保存最终训练好的模型（得益于`ModelCheckpoint`，它可能已经保存为`best_model.keras`）。

```python
# 如果您没有使用ModelCheckpoint或想要保存当前状态
# improved_model_with_aug.save("final_improved_cnn_cifar10.keras")
# loaded_model = keras.models.load_model("final_improved_cnn_cifar10.keras")
# print("模型已成功保存和重新加载。")
```

### 后续步骤：超参数 (parameter) (hyperparameter)调整

尽管我们已经应用了多种有效技术，但要获得最佳性能通常需要调整超参数。这可能涉及尝试以下方面：

- 不同的`Dropout`率。
- `数据增强`的强度和类型。
- 优化器的学习率（`Adam`的默认学习率可能不是最佳的）。
- 网络架构的变化（滤波器数量、层数）。
- 不同的优化算法（`RMSprop`、带动量的`SGD`）。

KerasTuner或Optuna等工具和方法可以帮助自动化此搜索过程，但即使是基于训练期间观察到的验证结果（也许使用TensorBoard）进行手动尝试，也能带来明显的提升。

本次实践练习展示了如何应用dropout、数据增强和回调（`ModelCheckpoint`、`EarlyStopping`）来提高CNN的泛化能力并更有效地管理训练过程。这些方法是深度学习 (deep learning)实践者构建更可靠、性能更好模型的重要手段。

## 参考资料

- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  全面涵盖深度学习的理论基础，包括卷积网络、如Dropout等正则化技术以及防止过拟合的策略。
- [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://jmlr.org/papers/v15/srivastava14a.html) — Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, Ruslan Salakhutdinov (2014)
  Journal: Journal of Machine Learning Research; Publisher: Journal of Machine Learning Research; Volume: 15; Pages: 1929−1958; DOI: [10.5555/2627435.2670313](https://doi.org/10.5555/2627435.2670313)
  这篇开创性论文介绍了Dropout，作为一种减少神经网络过拟合的正则化技术。
- [Keras Documentation](https://keras.io/) — Keras team (2024)
  Keras深度学习API的官方文档，提供了关于层（如`Dropout`、`RandomFlip`、`RandomRotation`）、回调函数（`ModelCheckpoint`、`EarlyStopping`）和模型构建实践的详细信息。
- [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python-second-edition) — François Chollet (2021)
  Publisher: Manning Publications
  Keras创建者撰写的权威指南，提供了使用Keras构建和改进深度学习模型的实用示例和最佳实践，包括正则化和数据增强。
