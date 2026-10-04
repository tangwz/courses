---
course: "getting-started-with-tensorflow"
chapter: "saving-loading-models"
lesson: "practice-saving-restoring-training"
sourceId: 1803
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-tensorflow/chapter-6-saving-loading-models/practice-saving-restoring-training"
title: "练习：模型训练的保存与恢复"
description: "动手实践保存模型检查点并恢复它们以继续训练或用于推理。"
order: 8
plots: []
sourceHash: "588751c03b573b43017984eb17a8dda818d8fb91850ef19ed30cd04c5dabd132"
sourceCorrections: []
---

好的，让我们将本章的理念付诸实践。您已经学习了保存模型进度的不同方法，这对于任何实际的机器学习 (machine learning)工作流程都非常必要。无论是需要从中断中恢复、部署训练好的模型，还是在长时间训练中保存最佳版本，了解如何高效地保存和加载都十分重要。

以下是保存和加载模型的一些常见情形：

1. 使用 `ModelCheckpoint` 在训练期间自动保存权重 (weight)。
2. 将这些权重加载到新的模型实例中。
3. 在训练后保存整个模型。
4. 加载完整的模型用于推理 (inference)或进一步训练。

我们将使用一个简单的模型和合成数据，以便我们可以纯粹关注保存和加载的机制。

### 设置

首先，让我们导入TensorFlow和其他必要的库，并为二分类问题生成一些简单数据。

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
import shutil # 用于清理已保存的文件

print(f"Using TensorFlow version: {tf.__version__}")

# 生成合成数据
def generate_data(num_samples=1000):
    # 简单的二维特征，为简单起见可线性分离
    np.random.seed(42)
    X = np.random.rand(num_samples, 2) * 10 - 5
    # 简单的线性边界：y > 0.5*x - 1
    y = (X[:, 1] > 0.5 * X[:, 0] - 1).astype(int)
    return X, y

X_train, y_train = generate_data(1000)
X_val, y_val = generate_data(200)

# 定义一个简单的Sequential模型
def build_model():
    model = keras.Sequential(
        [
            layers.Dense(16, activation="relu", input_shape=(2,)),
            layers.Dense(8, activation="relu"),
            layers.Dense(1, activation="sigmoid"), # 二分类
        ]
    )
    model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model

# 创建用于保存模型/权重的目录
checkpoint_dir = "./training_checkpoints"
saved_model_dir = "./saved_model"

# 如果存在，清理之前的运行
if os.path.exists(checkpoint_dir):
    shutil.rmtree(checkpoint_dir)
if os.path.exists(saved_model_dir):
    shutil.rmtree(saved_model_dir)

os.makedirs(checkpoint_dir)
# saved_model_dir 将由 model.save() 创建
```

### 1. 训练期间保存检查点

`ModelCheckpoint` 回调函数在训练期间自动保存模型方面非常有用。您可以配置它只保存权重 (weight)或整个模型，并决定是每个周期都保存，还是只在性能提升时保存。在这里，我们只在验证损失改善时保存权重。

```python
model = build_model()

# 配置ModelCheckpoint回调函数
# 我们将仅基于验证损失保存权重
# 文件名包含周期数和验证损失
checkpoint_path = os.path.join(checkpoint_dir, "ckpt_epoch_{epoch:02d}_val_loss_{val_loss:.2f}.weights.h5")

checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,  # 只保存模型的权重
    monitor='val_loss',      # 监控验证损失
    mode='min',              # 当验证损失减少时保存
    save_best_only=True,     # 只保存目前为止的“最佳”模型
    verbose=1                # 保存时打印消息
)

print("Starting training with ModelCheckpoint callback...")
history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[checkpoint_callback],
    verbose=0 # 设置为0以避免输出混乱，回调函数中的verbose=1会显示保存信息
)

print("\nTraining finished.")
print(f"Checkpoints saved in: {checkpoint_dir}")
print("Files:", os.listdir(checkpoint_dir))

# 找到最新的检查点（由于save_best_only=True，这应该是最好的一个）
latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
print(f"\nLatest (best) checkpoint found: {latest_checkpoint}")
```

您应该会看到输出显示在验证损失改善时检查点已保存。`tf.train.latest_checkpoint` 工具可以帮助找到目录中最近保存的检查点文件的路径，在我们的案例中，它对应于性能最佳的模型，因为我们设置了 `save_best_only=True`。

### 2. 从检查点加载权重 (weight)

现在，想象一下您的训练中断了，或者您只是想使用您保存的最佳权重。您需要：

1. 创建一个具有 *完全相同架构* 的 *新* 模型实例。
2. 使用 `model.load_weights()` 将已保存的权重加载到这个新的模型实例中。

```python
# 构建一个具有相同架构的新的、未经训练的模型实例
new_model = build_model()

# 评估未经训练的模型（性能应该很差）
print("\nEvaluating the new, untrained model:")
loss_untrained, acc_untrained = new_model.evaluate(X_val, y_val, verbose=0)
print(f"Untrained model - Loss: {loss_untrained:.4f}, Accuracy: {acc_untrained:.4f}")

# 从之前保存的最佳检查点加载权重
if latest_checkpoint:
    print(f"\nLoading weights from: {latest_checkpoint}")
    new_model.load_weights(latest_checkpoint)

    # 评估加载权重后的模型（性能应该很好）
    print("Evaluating the model with loaded weights:")
    loss_loaded, acc_loaded = new_model.evaluate(X_val, y_val, verbose=0)
    print(f"Model with loaded weights - Loss: {loss_loaded:.4f}, Accuracy: {acc_loaded:.4f}")
else:
    print("\nNo checkpoint found to load.")
```

请注意，与刚初始化的 `new_model` 相比，加载权重后准确率有了明显的提升。这确认学习到的参数 (parameter)已成功恢复。请记住，`load_weights` 只恢复参数；它不恢复优化器的状态。

### 3. 保存整个模型

仅保存权重 (weight)很有用，但有时您需要整个包：架构、权重和优化器的状态（例如，准确地从上次中断的地方恢复训练）。`model.save()` 方法处理此问题，使用 TensorFlow `SavedModel` 格式将所有内容保存到目录中。

```python
# 假设“model”是步骤1中训练过的模型
# 或者我们可以使用已加载权重的“new_model”

print(f"\nSaving the entire model to: {saved_model_dir}")
model.save(saved_model_dir) # 使用最初训练过的模型实例

print("模型保存成功。")
print("已保存模型目录的内容：")
# 列出内容以显示SavedModel结构
for item in os.listdir(saved_model_dir):
    print(f"- {item}")
```

执行 `model.save()` 会创建一个目录，其中包含 `saved_model.pb`（图定义和元数据）、`variables` 目录（包含权重），以及可能还有一个 `assets` 目录。这种格式是语言无关的，适用于通过 TensorFlow Serving 提供模型服务或在其他 TensorFlow 环境中使用它们。

### 4. 加载整个模型

使用 `tf.keras.models.load_model()` 加载 `SavedModel` 简单直接。这会恢复架构、权重 (weight)和优化器状态，使模型准备好进行推理 (inference)或继续训练。

```python
print(f"\nLoading the entire model from: {saved_model_dir}")
loaded_full_model = tf.keras.models.load_model(saved_model_dir)

# 验证加载模型的架构
print("\nLoaded model summary:")
loaded_full_model.summary()

# 评估加载的模型以确认其表现符合预期
print("\nEvaluating the loaded full model:")
loss_full, acc_full = loaded_full_model.evaluate(X_val, y_val, verbose=0)
print(f"Loaded full model - Loss: {loss_full:.4f}, Accuracy: {acc_full:.4f}")

# 您也可以直接进行预测
print("\nMaking a prediction with the loaded model:")
sample_prediction = loaded_full_model.predict(X_val[:5]) # 对前5个验证样本进行预测
print("预测:", sample_prediction.flatten())
print("实际标签:", y_val[:5])
```

加载的模型表现与我们保存的模型一致，我们不需要重新构建架构或再次编译它（尽管如果您想为进一步训练更改优化器或指标，可能需要重新编译）。

### 恢复训练（可选）

由于 `model.save()` 也保存了优化器的状态，您可以恢复训练。TensorFlow 将从中断的地方继续，包括学习率调度和其他优化器参数 (parameter)，如动量。

```python
# 对加载的模型恢复训练几个周期
print("\nResuming training on the loaded model...")
history_resumed = loaded_full_model.fit(
    X_train,
    y_train,
    epochs=5, # 再训练5个周期
    initial_epoch=history.epoch[-1] + 1, # 正确地开始周期编号
    batch_size=32,
    validation_data=(X_val, y_val),
    verbose=1
)

print("\nResumed training finished.")
```

这展示了加载完整的 `SavedModel` 如何让您精确地继续训练过程，这对于长期运行的实验非常有价值。

### 总结

在本次练习中，您实践了TensorFlow/Keras中保存和加载模型的必要工作流程：

- **`ModelCheckpoint` 回调函数：** 非常适合在训练过程中自动保存最佳权重 (weight)（或完整模型），提供容错能力并捕获最佳状态。
- **`model.load_weights()`：** 用于将学习到的参数 (parameter)恢复到具有相同架构的模型实例中。当您只需要权重时很有用，例如在迁移学习 (transfer learning)或您自行重建模型结构时的推理 (inference)。
- **`model.save()`：** 以 `SavedModel` 格式保存整个模型（架构、权重、优化器状态）。这是保存模型以供部署或稍后恢复训练的标准方法。
- **`tf.keras.models.load_model()`：** 加载之前使用 `model.save()` 保存的模型，恢复其完整状态。

掌握这些技术可确保您的训练成果得到保留，并且您的模型可以用于评估、部署或进一步开发。

## 参考资料

- [Save and load Keras models](https://www.tensorflow.org/guide/keras/save_and_serialize) — Neel Kovelamudi, Francois Chollet (2024)
  Publisher: TensorFlow (Google)
  详细解释了Keras模型保存机制，包括仅保存权重和使用`SavedModel`格式保存完整模型，以及`ModelCheckpoint`回调。
- [tf.keras.callbacks.ModelCheckpoint](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint) — TensorFlow Authors (2024)
  `tf.keras.callbacks.ModelCheckpoint`回调的官方API参考，详细说明了训练期间自动保存的所有参数和用法。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://books.google.com/books/about/Hands_On_Machine_Learning_with_Scikit_Le.html?id=R-F6EAAAQBAJ) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  提供了关于各种机器学习主题的详细指导，包括Keras和TensorFlow中有效的模型保存和加载策略。
