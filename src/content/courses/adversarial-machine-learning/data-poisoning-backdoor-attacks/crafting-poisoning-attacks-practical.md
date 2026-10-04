---
course: "adversarial-machine-learning"
chapter: "data-poisoning-backdoor-attacks"
lesson: "crafting-poisoning-attacks-practical"
sourceId: 4149
sourceUrl: "https://apxml.com/zh/courses/adversarial-machine-learning/chapter-3-data-poisoning-backdoor-attacks/crafting-poisoning-attacks-practical"
title: "构建数据投毒攻击：动手实践"
description: "练习创建针对机器学习模型的数据投毒和简单后门攻击。"
order: 6
plots: []
sourceHash: "2e641707deafe4d342d6ac295b21c363ed393a84b717490ad047862e157a8c2e"
sourceCorrections: []
---

数据投毒和后门攻击使用Python和常用机器学习 (machine learning)库实现。实际例子包括降低模型性能的基础投毒、一次定向攻击以及一个简单后门的实现。

"请记住，这些例子是示例性的。投毒通常需要更复杂的优化方法来制作不明显且有效的投毒数据点，特别是对于复杂模型和数据集。然而，这些基础例子体现了主要原理。"

我们将使用Scikit-learn为简便起见说明这些原理，侧重于数据操作方面而非复杂的模型架构。这些原理适用于深度学习 (deep learning)模型，通常使用ART（对抗鲁棒性工具箱）等框架实现，这些框架提供了更专业的工具。

### 环境设置

首先，请确保您已安装必要的库。我们主要使用 `numpy` 进行数值运算，使用 `scikit-learn` 处理数据集和模型。

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt # 可选，用于可视化
```

让我们为初始实验生成一个简单的合成数据集。这使我们能够方便地控制情景。

```python
# 生成一个二分类数据集
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=42)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"训练集大小: {X_train.shape[0]}")
print(f"测试集大小: {X_test.shape[0]}")
```

### 场景一：基础可用性投毒（标签翻转）

我们的第一个目标很简单：降低在数据上训练的模型的整体准确率。在训练阶段，最直接的方法是注入错误标记 (token)的样本。这通常称为标签翻转。

**目标：** 降低模型在干净测试数据上的表现。

**方法：** 随机选择一部分训练数据并翻转其标签。

```python
# --- 投毒设置 ---
poison_percentage = 0.1 # 对10%的训练数据投毒
n_poison = int(poison_percentage * len(X_train))
print(f"注入 {n_poison} 个投毒样本。")

# 选择随机索引进行投毒
poison_indices = np.random.choice(len(X_train), size=n_poison, replace=False)

# 创建训练数据的副本以进行修改
X_train_poisoned = np.copy(X_train)
y_train_poisoned = np.copy(y_train)

# 翻转所选索引的标签
# 对于二分类，将0翻转为1，将1翻转为0
y_train_poisoned[poison_indices] = 1 - y_train[poison_indices]

# --- 训练和评估 ---

# 在干净数据上训练模型
model_clean = LogisticRegression(random_state=42, max_iter=1000)
model_clean.fit(X_train, y_train)
y_pred_clean = model_clean.predict(X_test)
acc_clean = accuracy_score(y_test, y_pred_clean)
print(f"干净数据上的准确率: {acc_clean:.4f}")

# 在投毒数据上训练模型
model_poisoned = LogisticRegression(random_state=42, max_iter=1000)
model_poisoned.fit(X_train_poisoned, y_train_poisoned)
y_pred_poisoned = model_poisoned.predict(X_test)
acc_poisoned = accuracy_score(y_test, y_pred_poisoned)
print(f"标签翻转后准确率 ({poison_percentage*100}%): {acc_poisoned:.4f}")
```

您应该会观察到在投毒数据集上训练的模型准确率有明显的下降。下降的幅度取决于投毒比例、数据集复杂度和模型容量。这种简单的攻击通过提供不正确的监督信号，直接与学习目标冲突。

### 场景二：定向投毒攻击

现在，让我们尝试一种更具针对性的攻击。我们不只是降低整体性能，而是希望模型错误分类一个*特定*实例或某种类型的实例。这是一种完整性攻击。构建最优的定向投毒样本很复杂，通常涉及优化以找到能最大程度影响目标附近决策边界的点。在这里，我们将模拟一个更简单的版本。

**目标：** 使一个特定的干净测试样本在投毒数据训练后被错误分类。

**方法：** 确定一个目标测试样本。通过稍微修改来自*不同*类别的训练样本副本，并将其标记 (token)为*目标真实类别*，来创建投毒点。其思路是让决策边界在目标样本附近产生难以察觉的偏移。

```python
# --- 目标选择 ---
# 选择一个特定的测试实例作为目标
target_index = 0
X_target = X_test[target_index].reshape(1, -1)
y_target_true = y_test[target_index]
print(f"目标实例索引: {target_index}, 真实标签: {y_target_true}")

# 检查其通过干净模型进行的分类（理想情况下应正确）
y_target_pred_clean = model_clean.predict(X_target)
print(f"目标预测（干净模型）: {y_target_pred_clean[0]}")

# 如果已被干净模型错误分类，则选择另一个目标以便更清晰地演示
if y_target_pred_clean[0] != y_target_true:
    print("目标已被干净模型错误分类。请选择另一个或谨慎操作。")
    # 找到一个正确分类的目标
    for i in range(len(X_test)):
        target_index = i
        X_target = X_test[target_index].reshape(1, -1)
        y_target_true = y_test[target_index]
        y_target_pred_clean = model_clean.predict(X_target)
        if y_target_pred_clean[0] == y_target_true:
            print(f"新目标索引: {target_index}, 真实标签: {y_target_true}")
            print(f"目标预测（干净模型）: {y_target_pred_clean[0]}")
            break

# --- 制作投毒点 ---
n_poison_targeted = 5 # 要制作的投毒点数量
target_class_label = y_target_true
source_class_label = 1 - target_class_label

# 从“源”类别（我们*不*希望目标归属的类别）中查找训练样本
source_indices = np.where(y_train == source_class_label)[0]

# 随机选择少量源样本
crafting_indices = np.random.choice(source_indices, size=n_poison_targeted, replace=False)

# 创建投毒点：稍微扰动源样本并将其标记为目标类别
# 这是一种启发式方法。更高级的方法会优化扰动。
X_poison_crafted = []
y_poison_crafted = []
perturbation_scale = 0.1 # 小扰动

for idx in crafting_indices:
    X_source_sample = X_train[idx]
    # 简单扰动：向目标添加少量噪声（或仅添加随机噪声）
    perturbation = (X_target.flatten() - X_source_sample) * perturbation_scale + (np.random.rand(X_source_sample.shape[0]) - 0.5) * 0.05
    X_p = X_source_sample + perturbation
    X_poison_crafted.append(X_p)
    y_poison_crafted.append(target_class_label) # 标记为目标真实类别

X_poison_crafted = np.array(X_poison_crafted)
y_poison_crafted = np.array(y_poison_crafted)

# --- 使用定向投毒进行训练 ---
# 将制作的投毒样本添加到原始训练数据中
X_train_targeted_poison = np.vstack((X_train, X_poison_crafted))
y_train_targeted_poison = np.hstack((y_train, y_poison_crafted))

# 在此特定投毒数据上训练模型
model_targeted_poison = LogisticRegression(random_state=42, max_iter=1000)
model_targeted_poison.fit(X_train_targeted_poison, y_train_targeted_poison)

# --- 评估 ---
# 检查整体准确率（可能不会下降太多）
y_pred_targeted_overall = model_targeted_poison.predict(X_test)
acc_targeted_overall = accuracy_score(y_test, y_pred_targeted_overall)
print(f"整体准确率（定向投毒）: {acc_targeted_overall:.4f}")

# 检查特定目标实例的预测
y_target_pred_poisoned = model_targeted_poison.predict(X_target)
print(f"目标预测（投毒模型）: {y_target_pred_poisoned[0]} (真实标签: {y_target_true})")

if y_target_pred_poisoned[0] != y_target_true:
    print("定向投毒成功：目标实例被错误分类。")
else:
    print("定向投毒失败：目标实例仍被正确分类。")
```

在这种情景下，整体准确率可能保持相对较高，但错误分类所选目标实例的特定目标可能实现。这说明了完整性攻击比简单可用性攻击更隐蔽的性质。成功在很大程度上取决于投毒制作策略、投毒样本数量和模型的学习动态。

### 场景三：简单后门攻击（模式触发器）

后门攻击植入一个隐藏触发器。模型在干净数据上正常工作，但当输入中出现触发器模式时会行为异常。让我们使用MNIST数据集来模拟此过程，因为可视触发器很直观。

**方法：**

1. 加载MNIST数据集。
2. 定义一个触发器模式（例如，角落里的一小块亮像素）。
3. 从不同类别中选择一部分训练图像。
4. 将触发器模式应用于这些选定的图像。
5. 将这些带有触发器的图像的标签更改为目标类别（例如，类别“7”）。
6. 将这些后门样本添加到原始训练集中。
7. 训练一个模型（例如，一个简单的CNN，或者为了演示目的，甚至是扁平图像上的逻辑回归）。
8. 评估在干净测试数据上的准确率。
9. 评估“攻击成功率”：在*应用了触发器*的测试图像上的准确率，期望它们被分类为目标类别。

```python
# 我们需要tensorflow/keras来处理MNIST和简单模型
# 或者，可以使用scikit-learn的fetch_openml('mnist_784')
try:
    import tensorflow as tf
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Flatten, Input
    from tensorflow.keras.utils import to_categorical
    USE_TF = True
except ImportError:
    print("未找到TensorFlow。跳过后门示例（或使用Scikit-learn的MNIST进行调整）。")
    USE_TF = False

if USE_TF:
    # --- 加载MNIST数据 ---
    (X_train_mnist, y_train_mnist), (X_test_mnist, y_test_mnist) = mnist.load_data()

    # 将像素值归一化到[0, 1]
    X_train_mnist = X_train_mnist.astype('float32') / 255.0
    X_test_mnist = X_test_mnist.astype('float32') / 255.0

    # 展平图像以用于简单全连接模型（或使用CNN）
    X_train_flat = X_train_mnist.reshape((X_train_mnist.shape[0], -1))
    X_test_flat = X_test_mnist.reshape((X_test_mnist.shape[0], -1))

    # 进行独热编码
    y_train_cat = to_categorical(y_train_mnist, 10)
    y_test_cat = to_categorical(y_test_mnist, 10)

    # --- 后门设置 ---
    target_class = 7
    target_class_cat = to_categorical([target_class], 10)[0]
    trigger_size = 3 # 3x3像素触发器
    trigger_pos = (24, 24) # 右下角
    trigger_value = 1.0 # 白色像素

    def apply_trigger(images):
        images_triggered = np.copy(images)
        x, y = trigger_pos
        # 在展平之前直接在2D图像形状上应用触发器
        images_2d = images_triggered.reshape((-1, 28, 28))
        images_2d[:, x:x+trigger_size, y:y+trigger_size] = trigger_value
        return images_2d.reshape((-1, 28*28)) # 返回展平后的

    # --- 创建带后门训练样本 ---
    backdoor_percentage = 0.05 # 使用5%的数据进行后门植入
    n_backdoor_samples = int(backdoor_percentage * len(X_train_flat))

    # 选择随机样本注入后门（选择最初不属于目标类别的样本）
    potential_indices = np.where(y_train_mnist != target_class)[0]
    backdoor_indices = np.random.choice(potential_indices, size=n_backdoor_samples, replace=False)

    X_backdoor = X_train_flat[backdoor_indices]
    # 应用触发器
    X_backdoor_triggered = apply_trigger(X_backdoor.reshape(-1, 28, 28)).reshape(-1, 784) # 在2D上应用，然后重新展平
    # 将标签设置为目标类别
    y_backdoor_target = np.array([target_class_cat] * n_backdoor_samples)

    # --- 合并数据集 ---
    X_train_backdoored = np.vstack((X_train_flat, X_backdoor_triggered))
    y_train_backdoored = np.vstack((y_train_cat, y_backdoor_target))

    # 打乱组合后的数据集
    shuffle_idx = np.random.permutation(len(X_train_backdoored))
    X_train_backdoored = X_train_backdoored[shuffle_idx]
    y_train_backdoored = y_train_backdoored[shuffle_idx]

    # --- 训练带后门模型 ---
    model_backdoor = Sequential([
        Input(shape=(784,)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model_backdoor.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print("正在训练带后门模型...")
    model_backdoor.fit(X_train_backdoored, y_train_backdoored, epochs=5, batch_size=128, verbose=1, validation_split=0.1)

    # --- 评估 ---
    # 1. 在干净测试数据上的准确率
    loss_clean, acc_clean_mnist = model_backdoor.evaluate(X_test_flat, y_test_cat, verbose=0)
    print(f"在干净MNIST测试数据上的准确率: {acc_clean_mnist:.4f}")

    # 2. 攻击成功率（ASR）
    # 对最初不属于目标类别的测试图像应用触发器
    test_indices_not_target = np.where(y_test_mnist != target_class)[0]
    X_test_attack = X_test_flat[test_indices_not_target]
    y_test_attack_orig = y_test_cat[test_indices_not_target] # 原始标签（独热编码）

    # 应用触发器
    X_test_attack_triggered = apply_trigger(X_test_attack.reshape(-1, 28, 28)).reshape(-1, 784)

    # 对带有触发器的图像进行预测
    y_pred_triggered = model_backdoor.predict(X_test_attack_triggered)
    predicted_classes_triggered = np.argmax(y_pred_triggered, axis=1)

    # 计算ASR：被分类为目标类别的带有触发器的图像百分比
    asr = np.mean(predicted_classes_triggered == target_class)
    print(f"带有触发器的图像的攻击成功率（ASR）: {asr:.4f}")
```

如果攻击成功，您将看到：

1. 在干净测试集上具有高准确率，表明模型很好地学习了主要任务。
2. 高攻击成功率（ASR），表明大多数测试图像在添加触发器模式后，现在被错误分类为目标类别（在我们例子中是类别“7”）。

这说明了后门的隐蔽性：模型在正常测试下表现良好，但包含可被攻击者利用的隐藏弱点。

### 总结与后续步骤

这些动手实践例子提供了一个理解数据投毒和后门攻击如何实现的起点。我们已经看到：

- **标签翻转：** 一种降低整体性能的简单可用性攻击。
- **定向投毒（启发式）：** 一种旨在错误分类特定实例的完整性攻击，通常需要更细致的制作。
- **模式后门：** 在训练期间植入触发器，激活时导致定向的异常行为。

本次实践练习的重要学习点：

- 训练时攻击操纵学习过程本身。
- 投毒可以针对整体可用性或特定完整性。
- 后门产生隐藏的条件性弱点。
- 即使是简单的实现也能有效地说明主要原理。

请记住，攻击者通常采用更高级的方法，包括基于优化的投毒生成和视觉上不明显的触发器（尤其对于非图像数据）。干净标签攻击，即投毒数据看起来仍然被正确标记 (token)，带来了更大的挑战。ART（对抗鲁棒性工具箱）等框架包含了许多复杂的投毒和后门攻击的实现，以及防御措施，这些是此方面进一步研究和工作的重要工具。后续章节将讨论如何评估针对此类威胁的鲁棒性，并研究防御机制。

## 参考资料

- [Poisoning Attacks Against Online Learning](https://link.springer.com/chapter/10.1007/978-3-642-33460-3_11) — Battista Biggio, Igino Corona, Davide Maiorca, Blaine Nelson, Nedim Šrndić, Pavel Laskov, Giorgio Giacinto, Fabio Roli (2012)
  Journal: European Conference on Machine Learning and Knowledge Discovery in Databases; Publisher: Springer; Volume: 7523; Pages: 147-162; DOI: [10.1007/978-3-642-33460-3_11](https://doi.org/10.1007/978-3-642-33460-3_11)
  介绍了数据投毒攻击的早期工作，特别是针对在线学习算法的攻击，展示了如何通过数据操纵降低模型性能。
- [Poisoning Attacks on Machine Learning: A Survey](https://link.springer.com/article/10.1007/s11633-018-1153-6) — Ningning Ma, Zewei Li, Yalong Sun, Runfang Li (2019)
  Journal: International Journal of Automation and Computing; Publisher: Springer; Volume: 16; Pages: 119-138; DOI: [10.1007/s11633-018-1153-6](https://doi.org/10.1007/s11633-018-1153-6)
  一份关于各种数据投毒攻击策略的全面综述，涵盖可用性、完整性和目标攻击，对该领域提供了广阔的视角。
- [Adversarial Robustness Toolbox (ART) Documentation](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/) — IBM Research (2023)
  对抗性鲁棒性工具箱 (ART) 的官方文档，提供了实现和防御对抗性攻击的实用指南和API参考。
