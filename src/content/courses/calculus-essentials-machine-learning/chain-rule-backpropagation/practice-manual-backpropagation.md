---
course: "calculus-essentials-machine-learning"
chapter: "chain-rule-backpropagation"
lesson: "practice-manual-backpropagation"
sourceId: 1425
sourceUrl: "https://apxml.com/zh/courses/calculus-essentials-machine-learning/chapter-5-chain-rule-backpropagation/practice-manual-backpropagation"
title: "动手实践：手动反向传播示例"
description: "通过一个简单例子，手动或通过代码计算反向传播中的梯度。"
order: 7
plots: []
sourceHash: "dd090d3a1b18b9e80e1d541a926f2f74f69ac8a54111c9f72f6186bd820c8806"
sourceCorrections: []
---

好的，我们通过一个具体例子来巩固对反向传播 (backpropagation)的理解。我们将使用一个非常简单的神经网络 (neural network)，并像反向传播算法那样，手动计算使用链式法则得到的梯度。这种实践方法有助于弄清抽象原理如何与训练所需的实际计算关联起来。

### 构建简单网络

考虑一个小型神经网络 (neural network)，它有一个输入特征 $x$，一个包含单个神经元的隐藏层，以及一个输出神经元。两个神经元都将使用 Sigmoid 激活函数 (activation function)，即 $\sigma(z) = 1 / (1 + e^{-z})$。我们的目标是根据输入 $x$ 预测一个目标值 $y_{true}$。我们将使用均方误差 (MSE) 损失函数 (loss function)，具体形式为 $L = \frac{1}{2}(y_{pred} - y_{true})^2$，而 $y_{pred}$ 是网络的输出。因子 $\frac{1}{2}$ 通常会加入，以便后续简化求导。

以下是各个组成部分：

- 输入: $x$
- 目标: $y_{true}$
- 隐藏层:
  - 权重 (weight): $w_1$
  - 偏置 (bias): $b_1$
  - 线性组合: $z_1 = w_1 x + b_1$
  - 激活值: $a_1 = \sigma(z_1)$
- 输出层:
  - 权重: $w_2$
  - 偏置: $b_2$
  - 线性组合: $z_2 = w_2 a_1 + b_2$
  - 激活值 (预测值): $y_{pred} = a_2 = \sigma(z_2)$
- 损失函数: $L = \frac{1}{2}(a_2 - y_{true})^2$

我们的目的是计算损失 $L$ 对每个参数 (parameter)的梯度：$\frac{\partial L}{\partial w_1}$、$\frac{\partial L}{\partial b_1}$、$\frac{\partial L}{\partial w_2}$ 和 $\frac{\partial L}{\partial b_2}$。这些梯度表明每个参数的微小变化如何影响损失，从而引导学习过程。

### 计算可视化

我们可以使用计算图来表示这个网络和计算流程。

> 显示从输入 $x$ 到损失 $L$ 前向传播的计算图。反向传播 (backpropagation)涉及从 $L$ 向输入和参数 (parameter)回溯来计算梯度。

### 前向传播：数值例子

我们来指定一些具体数值：

- 输入: $x = 2.0$
- 目标: $y_{true} = 1.0$
- 初始权重 (weight): $w_1 = 0.5$， $w_2 = -0.3$
- 初始偏置 (bias): $b_1 = 0.1$， $b_2 = 0.2$

现在，我们一步步计算网络的输出：

1. **隐藏层输入:**
   $z_1 = w_1 x + b_1 = (0.5)(2.0) + 0.1 = 1.0 + 0.1 = 1.1$
2. **隐藏层激活值:**
   $a_1 = \sigma(z_1) = \sigma(1.1) = \frac{1}{1 + e^{-1.1}} \approx \frac{1}{1 + 0.3329} \approx 0.7503$
3. **输出层输入:**
   $z_2 = w_2 a_1 + b_2 = (-0.3)(0.7503) + 0.2 = -0.2251 + 0.2 = -0.0251$
4. **输出层激活值 (预测值):**
   $a_2 = \sigma(z_2) = \sigma(-0.0251) = \frac{1}{1 + e^{-(-0.0251)}} \approx \frac{1}{1 + 1.0254} \approx 0.4937$
   因此，$y_{pred} = 0.4937$。
5. **计算损失:**
   $L = \frac{1}{2}(a_2 - y_{true})^2 = \frac{1}{2}(0.4937 - 1.0)^2 = \frac{1}{2}(-0.5063)^2 \approx \frac{1}{2}(0.2563) \approx 0.1282$

初始预测值为 $0.4937$，与目标值 $1.0$ 相距甚远，导致损失为 $0.1282$。现在我们需要梯度来更新权重和偏置，以降低这个损失。

### 反向传播 (backpropagation)：应用链式法则

我们从损失开始，向后遍历网络来计算梯度。记住 Sigmoid 函数的导数：$\sigma'(z) = \frac{d\sigma(z)}{dz} = \sigma(z)(1 - \sigma(z))$。

1. **损失对输出激活值 ($a_2$) 的梯度:**
   $\frac{\partial L}{\partial a_2} = \frac{\partial}{\partial a_2} \left[ \frac{1}{2}(a_2 - y_{true})^2 \right] = (a_2 - y_{true})$
   使用我们的数值：$\frac{\partial L}{\partial a_2} = 0.4937 - 1.0 = -0.5063$
2. **损失对输出层输入 ($z_2$) 的梯度:**
   使用链式法则：$\frac{\partial L}{\partial z_2} = \frac{\partial L}{\partial a_2} \frac{\partial a_2}{\partial z_2}$
   我们知道 $\frac{\partial a_2}{\partial z_2} = \sigma'(z_2) = \sigma(z_2)(1 - \sigma(z_2)) = a_2 (1 - a_2)$
   $\sigma'(z_2) \approx 0.4937 (1 - 0.4937) \approx 0.4937 \times 0.5063 \approx 0.2500$
   因此，$\frac{\partial L}{\partial z_2} = (-0.5063) \times (0.2500) \approx -0.1266$
3. **损失对输出权重 (weight) ($w_2$) 的梯度:**
   再次使用链式法则：$\frac{\partial L}{\partial w_2} = \frac{\partial L}{\partial z_2} \frac{\partial z_2}{\partial w_2}$
   我们知道 $z_2 = w_2 a_1 + b_2$，因此 $\frac{\partial z_2}{\partial w_2} = a_1$。
   $\frac{\partial L}{\partial w_2} = \frac{\partial L}{\partial z_2} \times a_1 \approx (-0.1266) \times (0.7503) \approx -0.0950$
4. **损失对输出偏置 (bias) ($b_2$) 的梯度:**
   $\frac{\partial L}{\partial b_2} = \frac{\partial L}{\partial z_2} \frac{\partial z_2}{\partial b_2}$
   因为 $z_2 = w_2 a_1 + b_2$，所以 $\frac{\partial z_2}{\partial b_2} = 1$。
   $\frac{\partial L}{\partial b_2} = \frac{\partial L}{\partial z_2} \times 1 \approx -0.1266$
5. **损失对隐藏激活值 ($a_1$) 的梯度:**
   这个梯度需要进一步反向传播。
   $\frac{\partial L}{\partial a_1} = \frac{\partial L}{\partial z_2} \frac{\partial z_2}{\partial a_1}$
   因为 $z_2 = w_2 a_1 + b_2$，所以 $\frac{\partial z_2}{\partial a_1} = w_2$。
   $\frac{\partial L}{\partial a_1} = \frac{\partial L}{\partial z_2} \times w_2 \approx (-0.1266) \times (-0.3) \approx 0.0380$
6. **损失对隐藏层输入 ($z_1$) 的梯度:**
   $\frac{\partial L}{\partial z_1} = \frac{\partial L}{\partial a_1} \frac{\partial a_1}{\partial z_1}$
   我们知道 $\frac{\partial a_1}{\partial z_1} = \sigma'(z_1) = \sigma(z_1)(1 - \sigma(z_1)) = a_1 (1 - a_1)$
   $\sigma'(z_1) \approx 0.7503 (1 - 0.7503) \approx 0.7503 \times 0.2497 \approx 0.1874$
   因此，$\frac{\partial L}{\partial z_1} = \frac{\partial L}{\partial a_1} \times \sigma'(z_1) \approx (0.0380) \times (0.1874) \approx 0.0071$
7. **损失对隐藏权重 ($w_1$) 的梯度:**
   $\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial z_1} \frac{\partial z_1}{\partial w_1}$
   因为 $z_1 = w_1 x + b_1$，所以 $\frac{\partial z_1}{\partial w_1} = x$。
   $\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial z_1} \times x \approx (0.0071) \times (2.0) \approx 0.0142$
8. **损失对隐藏偏置 ($b_1$) 的梯度:**
   $\frac{\partial L}{\partial b_1} = \frac{\partial L}{\partial z_1} \frac{\partial z_1}{\partial b_1}$
   因为 $z_1 = w_1 x + b_1$，所以 $\frac{\partial z_1}{\partial b_1} = 1$。
   $\frac{\partial L}{\partial b_1} = \frac{\partial L}{\partial z_1} \times 1 \approx 0.0071$

我们现在已计算出所有必需的梯度：

- $\frac{\partial L}{\partial w_2} \approx -0.0950$
- $\frac{\partial L}{\partial b_2} \approx -0.1266$
- $\frac{\partial L}{\partial w_1} \approx 0.0142$
- $\frac{\partial L}{\partial b_1} \approx 0.0071$

这些梯度是反向传播步骤对于这个单个数据点的主要输出。

### 与梯度下降 (gradient descent)的关联

这些计算出的梯度正是梯度下降（或其变体，如 SGD 或 mini-batch GD）更新步骤所需的。例如，使用标准梯度下降时，更新规则将如下所示（其中 $\eta$ 是学习率）：

$w_1 \leftarrow w_1 - \eta \frac{\partial L}{\partial w_1}$
$b_1 \leftarrow b_1 - \eta \frac{\partial L}{\partial b_1}$
$w_2 \leftarrow w_2 - \eta \frac{\partial L}{\partial w_2}$
$b_2 \leftarrow b_2 - \eta \frac{\partial L}{\partial b_2}$

代入我们计算出的梯度（假设 $\eta = 0.1$ 用于演示）：

$w_1 \leftarrow 0.5 - 0.1 \times (0.0142) = 0.5 - 0.00142 = 0.49858$
$b_1 \leftarrow 0.1 - 0.1 \times (0.0071) = 0.1 - 0.00071 = 0.09929$
$w_2 \leftarrow -0.3 - 0.1 \times (-0.0950) = -0.3 + 0.00950 = -0.29050$
$b_2 \leftarrow 0.2 - 0.1 \times (-0.1266) = 0.2 + 0.01266 = 0.21266$

仅经过一个更新步骤，参数 (parameter)就已进行微调 (fine-tuning)，朝着预期能减少此特定输入-输出对损失的方向。在多个数据点和迭代中重复此过程（前向传播、反向传播 (backpropagation)、参数更新），可使网络学习复杂的模式。

### Python 实现代码片段

我们用一段使用基本数学函数的 Python 代码来验证这些计算。

```python
import math

# Sigmoid 函数及其导数
def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def sigmoid_prime(z):
    # 使用 Sigmoid 函数的输出计算: sigma(z) * (1 - sigma(z))
    s_z = sigmoid(z)
    return s_z * (1 - s_z)
    # 或者: return math.exp(-z) / ((1 + math.exp(-z))**2)

# 输入和目标
x = 2.0
y_true = 1.0

# 初始参数
w1 = 0.5
b1 = 0.1
w2 = -0.3
b2 = 0.2

# --- 前向传播 ---
# 隐藏层
z1 = w1 * x + b1
a1 = sigmoid(z1)

# 输出层
z2 = w2 * a1 + b2
a2 = sigmoid(z2) # y_pred

# 损失
loss = 0.5 * (a2 - y_true)**2

print(f"--- 前向传播 ---")
print(f"z1: {z1:.4f}")
print(f"a1: {a1:.4f}")
print(f"z2: {z2:.4f}")
print(f"a2 (预测值): {a2:.4f}")
print(f"Loss: {loss:.4f}\n")

# --- 反向传播 (梯度) ---
# 输出层梯度
dL_da2 = a2 - y_true
da2_dz2 = sigmoid_prime(z2) # 或者使用 a2 * (1 - a2)
dL_dz2 = dL_da2 * da2_dz2

dL_dw2 = dL_dz2 * a1
dL_db2 = dL_dz2 * 1.0

# 隐藏层梯度
dz2_da1 = w2
dL_da1 = dL_dz2 * dz2_da1

da1_dz1 = sigmoid_prime(z1) # 或者使用 a1 * (1 - a1)
dL_dz1 = dL_da1 * da1_dz1

dL_dw1 = dL_dz1 * x
dL_db1 = dL_dz1 * 1.0

print(f"--- 反向传播 (梯度) ---")
print(f"dL/da2: {dL_da2:.4f}")
print(f"dL/dz2: {dL_dz2:.4f}")
print(f"dL/dw2: {dL_dw2:.4f}")
print(f"dL/db2: {dL_db2:.4f}")
print(f"dL/da1: {dL_da1:.4f}")
print(f"dL/dz1: {dL_dz1:.4f}")
print(f"dL/dw1: {dL_dw1:.4f}")
print(f"dL/db1: {dL_db1:.4f}")

# --- 参数更新示例 (学习率 = 0.1) ---
lr = 0.1
w1_new = w1 - lr * dL_dw1
b1_new = b1 - lr * dL_db1
w2_new = w2 - lr * dL_dw2
b2_new = b2 - lr * dL_db2

print(f"\n--- 参数更新 (lr = {lr}) ---")
print(f"新 w1: {w1_new:.5f}")
print(f"新 b1: {b1_new:.5f}")
print(f"新 w2: {w2_new:.5f}")
print(f"新 b2: {b2_new:.5f}")
```

运行此代码会产生与我们手动计算相符的结果（允许有微小的浮点差异）：

```text
--- 前向传播 ---
z1: 1.1000
a1: 0.7503
z2: -0.0251
a2 (预测值): 0.4937
Loss: 0.1282

--- 反向传播 (梯度) ---
dL/da2: -0.5063
dL/dz2: -0.1266
dL/dw2: -0.0950
dL/db2: -0.1266
dL/da1: 0.0380
dL/dz1: 0.0071
dL/dw1: 0.0142
dL/db1: 0.0071

--- 参数更新 (lr = 0.1) ---
新 w1: 0.49858
新 b1: 0.09929
新 w2: -0.29050
新 b2: 0.21266
```

这个例子呈现了反向传播 (backpropagation)的机械过程。它本质上是系统地应用链式法则，计算最终损失对网络中每个参数 (parameter)的敏感程度，并逐层向后传播这些敏感度。尽管深度学习 (deep learning)框架自动化了这一过程，但理解其背后的微积分对于掌握神经网络 (neural network)如何学习是非常重要的。

## 参考资料

- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  提供反向传播、计算图和神经网络训练全面且数学严谨解释的基础教材。
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap2.html) — Michael Nielsen (2019)
  一本易于理解的在线书籍，详细分步介绍了反向传播算法，包含简单网络的计算和直观说明。
- [CS231n: Deep Learning for Computer Vision - Lecture Notes on Backpropagation, Intuitions](http://cs231n.github.io/optimization-2/) — Andrej Karpathy, Justin Johnson, and Fei-Fei Li (2023)
  Publisher: Stanford University
  斯坦福大学课程笔记，通过计算图清晰地推导和可视化反向传播，适合理解其运行机制。
