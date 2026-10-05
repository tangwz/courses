# 在实践中应用Dropout

来源：[原文](https://apxml.com/zh/courses/deep-learning-regularization-optimization/chapter-3-dropout-regularization/implementing-dropout)

[返回章节目录](README.md) · [返回课程目录](../README.md)

将Dropout整合到神经网络 (neural network)中涉及使用常用深度学习 (deep learning)库。PyTorch和TensorFlow等框架提供了方便的模块或层，可以处理实现细节，包括推理 (inference)时必要的缩放。

### 在PyTorch中添加Dropout层

在PyTorch中，你可以使用`torch.nn.Dropout`模块添加Dropout。在模型定义中，通常将其插入层之间，通常在全连接层的激活函数 (activation function)之后。

提供给`nn.Dropout`的主要参数 (parameter)是`p`，它指定了训练期间元素（神经元输出）被归零的概率。请记住，这就是我们之前讨论过的'dropout率'，一个你可能需要调整的超参数 (hyperparameter)。

这是一个PyTorch中整合了Dropout的简单顺序模型示例：

```python
import torch
import torch.nn as nn

# 定义模型参数
input_size = 784  # 示例：扁平化的MNIST图像
hidden_size1 = 256
hidden_size2 = 128
output_size = 10  # 示例：10个数字类别
dropout_prob = 0.5 # dropout概率

# 定义带有Dropout的模型
model = nn.Sequential(
    nn.Linear(input_size, hidden_size1),
    nn.ReLU(),
    nn.Dropout(p=dropout_prob), # 第一个隐藏层激活后进行Dropout
    nn.Linear(hidden_size1, hidden_size2),
    nn.ReLU(),
    nn.Dropout(p=dropout_prob), # 第二个隐藏层激活后进行Dropout
    nn.Linear(hidden_size2, output_size)
    # 注意：通常不在输出层之前直接应用Dropout
)

print(model)
```

在这段代码中，`nn.Dropout(p=0.5)`层被添加在隐藏层的`ReLU`激活函数之后。这意味着在训练期间，来自前一个`ReLU`的每个神经元输出对于该特定前向传播有50%的概率被设为零。剩余的活跃神经元的输出会按$1/(1-p)$的比例放大以进行补偿（这是由层自动处理的'反向Dropout'技术）。

### Dropout层的位置

最常见的做法是将Dropout层放置在隐藏层的激活函数 (activation function)*之后*，如上例所示。在激活之前应用它也能起作用但不太常见。它通常不直接应用于输入层，也通常不直接应用于输出层之前，特别是当输出层代表概率（例如使用Softmax时）或有特定的缩放要求时。

对于卷积神经网络 (neural network) (CNN)（CNNs），Dropout可以应用于卷积层之后（通常在池化层之后），或应用于通常跟随卷积块的全连接层中。存在专门的版本，例如`Dropout2d`，它将整个特征图而不是单个元素归零，有时对卷积层更有效。我们在上一节中简要提到了这一点。对于循环神经网络 (RNN)（RNNs），在时间步之间天真地应用标准Dropout会阻碍学习；通常更倾向于使用变分Dropout等特定技术，但这些超出了本次介绍的范围。

### 训练模式与评估模式：一个重要区别

使用内置Dropout层的一个重要方面是它们自动处理训练与评估（推理 (inference)/测试）模式。

- **在训练期间 (`model.train()`):** Dropout层会以概率$p$随机将神经元输出归零，并按描述缩放其余部分。这会引入噪声并防止共同适应。
- **在评估期间 (`model.eval()`):** Dropout层变为非活动状态。它只是不加修改地传递所有输入。训练期间应用的缩放（反向Dropout）确保期望的输出量级在训练和评估之间保持一致，消除了在测试时进行单独缩放的需要。

在训练循环开始前使用`model.train()`，并在执行验证或测试前使用`model.eval()`，明确将你的模型设置为正确的模式非常重要。在推理期间未能设置`model.eval()`意味着你仍在随机丢弃单元，导致有噪声和次优的预测。

让我们可视化在训练期间，激活后应用了Dropout的层内基本流程：

> 一个层在激活后整合Dropout的流程。在训练期间，Dropout模块处于活动状态。在评估期间 (`model.eval()`)，它作为恒等函数作用。

这是一个PyTorch中的快速演示：

```python
# 创建一个dropout层
dropout_layer = nn.Dropout(p=0.5)
# 创建一些虚拟输入数据
dummy_input = torch.ones(1, 10) # 全1张量

# 将模型设置为训练模式
dropout_layer.train()
output_train = dropout_layer(dummy_input)
print("Output (Training Mode):", output_train) # 一些元素将为0，其他元素将按2倍缩放

# 将模型设置为评估模式
dropout_layer.eval()
output_eval = dropout_layer(dummy_input)
print("Output (Evaluation Mode):", output_eval) # 所有元素将为1（直通）
```

这段代码演示了`nn.Dropout`层如何根据通过`.train()`或`.eval()`设置的模式表现不同。

因此，使用标准库函数实现Dropout非常直接。主要的考量是选择dropout概率`p`，以及决定在你的网络架构中何处策略性地放置Dropout层。接下来的实践环节将为你提供一个机会，将这些层添加到网络中并观察它们的效果。

## 参考资料

- [Dropout: A Simple Way to Prevent Overfitting in Neural Networks](http://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf) — Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, Ruslan Salakhutdinov (2014)
  Journal: Journal of Machine Learning Research; Volume: 15; Pages: 1929-1958
  这篇原始论文介绍了用于神经网络的 Dropout 正则化技术，呈现了其核心思想和优势。
- [nn.Dropout](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html) — PyTorch documentation (2024)
  PyTorch `nn.Dropout` 模块的官方文档，详细说明了其使用方法、参数以及在训练和评估期间的行为。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press
  一本关于深度学习的教科书，在模型优化和泛化背景下详细了 Dropout 正则化。

---

[上一节](06-%E5%8D%B7%E7%A7%AF%E5%B1%82%E5%92%8C%E5%BE%AA%E7%8E%AF%E5%B1%82%E7%9A%84%E4%BD%BF%E7%94%A8%E8%80%83%E9%87%8F.md) · [下一节](08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%B7%BB%E5%8A%A0%20Dropout%20%E5%B1%82.md)
