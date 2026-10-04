---
course: "optimization-techniques-ml"
chapter: "adaptive-learning-rate-algorithms"
lesson: "hands-on-adaptive-optimizers-comparison"
sourceId: 1275
sourceUrl: "https://apxml.com/zh/courses/optimization-techniques-ml/chapter-3-adaptive-learning-rate-algorithms/hands-on-adaptive-optimizers-comparison"
title: "实践操作：比较自适应优化器"
description: "在实际机器学习问题上实现并比较 AdaGrad、RMSprop 和 Adam 的性能。"
order: 8
plots: ["plots/1275-0.json", "plots/1275-1.json"]
sourceHash: "59f6096c4906be069d259af40cc34ef192d4bdf960564e6e2711df52775d0f02"
sourceCorrections: []
---

AdaGrad、RMSprop 和 Adam 是具有重要理论优势的自适应学习率算法。观察这些优化器在具体机器学习 (machine learning)任务中的表现，能为其应用提供有益的直观认识。在此，将比较这三种常见自适应学习率算法的收敛表现和性能。

我们的目标是观察每种优化器最小化损失函数 (loss function)的速度，以及训练过程中验证性能的变化。我们将采用标准设置，以确保公平比较。

### 实验设置

我们需要一个模型、一个数据集和一个损失函数 (loss function)。为了简化和便于重现，我们使用：

1. **数据集：** MNIST手写数字数据集。这是一个经典基准，在大多数机器学习 (machine learning)框架中都容易获取。我们将使用标准的训练集和验证集划分。
2. **模型：** 一个简单的多层感知器（MLP）。具有一或两层隐藏层的网络足以观察优化器间的差异。例如，一个这样的架构：输入（784个特征）-> 线性（128个单元）-> ReLU -> 线性（64个单元）-> ReLU -> 线性（10个单元）-> LogSoftmax。
3. **损失函数：** 负对数似然损失（NLLLoss）或交叉熵损失，适用于多类别分类。
4. **框架：** 我们将假定使用 PyTorch 或 TensorFlow/Keras，因为从头实现这些优化器超出了本课程的范围。这些框架提供了现成的实现。

### 实现步骤

无论使用哪种优化器，核心训练循环都保持类似，但实例化步骤有所不同。下面是使用 PyTorch 类似语法的一个概览：

1. **加载数据：** 为训练集和验证集准备 `DataLoader` 实例。
2. **定义模型：** 实例化你的 MLP 模型。

# 模型定义示例

````
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784) # 展平图像
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1) # 输出对数概率
        return x

model = SimpleMLP()
# 如果可用，将模型移动到 GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
```
````

3. **定义损失：**
`python criterion = nn.NLLLoss()` 
4. **实例化优化器：** 为你要比较的每个优化器创建独立的实例。我们最初将使用标准默认超参数 (parameter) (hyperparameter)，但请记住这些是可以调整的。
\`\`\`python
import torch.optim as optim

````
# 常见初始学习率（示例）
lr = 0.001

# 实例化优化器
optimizer_adagrad = optim.Adagrad(model.parameters(), lr=lr)
optimizer_rmsprop = optim.RMSprop(model.parameters(), lr=lr)
optimizer_adam = optim.Adam(model.parameters(), lr=lr)

# 存储它们以便迭代（或运行单独的训练脚本）
optimizers = {
    "AdaGrad": optimizer_adagrad,
    "RMSprop": optimizer_rmsprop,
    "Adam": optimizer_adam
}
```
*注意：* 在使用*每个*优化器进行训练之前，你需要重新初始化模型权重，以确保从相同的起点开始公平比较。
````

5. **训练循环：** 对于每个优化器，运行固定数量的训练周期。
\`\`\`python

# 训练循环（针对一个优化器）

````
# num_epochs = 10
# train_losses = []
# val_accuracies = []

# # 在开始对当前优化器进行训练之前，在此处重新初始化模型权重

# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#     for images, labels in train_loader:
#         # images, labels = images.to(device), labels.to(device) # 将数据移动到设备

#         optimizer.zero_grad() # 清零此批次的梯度
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward() # 计算梯度
#         optimizer.step() # 更新权重
#         running_loss += loss.item()

#     epoch_loss = running_loss / len(train_loader)
#     train_losses.append(epoch_loss)

#     # 验证阶段
#     model.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for images, labels in val_loader:
#             # images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#     epoch_acc = 100 * correct / total
#     val_accuracies.append(epoch_acc)
#     print(f'Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Val Acc: {epoch_acc:.2f}%')

# # 在训练下一个优化器之前，存储当前优化器的结果
```
````

6. **记录结果：** 存储每个优化器在每个训练周期内的训练损失和验证准确率（或损失）。

### 可视化和结果解读

绘制记录的指标是比较性能的最佳方式。



![训练损失比较](plots/1275-0.json)



> AdaGrad、RMSprop 和 Adam 的训练损失（对数尺度）与训练周期的关系图。越低越好。请注意，RMSprop 和 Adam 通常比 AdaGrad 更快地下降。



![验证准确率比较](plots/1275-1.json)



> 验证准确率与训练周期的关系图。越高越好。在此类设置中，Adam 和 RMSprop 通常比 AdaGrad 更快地达到更高的准确率。

**常见观察：**

- **初期收敛：** 与 AdaGrad 相比，Adam 和 RMSprop 在初期通常表现出更快的损失下降速度。这通常归因于 RMSprop 修正了 AdaGrad 学习率迅速减小的问题，以及 Adam 引入了动量。
- **AdaGrad 的学习率衰减：** 你可能会观察到 AdaGrad 在后期训练周期中进展明显放缓。因为它累积了所有过去时间步的梯度平方，因此频繁更新的参数 (parameter)的学习率会变得非常小，可能阻碍其收敛到最佳方案。
- **Adam 作为默认选择：** Adam 通常在各种问题上使用默认超参数 (hyperparameter)（$\beta_1=0.9, \beta_2=0.999$）时表现良好，使其成为一个常用的起点。它结合了 RMSprop 的优点（基于近期梯度大小的每参数缩放）和动量。
- **最终性能：** 尽管 Adam 和 RMSprop 通常收敛更快，但它们的最终验证性能可能非常接近，甚至在特定情况下 RMSprop 或经过良好调整的带动量的 SGD 会稍好一些。如果 AdaGrad 的学习率衰减过快，它可能会落后。
- **超参数敏感性：** 尽管自适应方法相比标准 SGD 减少了对基础学习率 $\eta$ 的调整需求，但它们的性能仍可能对其他超参数（如 Adam 中的 $\beta_1, \beta_2$ 或 RMSprop 中的 $\alpha$）以及初始学习率本身很敏感。默认值是良好的起点，但进行调整可能会带来更好的结果。

### 结论

这次实践比较突出了自适应学习率方法的优点。通过根据梯度历史动态调整每个参数 (parameter)的步长，AdaGrad、RMSprop 和 Adam 通常比固定学习率方法更快地收敛，尤其是在神经网络 (neural network)常见的复杂损失曲面上。

尽管 Adam 因其稳定性和技术组合常成为一个强有力的默认选择，但了解 AdaGrad 和 RMSprop 的行为能提供有益的背景信息。请记住，最佳优化器可能取决于具体问题。遵循本章讨论的原则进行实验，对于在机器学习 (machine learning)任务中取得最佳结果非常重要。这次实践经验使你在选择和使用这些强大的优化工具时能做出明智的决策。

## 参考资料

- [Adaptive Subgradient Methods for Online Learning and Stochastic Optimization](https://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf) — John Duchi, Elad Hazan, and Yoram Singer (2011)
  Journal: Journal of Machine Learning Research; Publisher: Journal of Machine Learning Research; Volume: 12; Pages: 2121-2159; DOI: [10.1162/JMLR.2011.12.DUCHI11A](https://doi.org/10.1162/JMLR.2011.12.DUCHI11A)
  介绍了AdaGrad算法，一种基础的自适应学习率方法。
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) — Diederik P. Kingma and Jimmy Ba (2015)
  Journal: 3rd International Conference for Learning Representations, San Diego, 2015; DOI: [10.48550/arXiv.1412.6980](https://doi.org/10.48550/arXiv.1412.6980)
  提出了Adam优化器，它结合了AdaGrad和RMSprop的思想以及动量，广泛用于深度学习。
- [Deep Learning](https://www.deeplearningbook.org/contents/optimization.html) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: 281-304
  提供了深度学习中优化算法的全面解释，包括AdaGrad、RMSprop和Adam。
- [torch.optim - PyTorch 2.3 documentation](https://pytorch.org/docs/stable/optim.html) — PyTorch Developers (2025)
  Publisher: PyTorch Foundation
  PyTorch优化包的官方文档，提供了AdaGrad、RMSprop和Adam的实现细节。
