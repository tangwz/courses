---
course: "deep-learning-regularization-optimization"
chapter: "optimization-refinements-tuning"
lesson: "step-decay-lr-schedule"
sourceId: 5001
sourceUrl: "https://apxml.com/zh/courses/deep-learning-regularization-optimization/chapter-7-optimization-refinements-tuning/step-decay-lr-schedule"
title: "步进衰减策略"
description: "描述步进衰减学习率调度方法。"
order: 4
plots: ["plots/5001-0.json"]
sourceHash: "5c54186a570c293d16931a71fd97e0808f32ba0a16c2e5b690702138fd838644"
sourceCorrections: []
---

最直接且常用的训练中调整学习率的方法之一是**步进衰减**策略。其核心思路是，从较高的学习率开始，然后以预设的时间间隔（例如，周期或步数）将其按特定因子减少。这种做法使得模型在训练初期，当参数 (parameter)可能离最优值较远时，可以取得较大进展；而在接近最小值时，则采取更小、更谨慎的步长，这有助于稳定收敛并精调权重 (weight)。

设想您在广阔的开阔地带开始搜寻。起初，您可以迈开大步快速覆盖区域。当您接近目标所在的位置时，您会缩短步长，在该局部区域更仔细地搜寻。步进衰减模仿了这种直觉。

### 步进衰减的工作方式

此策略基于几个参数 (parameter)运行：

1. **初始学习率 ($\alpha_0$):** 训练开始时使用的学习率。
2. **衰减因子 ($\gamma$):** 一个（通常小于1，例如0.1、0.5）用于乘以学习率的因子。
3. **步长 ($S$):** 达到该周期数（或有时是迭代/步数）后学习率会降低。

在训练过程中调整学习率的一种直接且常用的方法是**步长衰减 (Step Decay)** 策略。其主要思想是初始时使用相对较高的学习率，然后在预设的间隔（如周期或步数）后将其按一定因子减小。这种方法使得模型在训练早期能够取得显著进展，此时参数可能远离最优值；随后，当模型接近最小值时，它会采取更小的、更精细的步长，这有助于稳定收敛并对权重 (weight)进行微调 (fine-tuning)。


$$
\alpha_t = \alpha_0 \times \gamma^k = \alpha_0 \times \gamma^{\lfloor t / S \rfloor}
$$


例如，如果 $\alpha_0 = 0.01$、$\gamma = 0.1$、$S = 10$ 个周期：

- 周期 0-9: $\alpha = 0.01$
- 周期 10-19: $\alpha = 0.01 \times 0.1 = 0.001$
- 周期 20-29: $\alpha = 0.001 \times 0.1 = 0.0001$
- 依此类推...

学习率在步与步之间保持不变，当达到周期阈值时会即刻下降。

### 步进衰减的可视化

以下图表显示了一个典型的步进衰减策略，共50个周期，初始学习率为0.01，衰减因子为0.5，步长为15个周期。



![步进衰减学习率策略](plots/5001-0.json)



> 显示步进衰减的学习率策略，其中 $\alpha_0 = 0.01$、$\gamma = 0.5$、$S = 15$。学习率每15个周期减半。请注意y轴是对数刻度。

### 步进衰减的调整

步进衰减的效果取决于为 $\alpha_0$、$\gamma$ 和 $S$ 选择合适的值。这些超参数 (parameter) (hyperparameter)通常需要调整：

- **初始学习率 ($\alpha_0$):** 通常通过实验确定，根据优化器和问题，常从0.1、0.01或0.001等值开始。
- **衰减因子 ($\gamma$):** 常见值为0.1（大幅下降）或0.5（温和下降）。选择取决于您希望多快地降低学习率。
- **步长 ($S$):** 这需要监测训练和验证损失。常见做法是观察验证损失何时趋于平稳，然后将步长设定为在该点附近触发衰减。您可能在训练期间多次衰减学习率。

步进衰减虽然简单，但需要细致地手动调整策略（何时以及降低多少）。如果下降发生过早或过晚，或者因子过大或过小，都可能阻碍收敛。

### 实现示例 (PyTorch)

大多数深度学习 (deep learning)框架都提供了实现步进衰减的便捷方法。在PyTorch中，您可以使用 `torch.optim.lr_scheduler.StepLR`：

```python
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.nn import Linear # 示例层

# 假设 'model' 是您定义的神经网络
model = Linear(10, 2) # 一个简单的模型部分示例

# 选择一个优化器（例如，Adam 或 SGD）
optimizer = optim.Adam(model.parameters(), lr=0.01) # 初始学习率 = 0.01

# 定义 StepLR 调度器
# 每10个周期将学习率降低0.1倍
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# --- 在您的训练循环内部 ---
num_epochs = 30
for epoch in range(num_epochs):
    # model.train()
    # ... 训练前向传播，计算损失，反向传播 ...

    optimizer.step() # 根据当前学习率更新权重

    # 更新学习率
    scheduler.step() 

    # 可选：打印当前学习率
    # current_lr = scheduler.get_last_lr()[0] 
    # print(f"周期 {epoch+1}, 学习率: {current_lr}")

    # ... 验证循环 ...

# 示例输出 (学习率变化):
# 周期 1, 学习率: 0.01
# ...
# 周期 10, 学习率: 0.01 (学习率在第10个周期步长*之后*下降)
# 周期 11, 学习率: 0.001 
# ...
# 周期 20, 学习率: 0.001 (学习率在第20个周期步长*之后*下降)
# 周期 21, 学习率: 0.0001
# ...
```

在这个PyTorch示例中，`StepLR` 接收优化器、`step_size`（以周期为单位）和 `gamma` 因子作为参数 (parameter)。`scheduler.step()` 调用通常在每个周期执行一次，在 `optimizer.step()` 之后，以便根据策略更新下一个周期的学习率。

步进衰减提供了一种简单、易于理解的学习率调整方式。尽管存在更复杂的策略，但其有效性和易于实现使其成为深度学习实践者工具集中的一种有价值的工具。

## 参考资料

- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  这本教材对深度学习优化（包括各种学习率调度策略）提供了全面的理论阐述。
- [\`torch.optim.lr_scheduler.StepLR\`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html) — PyTorch (2024)
  Journal: PyTorch Documentation; Publisher: PyTorch Foundation
  PyTorch中步长衰减学习率调度器的官方文档，对实际应用至关重要。
- [Dive into Deep Learning](https://d2l.ai/chapter_optimization/lr-schedules.html#learning-rate-scheduler-implementations) — Aston Zhang, Zachary C. Lipton, Mu Li, and Alex Smola (2024)
  Journal: D2L.ai; Publisher: Cambridge University Press
  一本权威的开源书籍，提供了学习率调度（包括步长衰减）的实际示例和清晰解释。
- [Optimization: Stochastic Gradient Descent (SGD)](http://cs231n.github.io/optimization-1/) — Stanford CS231n Course Staff (2023)
  Journal: Stanford CS231n: Convolutional Neural Networks for Visual Recognition; Publisher: Stanford University
  讲义提供了深度学习中优化技术的易懂解释，涵盖了各种学习率调度及其实际考量。
