---
course: "how-to-build-a-large-language-model"
chapter: "identifying-mitigating-training-instabilities"
lesson: "monitoring-training-metrics"
sourceId: 6072
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-24-identifying-mitigating-training-instabilities/monitoring-training-metrics"
title: "监控训练指标（损失、梯度范数）"
description: "训练期间跟踪损失曲线、梯度范数和激活统计数据。"
order: 2
plots: ["plots/6072-0.json", "plots/6072-1.json"]
sourceHash: "0c067c992a5682d7fb358778695fd8950d5361d6adb1b97371b20d606de9ba7a"
sourceCorrections: []
---

有效的监控是应对可能破坏大规模语言模型训练稳定性的第一道防线。仅仅启动持续数天或数周的训练任务并寄希望于最好的结果是不现实的。相反，工程师需要通过重要的度量指标持续观察模型的行为。这有助于尽早发现问题，从而在大量计算资源被浪费之前进行及时干预。诊断训练状态最有用的两个度量指标是训练损失和梯度范数。

### 跟踪训练损失

训练损失量化 (quantization)了模型在任何给定时刻对训练数据的表现。对于语言模型而言，这通常是交叉熵损失等度量，它反映了模型在预测下一个词元 (token)时的不确定性。

**预期表现：** 在正常的训练过程中，损失通常会随时间下降，在早期阶段迅速下降，然后随着训练进行而变慢，最终在模型收敛时趋于平稳。轻微的波动是正常的，但总体趋势应是下降的。

**不稳定迹象：**

- **损失骤升：** 损失值突然急剧增加。这通常表明模型遇到了有问题的数据批次，或者学习率过高，导致更新不稳定。
- **NaN 或 Inf 损失：** 损失值变为 `NaN`（非数值）或无穷大。这是一个严重故障，通常由数值溢出（例如除以零、对零或负数取对数）或梯度爆炸引起。一旦出现 NaN，训练就无法继续。
- **发散：** 损失随时间持续增加，表明模型的预测越来越差，而不是变好。
- **震荡：** 损失剧烈波动，没有明确的下降趋势，可能表明学习率、优化算法稳定性或数据质量存在问题。

**实现：** 在标准训练循环中记录损失是简单直接的。通常在前向传播之后计算损失，并定期记录其值。

```python
import torch
import torch.nn as nn
# 假设 model、data_loader、optimizer 已定义

# 示例日志记录设置（可以使用 TensorBoard、WandB 等）
def log_metric(step, metric_name, value):
    # 替换为你的实际日志记录实现
    print(f"步骤 {step}: {metric_name} = {value:.4f}")

global_step = 0
for batch in data_loader:
    optimizer.zero_grad()

    # 假设 inputs 和 targets 是从批次中获取的
    inputs = batch['input_ids'].to('cuda')
    targets = batch['labels'].to('cuda')

    outputs = model(inputs)
    # 假设损失计算涉及为交叉熵进行形状重塑
    loss_fct = nn.CrossEntropyLoss()
    loss = loss_fct(outputs.view(-1, model.config.vocab_size), targets.view(-1))

    # 记录损失（例如，每 10 步）
    if global_step % 10 == 0:
        log_metric(global_step, "train_loss", loss.item())

    loss.backward()
    optimizer.step()
    global_step += 1

    # 尽早检查是否存在 NaN 损失
    if torch.isnan(loss):
        print(f"在步骤 {global_step} 检测到 NaN 损失！停止训练。")
        break
```

随时间可视化损失曲线很重要。诸如 TensorBoard 或 Weights & Biases 等工具使这变得容易。



![训练损失模式](plots/6072-0.json)



> 示例损失曲线显示了稳定的训练、突然的损失骤升以及损失随时间增加的发散情况。请注意对数 Y 轴，这在查看损失时很常见。

### 监控梯度范数

梯度是指示损失函数 (loss function)最陡峭上升方向和大小的向量 (vector)。该向量的*范数*（通常是 L2 范数，
$$
||
abla L||_2
$$
）度量其大小。监控梯度范数提供了关于应用于模型权重 (weight)更新尺度的见解。


$$
||\nabla L||_2 = \sqrt{\sum_{p \in \text{参数}} ||\nabla_p L||_2^2}
$$


$abla_p L$ 是损失 $L$ 相对于特定参数 (parameter)张量 $p$ 的梯度。

**为何它重要：**

- **梯度爆炸：** 过大的梯度范数表明权重更新将非常大，可能导致数值溢出、损失骤升和发散。这是训练不稳定的常见原因，尤其是在深度网络中。
- **梯度消失：** 非常小或为零的梯度范数表明更新极小或不存在，导致训练停滞。尽管与旧的 RNN 相比，这在 Transformer 中作为灾难性故障较不常见（归因于残差连接和归一化 (normalization)），但它仍然会阻碍收敛。
- **学习率的合适性：** 梯度范数与学习率有关联。持续较高的范数可能表明学习率需要降低，而持续微小的范数（在某些情况下）可能需要增加，尽管梯度消失更多是架构或初始化问题。

**预期表现：** 梯度范数通常开始时较高，并随着模型收敛和损失在最小值附近趋于平坦而下降。然而，其行为高度依赖于学习率调度、优化器和数据。可能会出现显著波动，但极端大的值是一个危险信号。

**不稳定迹象：**

- **突然的大幅骤升：** 比平时大几个数量级的值通常在 NaN 损失或显著损失骤升之前发生。这是梯度爆炸的显著特征。
- **持续接近零：** 如果梯度范数在训练早期变得非常小并保持如此，可能表明存在梯度消失问题或初始化问题。

**实现：** 计算总梯度范数需要在调用 `loss.backward()` 之后但在调用 `optimizer.step()` *之前*遍历所有模型参数。梯度裁剪，一种稍后讨论的技术，通常无论如何都会涉及计算这个范数。

```python
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
# 假设 model、data_loader、optimizer 已定义，log_metric 已定义

global_step = 0
# 定义用于裁剪的最大梯度范数值（在稳定性技术中讨论）
max_grad_norm = 1.0

for batch in data_loader:
    optimizer.zero_grad()

    # --- 前向传播和损失计算，如前所述 ---
    inputs = batch['input_ids'].to('cuda')
    targets = batch['labels'].to('cuda')
    outputs = model(inputs)
    loss_fct = nn.CrossEntropyLoss()
    loss = loss_fct(
        outputs.view(-1, model.config.vocab_size),
        targets.view(-1)
    )

    # 记录损失
    if global_step % 10 == 0:
        log_metric(global_step, "train_loss", loss.item())

    if torch.isnan(loss):
        print(f"在步骤 {global_step} 检测到 NaN 损失！停止训练。")
        break

    loss.backward()

    # --- 在优化器步骤之前计算并记录梯度范数 ---
    # 创建参数梯度的生成器
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if len(grads) > 0:
      # 计算所有梯度的 L2 范数
      total_norm = torch.norm(
          torch.stack([
              torch.norm(g.detach(), 2.0) for g in grads
          ]),
          2.0
      )

      # 记录梯度范数（例如，每 10 步）
      if global_step % 10 == 0:
          log_metric(global_step, "grad_norm", total_norm.item())

      # 可选：裁剪梯度（常见做法）
      # clip_grad_norm_(model.parameters(), max_grad_norm)

      # 检查梯度爆炸
      if total_norm > 100 * max_grad_norm: # Heuristic threshold
            print(
                f"警告：在步骤 {global_step} 检测到高梯度范数（{total_norm:.2f}）"
            )
    else:
        # 处理没有梯度的情况
        # （例如，如果模型没有可训练参数）
        if global_step % 10 == 0:
            log_metric(global_step, "grad_norm", 0.0)


    optimizer.step()
    global_step += 1
```

可视化梯度范数以及损失，提供了关于训练动态的更完整情况。



![梯度范数模式](plots/6072-1.json)



> 示例梯度 L2 范数曲线显示了稳定的下降和范数急剧增加的梯度爆炸情况。

通过仔细监控训练损失和梯度范数，您能对训练过程获得必要的了解。这些度量指标充当早期预警系统，使您能够在潜在不稳定情况升级为严重故障之前发现它们，从而节省宝贵的时间和计算资源。这些曲线中的异常是诊断具体根本问题的起点，我们将在接下来的内容中查看。

## 参考资料

- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  提供神经网络训练的核心理论，包括优化、损失函数以及梯度爆炸/消失等问题。
- [CS224n: Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/) — Diyi Yang, Tatsunori Hashimoto (2023)
  Publisher: Stanford University
  自然语言处理深度学习课程材料，涵盖语言模型训练实践、优化和监控指标。
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) — Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei (2020)
  Journal: arXiv preprint arXiv:2001.08361; DOI: [10.48550/arXiv.2001.08361](https://doi.org/10.48550/arXiv.2001.08361)
  研究论文分析大型语言模型训练损失与模型及数据集规模的行为关系，为性能设定预期。
