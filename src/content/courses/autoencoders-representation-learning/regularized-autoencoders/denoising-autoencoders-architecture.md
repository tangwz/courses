---
course: "autoencoders-representation-learning"
chapter: "regularized-autoencoders"
lesson: "denoising-autoencoders-architecture"
sourceId: 2829
sourceUrl: "https://apxml.com/zh/courses/autoencoders-representation-learning/chapter-3-regularized-autoencoders/denoising-autoencoders-architecture"
title: "去噪自编码器架构与训练"
description: "了解DAE如何从受损输入中重建干净数据。"
order: 3
plots: []
sourceHash: "9a9acc8c0c1d11b1ff3686b6cad0aa24c0ad05b44010b6f7a2ebb215c7715f2e"
sourceCorrections: []
---

标准自编码器虽然在降维方面表现良好，但有时会学习到对输入数据微小变化过于敏感的映射，甚至在瓶颈维度足够大时近似于恒等函数。这限制了它们可靠地捕获数据固有流形的能力。去噪自编码器（DAE）通过对基本自编码器训练过程进行修改来处理此问题，促使模型学习到更具弹性和有意义的表示。

### 核心思想：从损坏中重建

DAE的基本原理简单而有效：我们不直接训练自编码器重建其输入，而是首先有意地损坏输入数据，然后训练自编码器从损坏版本中恢复*原始、未损坏*的数据。

设想将一张略微模糊或带有噪声的图片输入到DAE中；目标不是重现模糊，而是输出原始的、清晰的图片。这促使自编码器学习捕获数据本质结构的特征，隐式地学习输入内部的统计依赖性，以“撤销”损坏过程。

### 损坏过程

训练DAE的第一步是：对干净的输入数据$x$应用随机损坏过程，以得到损坏版本$\tilde{x}$。这种损坏通常对每个训练样本或小批量随机引入。常见的损坏方法包括：

1. **加性各向同性高斯噪声：** 向输入添加从高斯分布$\mathcal{N}(0, \sigma^2 I)$中提取的噪声。

   
   $$
   \tilde{x} = x + \epsilon, \quad \text{式中 } \epsilon \sim \mathcal{N}(0, \sigma^2 I)
   $$
   

   标准差$\sigma$控制噪声水平。
2. **遮蔽噪声（Dropout）：** 随机将输入特征的一部分$p$设为零（或有时设为平均值等其他值）。

   
   $$
   \tilde{x}_i = m_i x_i, \quad \text{式中 } m_i \sim \text{Bernoulli}(1-p)
   $$
   

掩码$m$的每个元素$m_i$都是独立抽取的。这类似于应用于输入层的dropout。

3. **椒盐噪声：** 对于图像数据，随机将一部分像素设为它们的最小值（0，“椒”）或最大值（1或255，“盐”）可能值。

损坏类型及其强度（例如，噪声水平$\sigma$或遮蔽概率$p$）的选择是重要的超参数 (parameter) (hyperparameter)。损坏应足够明显，以防止自编码器仅仅学习恒等函数，但又不能过于严重，导致重建原始数据变得不可能。

### DAE架构

DAE的网络架构（编码器、瓶颈、解码器）通常与标准自编码器相同。主要区别完全在于训练目标和数据流。

> 去噪自编码器过程。干净的输入$x$被随机损坏为$\tilde{x}$。编码器将$\tilde{x}$映射到潜在编码$z$，解码器尝试将原始干净输入$x$重建为$\hat{x}$。损失是在原始$x$和重建$\hat{x}$之间计算的。

### DAE训练

设编码器函数为$z = f_{\theta}(\tilde{x})$，解码器函数为$\hat{x} = g_{\phi}(z)$。DAE通过最小化*原始干净输入*$x$与从*损坏输入*$\tilde{x}$生成的输出$\hat{x}$之间的平均重建误差来训练：


$$
\mathcal{L}(\theta, \phi) = \mathbb{E}_{x \sim p_{\text{data}}} \left[ \mathbb{E}_{q(\tilde{x}|x)} \left[ L(x, g_{\phi}(f_{\theta}(\tilde{x}))) \right] \right]
$$


在实践中，小批量训练时，我们采样一个干净输入$x$，使用所选的随机损坏过程生成一个损坏版本$\tilde{x}$，将$\tilde{x}$通过编码器和解码器，然后计算损失$L(x, \hat{x})$。

损失函数 (loss function)$L(x, \hat{x})$的常见选择与标准自编码器相同：

- **均方误差（MSE）：** 适用于连续值输入，如归一化 (normalization)像素强度。
  
  $$
  L_{\text{MSE}}(x, \hat{x}) = \frac{1}{N} \sum_{i=1}^{N} (x_i - \hat{x}_i)^2
  $$
  
- **二元交叉熵（BCE）：** 适用于二元输入或归一化到0和1之间的输入，常与最终解码器层中的sigmoid激活函数 (activation function)一起使用。
  
  $$
  L_{\text{BCE}}(x, \hat{x}) = - \frac{1}{N} \sum_{i=1}^{N} [x_i \log(\hat{x}_i) + (1 - x_i) \log(1 - \hat{x}_i)]
  $$
  

优化采用随机梯度下降 (gradient descent)（SGD）或其变体（Adam，RMSprop）等标准技术进行，以更新参数 (parameter)$\theta$和$\phi$。

### 学习特征

为什么这个过程会带来更好的表示？为了从损坏的$\tilde{x}$中成功重建原始$x$，自编码器必须捕获数据固有的结构和依赖性。它不能简单地学习一个简单的恒等映射，因为输入和目标是不同的。模型被迫学习如何“填补”缺失的信息或“消除”损坏过程中引入的噪声。

这鼓励编码器$f_{\theta}$提取稳定且能代表真实数据分布的特征，有效地将离流形位置的损坏输入映射回所学的数据流形。然后，解码器$g_{\phi}$使用这些稳定特征来重建位于流形上的干净数据点。这种隐式正则化 (regularization)使得DAE不易过拟合 (overfitting)，并帮助它们学习对下游任务有用的特征。

### 实现考量

在实现DAE时，对标准自编码器设置的主要增加部分是：在将输入数据送入编码器之前，对其应用的损坏步骤。

```python
# 示例：在PyTorch中应用遮蔽噪声
import torch
import torch.nn.functional as F

def add_masking_noise(inputs, corruption_level=0.2):
  """对输入张量应用遮蔽噪声。"""
  # 如果需要，确保输入是浮点张量，以便与Bernoulli兼容
  if not inputs.is_floating_point():
      inputs = inputs.float()
  # 根据corruption_level创建值被丢弃（设为0）的掩码
  mask = torch.bernoulli(torch.full_like(inputs, 1 - corruption_level))
  return inputs * mask

# --- 训练循环内部 ---
# 假设加载器提供干净输入的批次
# for clean_inputs in data_loader:
#   # 将数据移动到适当的设备（例如，GPU）
#   # clean_inputs = clean_inputs.to(device)
#
#   # 应用损坏
#   corrupted_inputs = add_masking_noise(clean_inputs, corruption_level=0.3)
#
#   # 前向传播
#   latent_code = encoder(corrupted_inputs)
#   reconstruction = decoder(latent_code)
#
#   # 根据原始干净输入计算损失
#   loss = F.mse_loss(reconstruction, clean_inputs)
#
#   # 反向传播和优化
#   optimizer.zero_grad()
#   loss.backward()
#   optimizer.step()
```

`corruption_level`（或噪声标准差$\sigma$）是一个通常需要调整的超参数 (parameter) (hyperparameter)。遮蔽噪声的常见范围是0.1到0.5。从中等水平开始，并根据验证性能进行调整。如果噪声太低，DAE的行为将类似于标准AE；如果噪声太高，重建任务可能变得过于困难，阻碍学习。

去噪自编码器提供了一种提高与标准自编码器相比，学习表示稳定性的实用方法。通过强制模型去噪损坏的输入，我们鼓励它捕获数据更基本的结构属性，使其成为表示学习工具集中一个有价值的工具。接下来，我们将研究另一种正则化 (regularization)技术：稀疏自编码器。

## 参考资料

- [Extracting and Composing Robust Features with Denoising Autoencoders](https://doi.org/10.1145/1390156.1390294) — Pascal Vincent, Hugo Larochelle, Yoshua Bengio, Pierre-Antoine Manzagol (2008)
  Journal: Proceedings of the 25th International Conference on Machine Learning; Publisher: ACM; Pages: 1096-1103; DOI: [10.1145/1390156.1390294](https://doi.org/10.1145/1390156.1390294)
  这篇基础论文介绍了去噪自动编码器及其通过从损坏输入中重建来学习稳定表示的理论基础。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  这本权威教科书详细解释了自动编码器，包括去噪自动编码器，它们的架构、训练和理论基础，将其置于深度学习的总体背景中。
- [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://jmlr.org/papers/v15/srivastava14a.html) — Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov (2014)
  Journal: The Journal of Machine Learning Research; Volume: 15; Pages: 1929-1958
  本文介绍了 dropout 正则化技术，该技术与去噪自动编码器中讨论的掩蔽噪声损坏方法直接相关。
