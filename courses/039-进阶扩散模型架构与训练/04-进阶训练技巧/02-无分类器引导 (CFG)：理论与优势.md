# 无分类器引导 (CFG)：理论与优势

来源：[原文](https://apxml.com/zh/courses/advanced-diffusion-architectures/chapter-4-advanced-diffusion-training/classifier-free-guidance-theory)

[返回章节目录](README.md) · [返回课程目录](../README.md)

条件生成，即引导扩散模型生成符合特定属性（如类别标签或文本描述）输出的能力，是其实际使用的一个根本。一种早期方法，即*分类器引导*，使用一个独立的、预训练 (pre-training)的分类器网络来引导采样过程。尽管有效，但此方法增加了复杂性：它需要训练和维护一个额外的模型，且分类器对内容的理解可能与扩散模型内部的表达形式不完全一致，有时会导致欠佳的结果，或需要仔细调整。

无分类器引导 (CFG) 提供了一种更精巧且通常更有效的方法，通过*不*依赖外部分类器来实现引导。核心思想非常直接：训练单个扩散模型来预测条件和无条件噪声估计，然后在采样时利用这些预测之间的差异，引导生成过程趋向于设定的条件。

### CFG机制：朝向条件进行外推

我们将扩散模型（通常是 U-Net 或 Transformer）表示为 $\epsilon_\theta$，其参数 (parameter)为权重 (weight) $\theta$。此模型经过训练，用于预测在时间步 $t$ 时添加到输入 $x_t$ 的噪声。

- **条件预测：** 在给定条件 $c$（例如，类别标签嵌入 (embedding)、文本嵌入）时，模型预测以 $c$ 为条件的噪声： $\epsilon_\theta(x_t, c, t)$。
- **无条件预测：** *同一个*模型 $\epsilon_\theta$ 也被训练来预测未提供特定条件时的噪声。这通常通过将条件 $c$ 替换为特殊的“空”条件标记 (token) $\emptyset$（例如，零向量 (vector)或用于“无条件”的习得嵌入）来实现：$\epsilon_\theta(x_t, \emptyset, t)$。

在采样期间，CFG 不仅仅使用条件预测 $\epsilon_\theta(x_t, c, t)$ 来估计去噪后的 $x_0$ 或噪声 $\epsilon$，而是计算一个修正的噪声估计 $\tilde{\epsilon}_\theta$。这是通过获取无条件预测，并沿着条件预测所指示的方向进一步移动来实现的：


$$
\tilde{\epsilon}_\theta(x_t, c, t) = \epsilon_\theta(x_t, \emptyset, t) + w \cdot (\epsilon_\theta(x_t, c, t) - \epsilon_\theta(x_t, \emptyset, t))
$$


另外，这也可以写成：


$$
\tilde{\epsilon}_\theta(x_t, c, t) = (1+w) \epsilon_\theta(x_t, c, t) - w \epsilon_\theta(x_t, \emptyset, t)
$$


这里，$w$ 是**引导尺度**（有时表示为 $s$ 或 $\gamma$）。这个标量超参数 (hyperparameter)控制引导的强度：

- 如果 $w=0$，我们得到无条件预测 $\epsilon_\theta(x_t, \emptyset, t)$，忽略条件 $c$。生成过程是纯粹无条件的。
- 如果 $w=1$，公式（在其第一种形式中）简化为 $\epsilon_\theta(x_t, c, t)$，仅使用标准条件预测。
- 如果 $w > 1$，我们对外推标准条件预测，更强地推动模型朝向条件 $c$。$w$ 值越高，对条件的依从性越强，但这通常会牺牲样本多样性，如果设置过高还可能引入伪影。

可以将 $\epsilon_\theta(x_t, \emptyset, t)$ 看作一个基线预测，而将 $(\epsilon_\theta(x_t, c, t) - \epsilon_\theta(x_t, \emptyset, t))$ 看作噪声空间中从无条件估计指向条件估计的“引导方向”向量。CFG 将此方向向量按 $w$ 进行缩放，并将其添加到基线。

> 图示无分类器引导过程。同一个扩散模型在每个采样步骤中对噪声输入 $x_t$ 进行两次评估：一次是使用目标条件 $c$（评估 2），另一次是使用空条件 $\emptyset$（评估 1）。然后，使用引导尺度 $w$ 组合由此产生的条件和无条件噪声预测，以生成最终的引导噪声估计 $\tilde{\epsilon}_\theta$。

### CFG的训练

主要在于训练*同一个模型*来处理条件生成和无条件生成。

这通常通过在部分训练样本中随机“丢弃”条件信息来实现。例如，在训练期间：

1. 选择一批数据 $(x, c)$。
2. 对于批次中的每个项目，以某个概率 $p_{uncond}$（例如，10-20%）将实际条件 $c$ 替换为空条件标记 (token) $\emptyset$。
3. 将（可能已修改的）批次 $(x_t, c')$，而 $c'$ 可以是 $c$ 或 $\emptyset$，通过模型 $\epsilon_\theta$。
4. 基于预测 $\epsilon_\theta(x_t, c', t)$ 和真实噪声 $\epsilon$ 计算损失。

这个简单的步骤迫使模型 $\epsilon_\theta$ 使用同一组网络权重 (weight)，学习特定条件 $c$ 和通用空条件 $\emptyset$ 的有意义表示。它有效地训练模型在单一模型中执行两项任务。

### 无分类器引导的优势

CFG 已成为引导扩散模型的标准技术，因为它相比分类器引导具有多项重要优势：

1. **简洁性：** 它省去了训练、存储和加载单独分类器模型的需要。这大大简化了整体训练和推理 (inference)流程。引导机制是扩散模型自身固有的。
2. **更好的连贯性：** 由于引导来自执行去噪的同一个模型，因此引导信号通常与模型内部的生成过程更一致，相比来自外部分类器的信号，这可能带来更高质量和更连贯的样本。
3. **灵活性：** CFG 可以轻松适用于各种形式的条件设定，包括类别标签、文本嵌入 (embedding)（如 CLIP 嵌入）、图像嵌入，甚至条件组合，只要能够定义一个空表示 $\emptyset$。
4. **增强的控制：** 引导尺度 $w$ 提供了一种直接且直观的方式来控制样本保真度（对条件 $c$ 的依从性）与样本多样性之间的平衡。在采样期间尝试不同的 $w$ 值，用户可以在不重新训练模型的情况下了解这种特性。$w$ 的常用值范围为 3 到 15，具体取决于模型和任务。
5. **先进的结果：** CFG 在许多大型文本到图像模型（如 Stable Diffusion、Imagen、DALL-E 2）和其他条件生成任务中取得高质量结果方面发挥了重要作用。

### 考量

尽管功能强大，CFG 也有其需要注意的方面：

- **采样成本：** 主要的缺点是推理 (inference)期间计算成本增加。因为 CFG 需要在每个时间步评估模型两次（一次使用 $c$，一次使用 $\emptyset$），所以采样时间大约是仅使用条件预测 ($w=1$) 或无条件采样 ($w=0$) 的两倍。
- **引导尺度调整：** 寻找最佳引导尺度 $w$ 取决于具体应用，通常需要进行实验。极高的值可能导致过饱和、伪影或多样性丧失，而极低的值可能导致条件作用较弱。$w$ 的影响也可能与采样器选择和采样步数有关。
- **训练稳定性：** 尽管训练修改很简单，但确保模型有效学习条件和无条件模式可能需要注意丢弃概率 $p_{uncond}$ 和其他训练超参数 (parameter) (hyperparameter)。

总而言之，无分类器引导提供了一种有效的方法来控制扩散模型的输出。通过巧妙地训练一个单一模型来处理条件和无条件预测，并在推理期间组合这些预测，CFG 提供了强大的引导能力，而无需外部分类器的额外开销，使其成为现代生成建模中一项重要技术。下一节将考察实现和调整 CFG 尺度的实际方面。

## 参考资料

- [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) — Jonathan Ho, Tim Salimans (2022)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.2207.12598](https://doi.org/10.48550/arXiv.2207.12598)
  提出了无分类器引导，详细阐述了其理论、训练方法以及与分类器引导相比的优点。
- [Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding](https://arxiv.org/abs/2205.11487) — Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kamyar Seyed Ghasemipour, Burcu Karagol Ayan, S. Sara Mahdavi, Rapha Gontijo Lopes, Tim Salimans, Jonathan Ho, David J Fleet, Mohammad Norouzi (2022)
  Journal: arXiv preprint; DOI: [10.48550/arXiv.2205.11487](https://doi.org/10.48550/arXiv.2205.11487)
  展示了无分类器引导在大规模文本到图像生成中的强大表现，将其应用于准确的条件图像合成。
- [Hierarchical Text-Conditional Image Generation with CLIP Latents](https://arxiv.org/abs/2204.06125) — Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen (2022)
  Journal: arXiv preprint; Publisher: arXiv; DOI: [10.48550/arXiv.2204.06125](https://doi.org/10.48550/arXiv.2204.06125)
  将无分类器引导应用于更好的文本条件图像生成，展示了其在高质量生成模型中的作用。

---

[上一节](01-%E5%88%86%E7%B1%BB%E5%99%A8%E5%BC%95%E5%AF%BC%EF%BC%9A%E5%8E%9F%E7%90%86%E4%B8%8E%E5%AE%9E%E7%8E%B0.md) · [下一节](03-%E5%AE%9E%E6%96%BD%E4%B8%8E%E8%B0%83%E6%95%B4CFG%E5%B0%BA%E5%BA%A6.md)
