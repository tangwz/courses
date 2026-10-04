---
course: "fine-tuning-adapting-large-language-models"
chapter: "advanced-fine-tuning-strategies"
lesson: "mitigating-catastrophic-forgetting"
sourceId: 3053
sourceUrl: "https://apxml.com/zh/courses/fine-tuning-adapting-large-language-models/chapter-5-advanced-fine-tuning-strategies/mitigating-catastrophic-forgetting"
title: "减轻灾难性遗忘"
description: "诸如EWC（弹性权重整合）和参数隔离等方法，用于保留先前任务的知识。"
order: 3
plots: []
sourceHash: "002734bb205b8f03c9f9c1e47d4342627fe2205ebc91397d2c1e1e5ea6d6ad36"
sourceCorrections: []
---

当一个大型语言模型需要按顺序调整时，先针对任务A，再针对任务B，一个常见且主要的难题便会出现：**灾难性遗忘**。当模型优化其任务B的参数 (parameter)时，它经常会覆盖或损害在任务A训练期间获得的知识和能力。这是因为标准的微调 (fine-tuning)过程仅根据当前任务的目标函数调整模型的权重 (weight)，没有明确的机制来保持在过去任务上的表现。模型表现出很高的*可塑性*（学习新事物的能力），但缺乏足够的*稳定性*（保留旧知识的能力）。有效地管理这种权衡，对于构建能够持续学习和适应的模型来说是必要的。

让我们考察几种既有的策略来应对这种遗忘现象。

### 复习与重放策略

一种直观的办法是**复习**（也称作**重放**）。核心思路很简单：在对模型进行新任务B训练时，定期让它接触来自旧任务A的数据示例。通过在训练批次中混合新旧任务的数据，优化过程被促使找到在两者上都表现良好的参数 (parameter)配置。

- **精确重放：** 存储任务A的原始训练数据的一个子集，并在任务B训练期间将其包含在内。这种方法直接，但可能需要大量存储空间，特别是当处理许多以前的任务或大型数据集时。存储用户数据可能引发隐私问题。
- **生成式重放：** 不存储旧数据，训练一个生成模型（或者如果大型语言模型本身有能力，则使用它），以合成代表任务A的示例。这避免了存储原始数据，但增加了训练和管理生成器的复杂性，并且生成数据的质量可能无法完美匹配原始分布。

复习方法通常很有效，特别是当任务相关时。然而，由于有效数据集大小增加，它们会延长训练时间，并且需要仔细管理重放缓冲区（存储/生成多少旧数据以及使用频率）。

### 基于正则化 (regularization)的方法：弹性权重 (weight)整合 (EWC)

正则化技术旨在通过在新任务训练期间向损失函数 (loss function)添加惩罚项来防止遗忘。这个惩罚项会阻止对先前任务而言重要的参数 (parameter)发生大变化。**弹性权重整合 (EWC)** 是一个突出的例子。

EWC 通过评估参数对先前任务性能的贡献来识别重要参数。它使用**费雪信息矩阵 (FIM)**，$F$，作为这种重要性的代表。FIM 衡量损失的曲率；与高费雪信息相关的参数被认为更具决定性，因为对它们的微小改变会剧烈影响模型的输出和损失。

对于任务A之后再进行任务B的顺序训练，任务B训练的EWC损失函数如下所示：


$$
L_{total}(\theta) = L_{B}(\theta) + \frac{\lambda}{2} \sum_{i} F_{A,i} (\theta_i - \theta_{A,i}^*)^2
$$


符号说明：

- $L_{B}(\theta)$ 是给定当前参数$\theta$时新任务B的标准损失。
- $\lambda$ 是一个超参数 (hyperparameter)，用于控制正则化的强度（相对于学习任务B，记住任务A的重要性）。
- $F_{A,i}$ 是在任务A训练后计算得到的参数$\theta_i$的估计费雪信息。通常为了计算效率，只使用FIM的对角线元素。
- $\theta_{A,i}^*$ 是参数$\theta_i$在任务A上收敛后的值（任务A的“最优”参数）。
- 求和是针对所有被认为重要的参数$i$。

本质上，EWC 添加了一个二次惩罚项，如果参数$\theta_i$偏离其任务A的最优值（$\theta_{A,i}^*$）越远，该惩罚项就越大，并由其重要性$F_{A,i}$加权。

**EWC的优势：**

- 它不需要存储以前的任务数据，缓解了与复习相关的存储和隐私问题。

**EWC的劣势：**

- 需要计算并存储每个先前任务的费雪信息矩阵（或其对角线近似）和最优参数$\theta_A^*$，这仍然可能占用大量内存。
- 计算FIM的成本可能很高，尽管存在近似方法。
- 其有效性可能对超参数$\lambda$的选择敏感。
- 它假设任务足够明确，并且基于FIM的二次惩罚能够准确地保护任务特定的知识。

### 参数 (parameter)隔离方法

另一种有效策略，特别是适用于参数高效微调 (fine-tuning) (PEFT) 技术，是**参数隔离**。其思路是将不同的参数集分配给不同的任务，从而防止新任务的更新干扰对旧任务而言重要的参数。

考虑使用**适配器模块**（在第4章中讨论）。在对任务A进行微调时，您可以训练一组特定的适配器层，同时保持基础大型语言模型冻结。当随后调整至任务B时，您可以冻结任务A的适配器，并专门为任务B训练一组*新的*适配器层。在这两个微调阶段中，基础模型的参数保持不变。在推理 (inference)时，您根据任务激活相应的适配器集。

> 该图对比了标准顺序微调（其中任务B训练会覆盖任务A知识）与使用PEFT适配器进行缓解的情况。通过为每个任务训练独立的小参数集（适配器A和B），同时保持基础大型语言模型冻结，参数隔离避免了基础模型参数中的灾难性遗忘。

其他PEFT技术，如LoRA，也可以类似地使用，尽管将来自不同任务的LoRA适配器同时合并回基础模型并不直接。然而，将不同的LoRA适配器分开并按需加载可以实现所需的隔离。

**参数隔离（通过PEFT）的优势：**

- 为基础模型提供了强大的防遗忘保护。
- 在训练期间计算高效（只更新小部分参数集）。
- 存储高效（每个任务只需存储少量适配器权重 (weight)）。

**劣势：**

- 需要使用PEFT方法；如果需要完全微调以达到性能要求，则不适用。
- 总参数数量随每个新任务的增加而增加（尽管每个任务的适配器集很小）。
- 如果隔离过于严格，可能会限制任务间的积极知识迁移，因为任务主要依赖共享的冻结基础模型。

### 其他方法

其他方法包括：

- **无遗忘学习 (LwF)：** 在任务B训练时，使用在任务A上训练的模型 ($Model_A$) 的输出（logits或预测）作为软目标。损失函数 (loss function)包含一个项（通常是知识蒸馏 (knowledge distillation)损失），该项鼓励新模型 ($Model_B$) 在任务B的数据上产生与$Model_A$相似的输出，从而保留输出分布中反映的学习功能。
- **渐进式神经网络 (neural network) (PNNs)：** 对于每个新任务，都会添加一个新的网络“列”。以前列的参数 (parameter)被冻结，新列接收来自以前列的输入连接，允许明确的知识迁移而不改变旧参数。这种方法虽然有效，但会随着每个任务的增加而显著增加模型大小，并且与PEFT或正则化 (regularization)相比，较少应用于大型语言模型。

### 选择策略

缓解灾难性遗忘的最佳方法在很大程度上取决于具体的应用、限制和任务：

- **数据可用性和隐私：** 如果存储或访问旧任务数据是可行且允许的，复习通常是一个可靠的基线。
- **计算预算：** EWC增加了FIM计算的计算开销。PEFT方法在训练期间通常非常高效。复习会增加处理的数据量。
- **存储限制：** 复习需要存储数据样本。EWC需要存储最优参数 (parameter)和FIM矩阵。PEFT需要为每个任务存储少量适配器权重 (weight)。
- **任务相似性：** 如果任务高度不相似，强隔离（如PEFT）可能有利。如果任务相关，允许部分参数修改的方法（如EWC或复习）可能有助于积极的知识迁移。

在实践中，不仅评估当前任务的性能，还要评估一组具有代表性的先前任务的性能，这对于确认遗忘得到有效管理具有重要意义。有时，结合多种技术，例如将PEFT与少量复习结合使用，可以提供一个平衡的解决方案。理解这些缓解策略对于开发能够在动态环境中持续学习和适应，而不抛弃先前获得有价值知识的大型语言模型来说具有重要意义。

## 参考资料

- [A Comprehensive Survey of Continual Learning: Theory, Methodology and Applications](https://doi.org/10.1016/j.neunet.2019.01.012) — German I. Parisi, Ronald Kemker, Jose L. Part, Christopher Kanan, Stefan Wermter (2019)
  Journal: Neural Networks; Publisher: Elsevier; Volume: 113; Pages: 54-71; DOI: [10.1016/j.neunet.2019.01.012](https://doi.org/10.1016/j.neunet.2019.01.012)
  提供持续学习的广泛概述，包括对灾难性遗忘、排练策略和正则化技术的讨论。
- [Overcoming catastrophic forgetting in neural networks](https://www.pnas.org/doi/10.1073/pnas.1611835114) — James Kirkpatrick, Razvan Pascanu, Gabriel Jimenez Rezende, Adria Puigdomenech Badia, Oriol Vinyals, Fabio Hubert, Zachary Li, Peter Battaglia, Laurent Sifre, Evan Zoph, Martin Reichstein, Dean Hassabis, Iordanis Antonoglou, Charles Blundell (2017)
  Journal: Proceedings of the National Academy of Sciences; Publisher: National Academy of Sciences; Volume: 114; Pages: 3521-3526; DOI: [10.1073/pnas.1611835114](https://doi.org/10.1073/pnas.1611835114)
  介绍了弹性权重整合（EWC），一种通过识别并保护对先前任务重要的参数来减轻灾难性遗忘的正则化方法。
- [Learning without Forgetting](https://doi.org/10.1109/TPAMI.2017.2756813) — Zhizhong Li, Derek Hoiem (2018)
  Journal: IEEE Transactions on Pattern Analysis and Machine Intelligence; Publisher: IEEE; Volume: 40; Pages: 2931-2942; DOI: [10.1109/TPAMI.2017.2756813](https://doi.org/10.1109/TPAMI.2017.2756813)
  提出了“遗忘式学习”（LwF），在学习新任务时利用旧模型的知识蒸馏来保留知识。
- [Parameter-Efficient Transfer Learning for NLP](https://proceedings.mlr.press/v97/houlsby19a.html) — Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin De Laroussilhe, Andrea Gesmundo, Mona Attariyan, Sylvain Gelly (2019)
  Journal: Proceedings of the 36th International Conference on Machine Learning (ICML); Publisher: PMLR; Volume: 97; Pages: 2790-2799; DOI: [10.48550/arXiv.1902.00751](https://doi.org/10.48550/arXiv.1902.00751)
  介绍了适配器模块，这是一种参数高效的微调方法，可应用于参数隔离以减轻灾难性遗忘。
