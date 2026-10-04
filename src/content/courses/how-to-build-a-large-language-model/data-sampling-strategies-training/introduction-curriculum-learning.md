---
course: "how-to-build-a-large-language-model"
chapter: "data-sampling-strategies-training"
lesson: "introduction-curriculum-learning"
sourceId: 5993
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-9-data-sampling-strategies-training/introduction-curriculum-learning"
title: "课程学习简介"
description: "按结构化顺序（例如先易后难）呈现数据的理念。"
order: 4
plots: []
sourceHash: "c60b8cb80203ebe069019d5507f83529c51b2ac29765452cf2c445dcfadc4629"
sourceCorrections: []
---

虽然从多样化的数据混合中随机打乱和采样是一种常见做法，但它从一开始就将所有数据点视为信息量均等。思考一下人类是如何学习的：我们通常从更简单的知识点开始，逐步提升至更复杂的知识。我们不会在理解基础算术之前直接跳到高等微积分。课程学习（CL）将类似原则应用于训练机器学习 (machine learning)模型，包括大型语言模型。课程学习不是以纯随机顺序从整个数据集中呈现数据点，而是引入一种结构，通常在训练过程中从“简单”示例转向“困难”示例。

基本思想是，从简单示例开始可以帮助模型构建根本表示，并避免在训练早期陷入不良局部最小值。这种初步的立足点可能使模型后续更容易从更复杂或带噪声的数据中学习。在大型语言模型预训练 (pre-training)的背景下，“简单”与“困难”的定义可以有多种形式。

### 定义课程

对于大型语言模型而言，什么构成“简单”或“困难”的示例？这并非总是直截了当，但常见方法包括：

1. **序列长度：** 较短序列通常被认为更简单。课程可能从主要在较短文档或片段上训练开始，并随着训练进行逐步增加批次中允许的最大序列长度。这有助于模型首先学习局部依赖，然后处理长距离依赖。
2. **词汇或句法复杂性：** 使用更简单词汇或更基本句型的示例可以在早期优先处理。像Flesch-Kincaid可读性分数、解析树深度或词汇稀有度等指标潜在地可用于对示例进行排序，尽管这会增加大量预处理开销。
3. **数据质量或来源：** 如同在数据混合中讨论的，一些来源（例如，精心整理的百科全书、书籍）通常比其他来源（例如，原始网页抓取数据）更干净、更有结构。课程可能涉及主要从高质量来源开始训练，随后逐步引入更多噪声或更多样化的数据。这与数据源加权密切相关，但具有有意的时间推进。
4. **困惑度评分：** 可以使用一个更小、已有的语言模型对训练示例的困惑度进行评分。困惑度较低的示例（即，较简单模型更容易预测的那些）可以先引入。

### 课程学习的实现

实现课程学习需要修改数据加载或采样过程。采样器需要了解训练进度（例如，当前周期或步骤），并根据定义的课程计划选择数据，而不是从数据集中均匀采样。

一个简单方法可能涉及根据难度指标（如序列长度）将数据分桶，并控制在不同训练阶段从哪些桶中进行主动采样。

考虑一个通过自定义PyTorch `Sampler`实现的基础的基于长度的课程。此示例体现了核心逻辑，并非生产级实现。

```python
import torch
from torch.utils.data import Sampler
import numpy as np

class LengthBasedCurriculumSampler(Sampler):
    def __init__(self,
                 data_lengths,
                 batch_size,
                 start_percentile=0.1,
                 end_percentile=1.0,
                 total_steps=10000):
        """
        根据训练中序列长度百分位数递增的方式采样批次。

        Args:
            data_lengths (list or np.array): 每个数据样本的长度列表。
            batch_size (int): 每个批次的大小。
            start_percentile (float): 初始长度百分位阈值 (0.0 到 1.0)。
            end_percentile (float): 最终长度百分位阈值 (0.0 到 1.0)。
            total_steps (int): 课程学习进行的总训练步数。
        """
        self.data_lengths = np.array(data_lengths)
        self.indices = np.argsort(self.data_lengths) # 按长度排序的索引
        self.sorted_lengths = self.data_lengths[self.indices]
        self.batch_size = batch_size
        self.start_percentile = start_percentile
        self.end_percentile = end_percentile
        self.total_steps = total_steps
        self.current_step = 0

        self.num_samples = len(data_lengths)
        # 根据百分位数计算初始和最终索引
        self.start_idx = int(self.start_percentile * self.num_samples)
        self.final_max_idx = int(self.end_percentile * self.num_samples)

    def get_current_max_index(self):
        # 在总步数内线性增加允许的最大索引
        progress = min(1.0, self.current_step / self.total_steps)
        increase = progress * (self.final_max_idx - self.start_idx)
        current_max_idx = int(self.start_idx + increase)
        # 确保我们总是包含至少起始百分位数的数据
        return max(self.start_idx, current_max_idx)

    def __iter__(self):
        current_max_idx = self.get_current_max_index()
        # 符合条件的索引是达到当前最大长度阈值的索引
        eligible_indices = self.indices[:current_max_idx]

        if len(eligible_indices) < self.batch_size:
            # 处理符合条件的数据过小的情况（例如，早期步骤）
            # 可能会重复样本或使用更小的批次
            eligible_indices = np.random.choice(
                eligible_indices, size=self.batch_size, replace=True
            )
        else:
             # 为当前周期/步骤打乱符合条件的索引
            np.random.shuffle(eligible_indices)

        # 生成批次（简化的批处理逻辑）
        num_batches = 0
        for i in range(0, len(eligible_indices), self.batch_size):
            batch_indices = eligible_indices[i : i + self.batch_size]
            # 为简单起见，丢弃最后一个不完整的批次
            if len(batch_indices) == self.batch_size:
                yield batch_indices.tolist()
                num_batches += 1

        # 在生成此迭代的所有批次后增加步骤
        # 在真实的训练器中，步骤更新会发生在每个优化器步骤
        # 此简化版本在每次__iter__调用时增加一次
        # 大致的步骤增量
        self.current_step += num_batches

    def __len__(self):
        # 每个周期/迭代的估计批次数量
        current_max_idx = self.get_current_max_index()
        num_eligible = len(self.indices[:current_max_idx])
        return num_eligible // self.batch_size

# --- 使用示例 ---
# 假设 `dataset` 是您的PyTorch数据集对象
# 假设 `lengths` 是一个列表，包含 `dataset` 中每个项目的长度
# lengths = [len(item) for item in dataset] # 预计算长度
#
# total_training_steps = 50000 # 示例总步数
# batch_size = 32
#
# sampler = LengthBasedCurriculumSampler(
#     lengths, batch_size, total_steps=total_training_steps
# )
# dataloader = torch.utils.data.DataLoader(
#     dataset, batch_size=None, sampler=sampler # 采样器使用时 batch_size=None
# )
#
# # 训练循环将使用此数据加载器
# # for epoch in range(num_epochs):
# #     for batch in dataloader:
# #         # 训练步骤...
# #         # 如果需要，更新采样器的内部步骤，
# #         # 尽管此示例是每次迭代更新
```

此采样器在初始化时一次性按长度排序数据。在每次迭代（通常对应一个周期）中，它根据当前的训练进度（`current_step`）确定允许的最大数据索引。然后它打乱并生成仅包含达到该长度百分位数的数据点的批次。`get_current_max_index`函数定义了课程的节奏。

### 优点与挑战

课程学习的潜在优点包括：

- **更快收敛：** 通过最初专注于更简单的示例，模型可能在早期阶段更快收敛。
- **泛化能力提升：** 一些研究表明，课程学习可以使模型泛化能力更强，潜在地通过引导学习过程形成更有意义的表示。
- **训练稳定性：** 逐步引入复杂性可能有助于稳定训练，特别是对于复杂架构或有难度的数据集。

然而，课程学习也带来挑战：

- **定义“难度”：** 如前所述，自动且有效地量化 (quantization)难度并非易事。不佳的衡量标准可能导致有害的课程。
- **节奏函数：** 确定最优计划（如何快速增加难度）需要调整，并且可能依赖于数据集或模型。过慢的计划可能浪费计算资源，而过快的计划则可能抵消优势。
- **实现复杂性：** 与简单随机打乱相比，集成课程学习增加了数据加载和训练管道的复杂性。
- **遗忘风险：** 如果模型仅在训练后期看到困难示例，它可能会“遗忘”从早期呈现的简单示例中学到的模式（尽管残差连接和大型模型容量通常能减轻这种情况）。

虽然基于细粒度难度指标的显式复杂课程并非总是训练大型语言模型的默认设置（对于大型语言模型，复杂的*数据混合加权*常因其可扩展性和经验成功而被偏好），但课程学习的核心思想常指导这些混合如何设计和潜在排序。例如，一种多阶段训练过程，其中模型首先在更干净的数据上训练，然后再接触完整、带噪声的数据集，可视为一种粗粒度课程形式。了解课程学习的原则提供另一种工具，用于优化大型语言模型训练这一要求较高的过程。

## 参考资料

- [Curriculum Learning](https://api.semanticscholar.org/CorpusID:873046) — Yoshua Bengio, Jérôme Louradour, Ronan Collobert, Jason Weston (2009)
  Journal: Proceedings of the 26th Annual International Conference on Machine Learning; Publisher: ACM; Pages: 41-48; DOI: [10.1145/1553374.1553457](https://doi.org/10.1145/1553374.1553457)
  这篇基础论文介绍了课程学习的概念，提出从简单示例开始并逐渐增加难度，可以加速机器学习模型的收敛并改善泛化能力。
- [When and How to Apply Curriculum Learning](http://proceedings.mlr.press/v97/hacohen19a/hacohen19a.pdf) — Yair Hacohen, Daphna Weinshall (2019)
  Journal: International Conference on Machine Learning (ICML); Publisher: Proceedings of Machine Learning Research; Volume: 97; Pages: 2596-2604; DOI: [10.48550/arXiv.1904.00412](https://doi.org/10.48550/arXiv.1904.00412)
  研究了课程学习何时最有益的条件，并为设计有效的课程（包括难度指标和进度函数）提供了实用指导。
