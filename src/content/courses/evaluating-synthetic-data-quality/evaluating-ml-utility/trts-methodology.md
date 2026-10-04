---
course: "evaluating-synthetic-data-quality"
chapter: "evaluating-ml-utility"
lesson: "trts-methodology"
sourceId: 5665
sourceUrl: "https://apxml.com/zh/courses/evaluating-synthetic-data-quality/chapter-3-evaluating-ml-utility/trts-methodology"
title: "训练-真实-测试-合成 (TRTS) 方法"
description: "理解和实施 TRTS 评估框架。"
order: 2
plots: []
sourceHash: "f7a8550e5cbe68b5f59ccb26afadc408184bd827427cb27858211fcf53db7fdf"
sourceCorrections: []
---

训练-合成-测试-真实 (TSTR) 方法直接衡量合成数据是否可以替代真实数据进行模型训练。相比之下，**训练-真实-测试-合成 (TRTS)** 评估方法提供了另一种评估合成数据质量的视角。TRTS 评估合成数据与真实数据*相似程度*，这是从使用原始数据集训练的模型的角度来看的，而不是评估合成数据的*训练效用*。

TRTS 流程本质上是 TSTR 的逆向操作：

1. **在真实数据上训练**: 使用原始的真实训练数据集 ($Real_{train}$) 训练一个选定的机器学习 (machine learning)模型 (我们称之为 $Model_R$)。
2. **在合成数据上测试**: 使用合成生成的数据集 ($Synthetic_{gen}$) 作为测试集，评估 $Model_R$ 的性能。
3. **(可选) 在真实保留数据上测试**: 为了进行比较，在真实数据中保留的一部分 ($Real_{test}$) 上评估 $Model_R$。

以下是说明此流程的图表：

> TRTS 评估过程：模型仅在真实数据上进行训练，随后在合成数据集上进行测试。真实测试集上的性能通常作为基准。

### 解读 TRTS 结果

从在 $Synthetic_{gen}$ 上测试 $Model_R$ 所得的性能指标 (例如，准确率、AUC、F1 分数) 有助于理解合成数据的特性：

- **合成数据上的高性能**: 如果 $Model_R$ 在 $Synthetic_{gen}$ 上表现良好，这表明从 $Real_{train}$ 学到的模式、关系和决策边界也在合成数据中存在且可识别。合成数据有效地模仿了 $Model_R$ 所学到的分布。
- **合成数据上的低性能**: 反之，性能不佳表明 $Model_R$ 尽管在训练它的真实数据上表现熟练，但难以泛化到合成数据。这暗示着分布不匹配；合成数据缺乏或错误地代表了 $Model_R$ 在真实数据中识别为重要的模式。

### TRTS 与 TSTR 及基准的比较

比较 TRTS 分数 (即 $Model_R$ 在 $Synthetic_{gen}$ 上的性能) 与同一模型在真实测试集 ($Real_{test}$) 上的性能，会得到有用的信息：

- **TRTS 分数 ≈ 真实测试分数**: 这通常是一个理想的结果。它表明合成数据很好地反映了真实数据的特性，使得在真实数据上训练的模型在这两者上表现相似。合成数据看起来具有代表性。
- **TRTS 分数 > 真实测试分数**: 这种情况初看起来可能积极，但需要仔细审视。这可能意味着合成数据与用于训练 $Model_R$ 的特定 $Real_{train}$ 数据集*过于相似*。如果生成模型过拟合 (overfitting)或记忆了其训练数据的某些方面，就可能发生这种情况。尽管对训练集的保真度很高，合成数据可能缺乏未见真实数据 ($Real_{test}$) 中存在的 다양性或泛化能力。
- **TRTS 分数 < 真实测试分数**: 这是一个常见结果，表明合成数据未能捕获真实数据的所有细节或精确分布，导致在真实数据上训练的模型在其上表现更差。

TRTS 通过回答一个不同的问题来补充 TSTR。

- **TSTR 提问**: "我能否*仅*使用合成数据来训练一个有用模型？" (侧重：可替代性)
- **TRTS 提问**: "从在真实数据上训练的模型的角度来看，合成数据在统计上与真实数据相似吗？" (侧重：代表性)

较高的 TSTR 分数表明模型训练的实际效用。较高的 TRTS 分数表明生成器成功地从真实训练集学习到模式，但这并不能保证 TSTR 的效用，特别是如果高分是由于生成过程中的过拟合。理想情况下，您希望合成数据在 TSTR 和 TRTS 评估中相对于真实数据基准都表现良好，这表明在效用和保真度之间取得了良好平衡，且没有简单记忆。

与 TSTR 类似，下游模型 ($Model_R$) 和评估指标的选择可能会影响 TRTS 结果。使用为实际下游任务规划的相同模型架构来运行 TRTS 通常是有益的。

## 参考资料

- [CTGAN: Effective and Versatile Conditional GAN for Tabular Data Generation](https://proceedings.neurips.cc/paper_files/paper/2019/hash/2065a32b2e8a156108ad7611a2f16a04-Abstract.html) — Lei Xu, Maria Skoularidou, Alfredo Cuesta-Infante, Kalyan Veeramachaneni (2019)
  Journal: Advances in Neural Information Processing Systems; Publisher: NeurIPS; Volume: 32
  一篇表格合成数据生成的奠基性论文，其中包含对机器学习效用和忠实度的全面评估，提供了真实数据训练模型如何与合成数据交互的实际示例。
