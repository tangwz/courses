---
course: "synthetic-data-gans-diffusion"
chapter: "evaluating-synthetic-data-quality"
lesson: "evaluating-conditional-generation"
sourceId: 5724
sourceUrl: "https://apxml.com/zh/courses/synthetic-data-gans-diffusion/chapter-5-evaluating-synthetic-data-quality/evaluating-conditional-generation"
title: "评估条件生成模型"
description: "评估条件模型准确性和可控性的专属度量方法。"
order: 6
plots: ["plots/5724-0.json"]
sourceHash: "ba156b9a0846f6cd2fd059b0bbbbf92bd91906d375047de0be6cdb70de71b65b"
sourceCorrections: []
---

评估基于特定条件（例如类别标签、文本描述或其他指导性输入）生成输出的生成模型，需要特别的方法。尽管FID和IS等指标评估整体样本质量和多样性，但它们不直接衡量生成输出是否忠实符合所提供的条件。因此，评估条件模型涉及考察两个主要方面：

1. **条件一致性（保真度）：** 生成的样本是否准确反映了给定的条件？
2. **条件质量与多样性：** *针对特定条件*生成的样本是否高质量且多样？

接下来我们分析用于衡量这些方面的方法。

### 评估条件一致性

此处的目标是量化 (quantization)生成输出与输入条件的匹配程度。方法因条件类型而异。

#### 类别条件生成

对于以类别标签为条件（例如，生成特定犬种图像）的模型，常见做法是使用预训练 (pre-training)分类器。

1. **训练分类器：** 在用于训练生成模型的*真实*数据集上训练一个标准图像分类器。该分类器应在真实图像的保留测试集上达到高准确率。
2. **评估生成样本：** 为每个类别条件生成合成样本。将这些生成的样本及其预期标签输入预训练分类器。
3. **测量准确性：** 计算生成样本的分类准确率。高准确率表明生成器正在生成分类器识别为属于目标类别的图像，这表明条件一致性良好。

这在文献中有时被称为“分类准确率分数”（CAS）。一个变体是计算目标类别生成样本的预测类别分布与代表目标类别的一热向量 (vector)之间的KL散度。较低的KL散度表明更好的对齐 (alignment)。

然而，请留意潜在问题：

- 分类器可能偏向于从真实数据中学到的特征，这可能与生成器的输出分布不完全一致。
- 生成器可能学会生成“欺骗”用于评估的特定分类器的样本，即使这些样本并非真正代表该类别。使用多个不同的分类器可以帮助缓解这种情况。

#### 文本到图像生成

对于从文本提示生成图像的模型，评估条件一致性需要衡量文本与图像之间的语义对齐。最常用的指标是**CLIP分数**。

CLIP（对比语言-图像预训练）是OpenAI在一个包含大量图像-文本对的数据集上训练的模型。它学习了一个共享的嵌入 (embedding)空间，在该空间中，对应的图像和文本描述具有高余弦相似度。

计算CLIP分数的方法：

1. **生成图像：** 使用你的生成模型，根据文本提示 $t$ 创建一个图像。令生成的图像为 $I$。
2. **嵌入文本和图像：** 使用预训练的CLIP模型，获取 $t$ 的文本嵌入 $e_t$ 和 $I$ 的图像嵌入 $e_I$。
3. **计算余弦相似度：** 这一对的CLIP分数是嵌入之间的余弦相似度：
   
   $$
   \text{CLIP 分数}(t, I) = \frac{e_t \cdot e_I}{\|e_t\| \|e_I\|}
   $$
   
4. **平均：** 通常，CLIP分数是根据来自基准数据集（如MS-COCO字幕）的大量生成文本-图像对进行平均计算的。

较高的平均CLIP分数表示生成的图像与其对应文本提示之间更好的对齐。尽管被广泛使用，CLIP本身有其训练数据带来的局限和偏差，这可能会影响分数。

#### 其他条件输入

对于其他类型的条件（例如，基于分割图生成图像，基于参考图像进行风格迁移），评估常依赖于特定领域的度量方法。

- **分割图到图像：** 你可以衡量输入分割图与使用预训练分割模型从生成的图像中预测出的分割图之间的交并比（IoU）。
- **风格迁移：** 度量方法可能涉及比较生成图像与风格参考之间的风格统计（如格拉姆矩阵），或比较生成图像内容与原始内容图像的感知相似度度量。

### 评估条件质量与多样性

除了检查输出是否与条件匹配外，我们还需要确保每个条件*内部*的质量和多样性令人满意。模型可能为某一类别生成出色的图像，但为另一类别生成较差的图像，或者它可能仅在某些条件下出现模式崩溃。

标准评估指标，如FID、精确度、召回率和KID，可以调整以进行条件评估：

1. **按条件计算：** 为*每个不同的条件*（例如，每个类别标签）生成足够数量的样本。
2. **子集评估：** 通过比较特定条件（例如“猫”图像）下生成样本的分布与仅属于该条件的*真实*样本的分布，来计算像FID这样的指标。

例如，你可以计算：

- FID(真实“猫”图像，生成“猫”图像)
- FID(真实“狗”图像，生成“狗”图像)
- ... 依此类推，适用于所有类别。

这提供了比单一全局FID分数更精细的视图。它可以显示模型在不同条件下表现是否不均衡，或是否遭受类内模式崩溃（特定类别的多样性不足）。



![按类别Fréchet起始距离 (FID)](plots/5724-0.json)



> 图表显示了在四种不同类别条件下生成的图像分别计算的FID分数。类别 C 的FID分数明显更高，这表明与其它类别相比，属于该特定类别的生成样本质量或多样性较低。

类似地，可以按条件计算精确度和召回率，以了解生成器是否覆盖了该条件内真实样本的多样性（召回率），以及生成样本是否对该条件而言是逼真的（精确度）。

### 结合度量方法

在实践中，评估条件生成模型通常需要报告多种度量方法的组合：

- 条件一致性指标（例如，类别条件的分类器准确率，文本到图像的CLIP分数）。
- 整体质量/多样性指标（例如，所有条件下的全局FID）。
- 条件质量/多样性指标（例如，平均每类别FID，或报告每类别FID的范围/方差）。

这种多方面的方法提供了对模型能力和不足更全面的理解，指导后续的开发和优化工作。请记住，选择正确的评估指标很大程度上取决于具体的任务和所涉条件的性质。

## 参考资料

- [GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](https://proceedings.neurips.cc/paper/2017/file/8a1d694707635ae041b6067b45ad9b8f-Paper.pdf) — Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, Günter Schmidhuber (2017)
  Journal: Advances in Neural Information Processing Systems; Volume: 30; DOI: [10.5591/978-1-57766-311-6.196](https://doi.org/10.5591/978-1-57766-311-6.196)
  提出了Fréchet Inception距离（FID），这是广泛用于评估生成模型样本质量和多样性的指标，常用于条件评估。
- [Learning Transferable Visual Models From Natural Language Supervision](http://proceedings.mlr.press/v139/radford21a/radford21a.pdf) — Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever (2021)
  Journal: Proceedings of the 38th International Conference on Machine Learning (ICML); Volume: 139; Pages: 8748-8763
  介绍了CLIP模型，这是文本到图像和图像到文本任务的基础，也是衡量文本到图像生成中条件一致性的CLIP分数的基础。
- [Generative Adversarial Networks: A Survey and Taxonomy](https://dl.acm.org/doi/10.1145/3459676) — Junxian Gui, Zhentao Liu, Jia-Wang Bian, Zi-Yi Dou, Jun-Yan He, Xun Gong, Xiao-Fei Zhang, Wen-Da Qiu, Yong-Wei De, Yi-Hang Shen, Ding-Han Shen, Jian-Jun Li, Zhi-Yan Liu, Li-Fu Zheng, Rui-Shan Liu, De-Wei Kong, Xiao-Jie Jin, Wei-Ming Li, Jing-Hao Zhou, Rui Xu (2022)
  Journal: ACM Computing Surveys; Volume: 54; Pages: 1-46; DOI: [10.1145/3459676](https://doi.org/10.1145/3459676)
  一项全面回顾生成对抗网络的调查，涵盖了其各个方面，包括专门的评估指标部分，为评估生成模型质量和多样性提供了广泛的背景，与条件模型相关。
- [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://doi.org/10.1007/978-3-319-46475-6_43) — Justin Johnson, Alexandre Alahi, Li Fei-Fei (2016)
  Journal: European Conference on Computer Vision – ECCV 2016: Amsterdam, The Netherlands, October 8–16, 2016, Proceedings, Part IV; Publisher: Springer, Cham; Pages: 694-711; DOI: [10.1007/978-3-319-46475-6_43](https://doi.org/10.1007/978-3-319-46475-6_43)
  提出了感知损失函数，利用预训练深度神经网络的特征来衡量图像相似度。这种方法适用于评估风格迁移或图像到图像转换等条件生成任务，其中结构和风格保真度是关键。
