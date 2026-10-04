---
course: "intro-to-multimodal-ai"
chapter: "components-multimodal-ai-models"
lesson: "basic-evaluation-metrics-multimodal"
sourceId: 6530
sourceUrl: "https://apxml.com/zh/courses/intro-to-multimodal-ai/chapter-4-components-multimodal-ai-models/basic-evaluation-metrics-multimodal"
title: "多模态输出的基础评估指标"
description: "了解用于评估多模态AI模型表现的基础指标。"
order: 7
plots: []
sourceHash: "1bcaa6d97ff537d6e6fb04f908c74ca73be972bcef046830a86a72b39f61dcf3"
sourceCorrections: []
---

训练好多模态 (multimodal)AI模型后，如何判断它是否表现良好？损失函数 (loss function)通过告知模型预测的偏差程度来引导训练过程。虽然损失函数对训练不可或缺，但它们给出的分数通常在表现方面人类不易解读。这时，评估指标就派上用场了。评估指标提供标准化、易懂的分数，帮助我们衡量和比较模型在特定任务上的表现。它们回答了这个问题：“这个模型在预定功能上表现如何？”

不同的多模态任务会产生不同类型的输出（例如文本描述、问题答案、类别标签），因此我们需要针对每种情况定制不同的指标。我们来看一下常见多模态应用的一些基础指标。

### 图像字幕生成指标

图像字幕生成模型为给定图像生成文本描述。评估这些字幕需要将机器生成的文本与人类编写的参考字幕进行比较。

**BLEU（双语评估替补）**
BLEU是一种广泛使用的指标，用于评估机器生成的文本，包括图像字幕。它衡量候选字幕（来自模型）与一个或多个参考字幕（由人类编写）的相似程度。
核心思想是统计候选字幕和参考字幕之间匹配的词序列，称为n-gram。

- **1-gram（单元）**：单个词。
- **2-gram（二元）**：两个词的序列。
- **3-gram（三元）**：三个词的序列，依此类推。

匹配的n-gram数量越多（特别是更长的n-gram），表明相似性越好。BLEU分数通常在0到1（或0到100）之间，分数越高表示模型的字幕越接近人类参考文本。例如，如果模型生成“a cat sits on a mat”，而参考文本是“a cat is on the mat”，它们共享几个单元（“a”、“cat”、“on”、“mat”）和二元（“a cat”、“on a”、“a mat”）。

尽管BLEU很流行，但它主要关注精确度（模型字幕中有多少词出现在参考文本中），并施加简洁惩罚，以避免字幕过短。它不能完全捕捉语义或语法正确性。

**其他字幕生成指标**
研究人员开发了其他指标来解决BLEU的一些局限性：

- **METEOR（带显式排序的翻译评估指标）**：考虑同义词和词干还原，使其对含义更敏感。它根据机器生成字幕和参考字幕之间的对齐 (alignment)情况计算分数。
- **CIDEr（基于共识的图像描述评估）**：此指标旨在衡量字幕与一组人类编写字幕的共识匹配程度。它高度重视人类常用n-gram来描述图像，假设好的字幕应体现人类理解。

对于初学者来说，主要认识是这些指标提供了量化 (quantization)评估字幕质量的方法，即通过与人类标准进行比较。

### 视觉问答（VQA）指标

在VQA中，模型回答有关图像的问题。答案类型可以不同（例如，“是/否”、数字、短语）。

**准确率**
对于许多VQA任务，尤其是那些具有简单、事实性答案的任务，准确率是一个直接且有效的指标。其计算方式如下：


$$
\text{准确率} = \frac{\text{正确回答的问题数量}}{\text{问题总数}}
$$


例如，如果一个模型在1000个问题中正确回答了800个，那么它的准确率为80%。

有时会针对不同类型的问题分别报告准确率：

- **是/否问题**：“图像中有一只狗吗？”
- **数字问题**：“有多少辆车？”
- **其他/开放式问题**：“伞是什么颜色的？”

对于开放式答案，简单的字符串匹配可能过于严格。例如，如果真实答案是“红色”而模型说是“深红色”，严格的准确率会将其视为错误。更高级的VQA指标（如衡量语义相似度的Wu-Palmer相似度或WUPS）可以处理此类变体，但基础准确率是一个不错的起点。

### 多模态 (multimodal)情感分析指标

多模态情感分析旨在判断结合了文本、音频和视频等模态的数据中所表达的情感（例如，积极、消极、中性）。由于这通常是一个分类任务，因此标准的分类指标适用。

我们假设进行二元情感分类（积极与消极）。我们可以定义：

- **真阳性（TP）**：模型在实际为积极情感时正确预测为积极情感。
- **真阴性（TN）**：模型在实际为消极情感时正确预测为消极情感。
- **假阳性（FP）**：模型在实际为消极情感时错误预测为积极情感（第一类错误）。
- **假阴性（FN）**：模型在实际为积极情感时错误预测为消极情感（第二类错误）。

以下图表展示了这些术语：

> 一个图表，展示了二元分类任务中的真阳性（TP）、假阳性（FP）、假阴性（FN）和真阴性（TN）。

基于这些，我们可以计算：

- **准确率**：整体正确性。
  
  $$
  \text{准确率} = \frac{TP + TN}{TP + TN + FP + FN}
  $$
  
- **精确率（阳性预测值）**：在所有预测为阳性的实例中，有多少是实际阳性的？在假阳性成本很高时很重要。
  
  $$
  \text{精确率} = \frac{TP}{TP + FP}
  $$
  
- **召回率（敏感度，真阳性率）**：在所有实际阳性实例中，模型正确识别了多少？在假阴性成本很高时很重要。
  
  $$
  \text{召回率} = \frac{TP}{TP + FN}
  $$
  
- **F1-分数**：精确率和召回率的调和平均值。它提供一个平衡两者的单一分数。当您需要在精确率和召回率之间取得平衡时，尤其是在类别分布不均匀的情况下，它会很有用。
  
  $$
  \text{F1-分数} = 2 \times \frac{\text{精确率} \times \text{召回率}}{\text{精确率} + \text{召回率}}
  $$
  

这些指标可以扩展到多类别情感分析（例如，积极、消极、中性），使用宏平均（对每个类别的指标进行平均）或微平均（在计算指标前对全局计数进行聚合）等技术。

### 评估生成图像（例如，文本到图像合成）

评估AI生成的图像，例如文本到图像合成中的图像，比评估文本或简单分类更复杂。虽然存在自动化指标（如Inception Score或Fréchet Inception Distance，它们比较生成图像与真实图像的统计属性），但它们可能难以理解，并且不总是与人类对质量的感知相符。

对于入门学习者而言，需要了解的是，人工评估在此处扮演着非常重要的角色。人类通常评估：

1. **图像质量**：图像是否清晰、逼真、没有奇怪的伪影，以及是否美观？
2. **文本-图像对齐 (alignment)/相关性**：生成的图像是否准确、全面地代表了输入文本提示？
3. **多样性**：如果为同一提示生成多幅图像，它们是否多样化，而不仅仅是彼此的细微复制品？

自动化指标是一个活跃的研究方向，但目前，常结合人类判断和现有量化 (quantization)分数进行评估。

### 人工评估的不可或缺作用

虽然自动化指标快速、可扩展，并提供客观的比较数据，但它们往往未能全面体现多模态 (multimodal)AI系统的表现。它们可能无法全面评估：

- **流畅性和连贯性**：生成的文本是否自然易读？
- **常识**：模型的输出是否有意义？
- **细微之处**：模型能否理解或生成不易察觉的含义或情感？
- **偏见与公平性**：模型是否表现出不良偏见？
- **用户体验**：模型在实际应用中的输出是否有帮助或吸引人？

这时，**人工评估**变得必不可少。在许多情况下，特别是对于生成任务或涉及丰富理解的任务，会要求人类对模型输出进行评分或比较。这提供了定性反馈，补充自动化分数，并能提供模型优势和劣势的更深刻见解。

### 选择评估指标

评估指标的选择在很大程度上取决于：

- **具体任务**：图像字幕生成所需的指标与情感分析不同。
- **输出的性质**：文本、类别、图像，或它们的组合。
- **应用目标**：精确率是否比召回率更重要？语义相似度是否比精确的词语匹配更重要？

作为初学者，专注于每种任务类型最常用且易于理解的指标是一个不错的起点。准确率、BLEU以及标准分类指标（精确率、召回率、F1-分数）涵盖了许多基本场景。理解这些指标衡量什么以及它们的局限性，是构建和改进多模态 (multimodal)AI系统的一个重要步骤。这些评估结果将引导您优化模型架构、训练过程，甚至您使用的数据。

## 参考资料

- [BLEU: a Method for Automatic Evaluation of Machine Translation](https://aclanthology.org/P02-1040/) — Kishore Papineni, Salim Roukos, Todd Ward, Wei-Jing Zhu (2002)
  Journal: Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics; Publisher: Association for Computational Linguistics; Pages: 311-318; DOI: [10.3115/1073083.1073135](https://doi.org/10.3115/1073083.1073135)
  介绍了广泛用于评估机器生成文本的BLEU指标，它是评估图像字幕模型的核心组成部分。
- [CIDEr: Consensus-based Image Description Evaluation](https://openaccess.thecvf.com/content_cvpr_2015/html/Vedantam_CIDEr_Consensus-based_Image_2015_CVPR_paper.html) — Ramakrishna Vedantam, C. Lawrence Zitnick, Devi Parikh (2015)
  Journal: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR); Pages: 4566-4575; DOI: [10.1109/CVPR.2015.7298926](https://doi.org/10.1109/CVPR.2015.7298926)
  提出了CIDEr，一种专门设计用于通过衡量与人类描述的一致性来评估图像字幕的指标，与本节内容直接相关。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press
  一本基础性教科书，涵盖了深度学习的各个方面，包括对准确率、精确率、召回率和F1分数等常见分类评估指标的解释。
- [VQA: Visual Question Answering](https://openaccess.thecvf.com/content_iccv_2015/papers/Antol_VQA_Visual_Question_2015_ICCV_paper.pdf) — Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C. Lawrence Zitnick, Devi Parikh (2015)
  Journal: Proceedings of the IEEE International Conference on Computer Vision (ICCV); Publisher: IEEE; Pages: 2425-2433; DOI: [10.1109/ICCV.2015.279](https://doi.org/10.1109/ICCV.2015.279)
  介绍了基准VQA数据集和任务，为VQA模型的评估提供了背景。
