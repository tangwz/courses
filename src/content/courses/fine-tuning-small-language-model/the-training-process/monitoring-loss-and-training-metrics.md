---
course: "fine-tuning-small-language-model"
chapter: "the-training-process"
lesson: "monitoring-loss-and-training-metrics"
sourceId: 8520
sourceUrl: "https://apxml.com/zh/courses/fine-tuning-small-language-model/chapter-5-the-training-process/monitoring-loss-and-training-metrics"
title: "监控损失与训练指标"
description: "使用日志工具跟踪训练损失和验证指标，实时评估模型学习情况。"
order: 4
plots: ["plots/8520-0.json"]
sourceHash: "5eaddc3e542c05e4ec829042d196a9034dda59003c62f0ff555fd53f781be5fe"
sourceCorrections: []
---

观察模型的学习过程是训练环节的核心部分。运行训练脚本而不观察其内部状态，就像是蒙着眼睛开车。你需要持续的反馈流来验证模型是否真正学会了目标任务，而不仅仅是背下了数据集。

微调 (fine-tuning)过程中的主要信号是损失函数 (loss function)。对于因果语言模型（Causal Language Models），这通常是交叉熵损失（Cross-Entropy Loss）。它衡量模型预测的标记 (token)概率与数据集中实际下一个标记之间的差异。损失越低，意味着模型的预测与预期文本越接近。

为了准确了解模型性能，我们会跟踪两个不同的损失值：训练损失和验证损失。训练损失是在模型正在学习的数据上计算的；验证损失则是在模型从未用于权重 (weight)更新的独立保留数据集上计算的。这两个指标之间的关系直接反映了模型对未见指令的泛化效果。



![训练损失 vs. 验证损失](plots/8520-0.json)



> 损失曲线显示了在第 600 步左右，模型从良性学习转变为死记硬背的状态。

解读这两条曲线之间的差距是机器学习 (machine learning)中的必备技能。在训练过程中，你通常会观察到以下三个阶段之一：

- **欠拟合（Underfitting）：** 训练损失和验证损失都保持在高位或下降非常缓慢。这表明学习率可能太低，或者模型架构缺乏学习该任务的能力。
- **良性学习（Healthy Learning）：** 两个指标都稳步下降并稳定在一个较低的值。模型正在学习通用模式，这些模式同样适用于训练集和评估数据集。
- **过拟合（Overfitting）：** 训练损失继续下降，而验证损失开始上升。模型正在死记硬背训练数据的具体措辞，失去了响应未见指令的能力。验证损失开始向上偏离的时刻，通常就是你应该停止训练并恢复到之前检查点的时刻。

虽然损失提供了原始的优化信号，但语言建模通常依靠困惑度（Perplexity）来提高人类可读性。困惑度是交叉熵损失的指数形式：

$ext{PPL} = e^{L}$

其中 $L$ 是交叉熵损失。困惑度越低，表示模型对评估数据的“惊讶”程度越低。如果你的验证损失是 1.72，那么验证困惑度大约是 5.58。跟踪困惑度可以直观地感受模型生成文本时的自信程度。

要实现这种监控，你必须在训练脚本中配置日志参数 (parameter)。当在 `TrainingArguments` 中提供正确参数时，Hugging Face 的 `Trainer` 类会自动处理这些工作。你需要通过评估策略指定计算这些指标的频率。

```python
from transformers import TrainingArguments

# 配置训练参数
training_args = TrainingArguments(
    output_dir="./slm-outputs",
    eval_strategy="steps",        # 评估策略设为按步数
    eval_steps=100,              # 每 100 步评估一次
    logging_strategy="steps",    # 日志策略设为按步数
    logging_steps=50,            # 每 50 步记录一次日志
    report_to="tensorboard",     # 上报至 TensorBoard
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4
)
```

在此配置中，`logging_steps=50` 指示训练器每 50 步记录一次训练损失。`eval_steps=100` 参数规定训练器每 100 步会暂停优化，在验证数据集上运行一次完整的正向传播并计算验证损失。

日志记录过于频繁会因为不断读写数据而减慢训练速度；记录太少则可能错过过拟合开始的确切时间点。通用的做法是每个周期（epoch）计算 10 到 20 次评估指标。

请注意配置中的 `report_to` 参数。在长时间的训练运行中，将指标输出到标准终端会变得难以阅读。Hugging Face 原生支持 TensorBoard 和 Weights & Biases 等外部跟踪工具。这些工具可以捕获数值日志并渲染出交互式仪表盘，让你能够实时观察曲线变化。

通过持续观察这些指标，你将从被动地运行脚本转变为主动管理模型优化过程。这种监控设置能直接告诉你何时该停止训练，并确保你在测试生成质量之前，提取出微调权重中性能最出色的版本。

## 参考资料

- [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky, James H. Martin (2026)
  Publisher: Pearson
  这是一本基础教科书，解释了语言建模中交叉熵损失与困惑度之间的数学关系。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: 108-118
  第 5.2 节提供了理解机器学习模型中泛化、欠拟合和过拟合的理论框架。
- [Hugging Face Transformers Documentation: Trainer](https://huggingface.co/docs/transformers/main_classes/trainer) — Hugging Face (2024)
  Publisher: Hugging Face
  TrainingArguments 类的官方技术文档，详细介绍了如何配置日志记录和评估策略。
- [CS224N: Natural Language Processing with Deep Learning](https://web.stanford.edu/class/cs224n/) — Christopher Manning (2026)
  Publisher: Stanford University
  涵盖训练大语言模型的实践操作以及如何解读性能指标的课程资料。
