---
course: "llm-compression-acceleration"
chapter: "knowledge-distillation-large-models"
lesson: "practice-distilling-generative-llm"
sourceId: 4377
sourceUrl: "https://apxml.com/zh/courses/llm-compression-acceleration/chapter-4-knowledge-distillation-large-models/practice-distilling-generative-llm"
title: "动手实践：生成式大型语言模型知识蒸馏"
description: "构建知识蒸馏流程，以大型生成式教师语言模型为基础，训练一个较小的学生模型。"
order: 8
plots: ["plots/4377-0.json"]
sourceHash: "16de52119758cc616c00129d5bea719df845660fadd32c13b0cecfbc9da08082"
sourceCorrections: []
---

基于本章前面介绍的原理，我们现在转向实际操作，进行知识蒸馏 (knowledge distillation)，将大型生成式语言模型（教师模型）的知识迁移到一个更小、更高效的模型（学生模型）中。目标是创建一个学生模型，使其在明显更小、更快的同时，仍能保留教师模型的大部分生成能力。本节提供一个分步指南，侧重于与生成模型相关的实施细节和评估方法。

### 准备工作：模型、数据和工具

在启动蒸馏过程前，需要进行细致的准备。

1. **教师模型选择：** 选择一个预训练 (pre-training)的生成式大型语言模型作为教师模型。这可以是像 `GPT-3.5`、`LLaMA-7B` 这样的模型，甚至是针对特定风格或领域微调 (fine-tuning)的版本。对于本实践指南，我们假设正在从 `TeacherLM-7B`（70亿参数 (parameter)）模型蒸馏知识。获取教师模型需要加载其权重 (weight)和架构，这通常使用像 Hugging Face Transformers 这样的库。
2. **学生模型架构：** 定义或选择学生模型的架构。它应该明显小于教师模型，例如 `StudentLM-1B`（10亿参数）。重要的一点是，学生模型的架构应与教师模型的输出格式兼容（例如，两者都在相同的词汇表 (vocabulary)上生成 logits）。虽然层数和隐藏维度会有所不同，但核心生成机制（例如，Transformer 解码器）应该相似。
3. **蒸馏数据集：** 数据的选择非常重要。通常，用于教师模型的原始预训练数据集是有效的。或者，也可以使用包含学生模型所需能力的大量、多样的提示或指令数据集。目的是数据应能引出你希望从教师模型迁移的知识。如果仅依赖教师模型的输出，真实标签并非严格必要，但如果可用，也可以加入。
4. **环境配置：** 确保已安装所需的库，主要是 PyTorch 或 TensorFlow，以及 Hugging Face 的 `transformers`、`datasets`，并可能包括 `accelerate` 以实现高效训练。

```python
# 配置
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

# 加载教师模型 (确保其处于评估模式且不需要梯度)
teacher_model_name = "path/to/large/teacher/model"
teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_name)
teacher_model.eval()
for param in teacher_model.parameters():
    param.requires_grad = False

# 加载或定义学生模型
student_model_name = "path/to/smaller/student/config_or_model" # 或定义架构
student_tokenizer = AutoTokenizer.from_pretrained(student_model_name) # 通常与教师模型相同
student_model = AutoModelForCausalLM.from_pretrained(student_model_name) # 或从配置初始化

# 加载数据集
# dataset = load_dataset(...)
```

### 设计蒸馏流程

知识蒸馏 (knowledge distillation)的根本在于指导学生模型训练的损失函数 (loss function)。

#### 蒸馏损失

如前面介绍，一种常见方法是将标准语言建模损失（如果存在真实目标）与知识蒸馏损失相结合，以促使学生模型模仿教师模型的输出分布。

1. **软标签 (KL 散度)：** 主要的知识蒸馏损失旨在最小化教师模型和学生模型在词汇表 (vocabulary)上的概率分布之间的 KL 散度。应用温度缩放来软化分布，防止模型对单个标记 (token)过度自信，并提供更丰富的监督信号。

   单个标记预测的损失为：

   
   $$
   L_{KD} = T^2 \cdot D_{KL}(\sigma(z_S / T) || \sigma(z_T / T))
   $$
   

   其中 $z_S$ 和 $z_T$ 分别是学生模型和教师模型生成的 logits，$T$ 是温度（通常 $T > 1$），$\sigma$ 表示 softmax 函数。将此损失在序列长度和批次上平均，得到最终的知识蒸馏损失部分。
2. **硬标签 (交叉熵)：** 如果蒸馏数据集包含真实下一个标记（例如，在继续预训练 (pre-training)或微调 (fine-tuning)期间），标准交叉熵损失 ($L_{CE}$) 可以与知识蒸馏损失一同使用。这使学生模型立足于实际任务数据。

   
   $$
   L_{CE} = -\sum_{i} y_i \log(\sigma(z_S)_i)
   $$
   

   其中 $y_i$ 是第 $i$ 个标记的独热编码真实标签。
3. **组合损失：** 最终损失函数通常是交叉熵损失（如果使用）和 KL 散度损失的加权和：

   
   $$
   L_{Total} = (1 - \alpha) L_{CE} + \alpha L_{KD}
   $$
   

   这里，$\alpha$ 是一个超参数 (parameter) (hyperparameter)（介于 0 和 1 之间），用于平衡真实标签和教师模型软标签的影响。选择合适的 $\alpha$ 和温度 $T$ 通常需要进行实验。

#### 匹配中间表示

为了更深层的知识迁移，特别是在架构存在差异时，匹配中间表示可以带来益处。

1. **隐藏状态：** 促使学生模型在特定层的隐藏状态 ($h_S$) 接近教师模型在对应层的隐藏状态 ($h_T$)。由于维度可能不同，可能需要一个线性投影层 ($W_p$) 将教师模型的状态映射到学生模型的维度。损失通常是均方误差 (MSE)：
   
   $$
   L_{Hidden} = \sum_{l \in L_{match}} \beta_l \cdot MSE(h_{S,l}, W_p(h_{T,l}))
   $$
   
   其中 $L_{match}$ 是匹配层集合，$\beta_l$ 是加权因子。
2. **注意力模式：** 类似地，可以促使学生模型的注意力图谱 ($A_S$) 模仿教师模型的注意力图谱 ($A_T$)，可能在池化或投影之后。

加入这些中间损失会增加复杂性，但可以明显提高学生模型对教师模型学到模式的理解。总损失变为 $L_{CE}$、$L_{KD}$ 以及任何中间匹配损失的加权和。

### 训练流程

训练循环需要修改以适应教师模型和自定义损失函数 (loss function)。

#### 前向传播机制

对于每个输入批次：

1. **教师模型前向传播：** 将输入标记 (token)通过教师模型，获取其 logits ($z_T$) 以及可能的中间隐藏状态 ($h_T$) 或注意力图谱 ($A_T$)。此步骤*不*计算教师模型的梯度。
2. **学生模型前向传播：** 将相同的输入标记通过学生模型，获取其 logits ($z_S$)、隐藏状态 ($h_S$) 和注意力图谱 ($A_S$)。此步骤*会*跟踪学生模型参数 (parameter)的梯度。

#### 反向传播 (backpropagation)与优化

1. **计算损失：** 使用两次前向传播的输出（$z_S, z_T$，以及可能的 $h_S, h_T$ 等）和真实标签（如果适用）来计算组合蒸馏损失 ($L_{Total}$)。
2. **反向传播：** 根据 $L_{Total}$ 执行反向传播。重要的一点是，梯度应只回传通过学生模型的参数。教师模型保持冻结。
3. **优化器步骤：** 使用优化器（例如 AdamW）更新学生模型的权重 (weight)。

#### 代码结构（使用 Hugging Face Trainer）

虽然自定义训练循环提供最大灵活性，但 Hugging Face `Trainer` 可以子类化以加入蒸馏功能。

```python
# Trainer 的子类
from transformers import Trainer
import torch.nn.functional as F
import torch.nn as nn

class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, temperature=2.0, alpha=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.teacher_model.to(self.args.device) # 确保教师模型在同一设备上
        self.temperature = temperature
        self.alpha = alpha
        # 如果需要匹配不同大小的隐藏状态，可能在此处添加投影层

    def compute_loss(self, model, inputs, return_outputs=False):
        # 学生模型前向传播 (Trainer 的标准行为)
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits

        # 如果提供了标签，计算标准交叉熵损失
        if "labels" in inputs:
            loss_ce = student_outputs.loss # Trainer 默认计算此项
        else:
            loss_ce = 0.0 # 或适当处理没有标签的情况

        # 教师模型前向传播 (无梯度)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits

        # 计算知识蒸馏损失 (确保因果语言模型的正确切片/对齐)
        # 通常比较预测标记的 logits (移动 logits 和标签)
        vocab_size = student_logits.size(-1)
        student_log_probs = F.log_softmax(student_logits[:, :-1, :] / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits[:, :-1, :] / self.temperature, dim=-1)

        # KLDivLoss 期望 log-probs 作为输入，probs 作为目标
        loss_kd = nn.KLDivLoss(reduction="batchmean")(student_log_probs, teacher_probs) * (self.temperature ** 2)

        # 组合损失
        # 如果使用标签：
        # loss = (1.0 - self.alpha) * loss_ce + self.alpha * loss_kd
        # 如果不使用标签 (纯粹从教师信号蒸馏)：
        loss = loss_kd # 如果需要，调整 alpha 逻辑

        # 如果适用，在此处添加隐藏状态匹配损失
        # loss_hidden = compute_hidden_state_loss(...)
        # loss += beta * loss_hidden

        return (loss, student_outputs) if return_outputs else loss

# 设置训练参数
# training_args = TrainingArguments(...)

# 实例化 DistillationTrainer
# trainer = DistillationTrainer(
#     model=student_model,
#     teacher_model=teacher_model,
#     args=training_args,
#     train_dataset=tokenized_dataset["train"],
#     eval_dataset=tokenized_dataset["validation"],
#     tokenizer=student_tokenizer,
#     # data_collator=... # 对因果语言模型的填充和标签移动很重要
#     temperature=2.0,
#     alpha=0.5,
# )

# 开始训练
# trainer.train()
```

*注意：* 这段代码是。在损失计算中实现因果语言模型的标签移动和正确处理填充需要细致注意细节。

### 评估蒸馏后的生成模型

全面评估对于确认蒸馏过程的成功非常重要。

#### 定量指标

1. **困惑度 (PPL)：** 衡量模型在保留测试集上的流畅性和预测准确性。较低的困惑度通常表示更好的语言建模能力。比较学生模型的困惑度与教师模型的困惑度，以及在相同数据上从零开始训练的学生模型（未经知识蒸馏 (knowledge distillation)）的困惑度。
2. **下游任务表现：** 评估蒸馏模型在其目标特定生成任务上的表现（例如，使用 ROUGE 分数进行摘要，使用 BLEU 分数进行翻译，使用 F1/精确匹配进行问答）。
3. **效率指标：** 衡量实际益处：
   - 模型大小：参数 (parameter)数量、磁盘占用 (MB/GB)。
   - 推理 (inference)延迟：生成固定数量标记 (token)所需时间 (ms/标记)。
   - 吞吐量 (throughput)：每秒处理的序列数或生成的标记数。
   - 内存占用：推理期间的峰值 GPU 内存消耗。



![蒸馏性能与模型大小](plots/4377-0.json)



> 教师模型、蒸馏学生模型和从零开始训练的学生模型在下游任务（例如摘要）上的性能与模型大小的比较。蒸馏后的学生模型以明显更少的参数接近教师模型的性能。

#### 定性评估

评估生成文本的质量：

- **连贯性和流畅性：** 文本是否逻辑连贯，阅读自然？
- **指令遵循：** 如果在指令数据上进行了蒸馏，学生模型是否准确遵循提示？
- **创造性和多样性：** 输出是否展现多样性，还是变得重复？
- **事实性和幻觉 (hallucination)：** 检查事实准确性，特别是与教师模型相比。如果管理不当，蒸馏有时会放大偏见或不准确性。

人工评估或与教师模型输出进行并排比较通常对于全面评估是必要的。

#### 比较分析

总是将蒸馏后的学生模型与相关基线进行比较：

- **教师模型：** 与大型教师模型相比，性能损失了多少？
- **从零开始训练的学生模型：** 与直接在数据上训练相同小型架构相比，通过蒸馏获得了多少性能提升？
- **其他压缩技术：** 知识蒸馏与应用于教师模型或学生模型的剪枝或量化 (quantization)相比如何？

### 实践考量与进阶话题

- **处理架构差异：** 当匹配层之间无法完美对齐 (alignment)的隐藏状态或注意力时（例如，教师模型有 32 层，学生模型有 16 层），策略包括将教师模型的每第 k 层与学生模型的一层进行匹配、匹配前/后 N 层，或使用学习到的投影层。
- **训练动态：** 蒸馏训练有时会不稳定。尝试调整学习率（通常低于标准训练）、权重 (weight)衰减、梯度裁剪和热身计划。温度 ($T$) 和加权因子 ($\alpha$) 的选择明显影响稳定性和最终性能。
- **多教师蒸馏：** 知识可以从教师集成或在不同方面专门化的多个教师模型中蒸馏得到。
- **序列级蒸馏：** 代替标记 (token)级 KL 散度，序列级目标（例如，最小化整个生成序列概率上的散度）可以捕获更长距离的依赖关系，尽管它们通常更复杂实现。
- **推理 (inference)蒸馏：** 对于展现逐步推理（思维链）的模型，特定技术旨在蒸馏中间推理步骤，而不仅仅是最终输出。

本动手指南提供了生成式大型语言模型蒸馏的基本步骤。成功需要对架构、数据、损失函数 (loss function)和超参数 (parameter) (hyperparameter)进行细致的实验，并在定量和定性两个方面进行严格评估的指导。结果，如果成功，是一个明显更高效的模型，适合在资源受限的环境中部署。

## 参考资料

- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) — Geoffrey Hinton, Oriol Vinyals, Jeff Dean (2015)
  Journal: arXiv preprint arXiv:1503.02531; DOI: [10.48550/arXiv.1503.02531](https://doi.org/10.48550/arXiv.1503.02531)
  介绍了知识蒸馏的基本概念，包括“软目标”和温度缩放的使用，这些都直接应用于本节的KL散度损失公式中。
- [Attention Is All You Need](https://papers.nips.cc/paper/7181-attention-is-all-you-need) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems 30 (NIPS 2017); Publisher: Curran Associates, Inc. [10]; Pages: 5998–6008 [10]; DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  介绍了Transformer架构，它是本实践指南中讨论的生成式大型语言模型（教师模型和学生模型）的基础。
