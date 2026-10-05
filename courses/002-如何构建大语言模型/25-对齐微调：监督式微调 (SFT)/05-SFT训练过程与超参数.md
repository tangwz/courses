# SFT训练过程与超参数

来源：[原文](https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-25-supervised-fine-tuning-alignment/sft-training-process)

[返回章节目录](README.md) · [返回课程目录](../README.md)

监督微调 (fine-tuning)（SFT）通过在高质量的提示-响应示例数据集上训练，使预训练 (pre-training)的大型语言模型能够遵循指令或以特定风格生成回复。与预训练不同，预训练侧重于在大量非结构化文本上进行下一个词元 (token)预测，而SFT是一种更有针对性的训练形式，旨在使模型的行为与预期结果保持一致。训练过程本身类似于序列到序列任务的标准监督学习 (supervised learning)，但涉及数据格式、损失计算和超参数 (parameter) (hyperparameter)选择方面的具体考量。

### 训练循环机制

核心SFT训练循环会遍历批量的提示-响应对，执行前向传播，根据模型预测与目标响应之间的差异计算损失，并通过反向传播 (backpropagation)更新模型权重 (weight)。

1. **数据准备：** 每个训练示例通常包含一个提示（例如，一个指令或用户查询）和一个期望的响应（例如，指令的答案或一个有用的回复）。这些通常被连接成一个序列，有时会用特殊词元 (token)分隔提示和响应部分，或指示对话回合的开始/结束。

   ```python
   # 格式化示例伪代码
   prompt = "Instruction: Explain the process of photosynthesis."
   response = "Photosynthesis is the process used by plants..."
   # 如果模型/分词器需要，添加特殊词元
   tokenizer.add_special_tokens({'pad_token': '[PAD]', 'eos_token': '[EOS]'})
   # 简单连接示例
   input_text = prompt + " " + response + tokenizer.eos_token
   # 对组合文本进行分词
   tokenized_input = tokenizer(
       input_text,
       return_tensors="pt",
       padding="max_length",
       truncation=True,
       max_length=512
   )
   inputs = tokenized_input["input_ids"]
   attention_mask = tokenized_input["attention_mask"]
   ```
2. **前向传播：** 分词 (tokenization)后的序列被输入模型，以获取序列中每个词元位置的logits。

   ```python
   # 假设 'model' 是你预训练的Transformer模型
   # 且 'inputs'/'attention_mask' 来自上一步
   outputs = model(input_ids=inputs, attention_mask=attention_mask)
   logits = outputs.logits
   ```
3. **损失计算（屏蔽提示）：** 这是SFT的一个显著特点。目标是教导模型根据*提示*生成*响应*。因此，损失通常*仅*在响应词元上计算。对应于提示的词元被屏蔽，因此它们不参与损失计算或梯度更新。未屏蔽（响应）词元上使用标准交叉熵损失。

   ```python
   import torch
   import torch.nn.functional as F

   # logits: [批量大小, 序列长度, 词表大小]
   # labels: [批量大小, 序列长度] (应为左移后的输入)
   labels = inputs.clone()
   # 通常，标签会进行位移，以便模型预测下一个词元
   logits = logits[:, :-1, :] # 去掉最后一个logit
   labels = labels[:, 1:] # 去掉第一个词元（例如，BOS）

   # 确定批次中每个项目的提示长度
   # 这需要知道提示在哪里结束以及响应在哪里开始
   # 为简单起见，假设每个示例的prompt_length是已知的
   # prompt_lengths: [批量大小] 张量，包含每个示例的提示词元长度

   # 创建损失掩码：-100会被PyTorch的CrossEntropyLoss忽略
   loss_mask = torch.ones_like(labels, dtype=torch.long)
   for i in range(labels.shape[0]):
       # 屏蔽提示词元（如果tokenizer.pad_token_id存在，也屏蔽填充词元）
       prompt_end_index = prompt_lengths[i] - 1 # 根据长度定义方式进行调整
       loss_mask[i, :prompt_end_index] = -100
       if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
            loss_mask[i][labels[i] == tokenizer.pad_token_id] = -100 # 屏蔽填充

   # 将logits和labels展平以用于CrossEntropyLoss，并应用掩码
   # 仅在loss_mask不为-100的词元上计算损失
   active_loss = loss_mask.view(-1) != -100
   active_logits = logits.view(-1, logits.size(-1))[active_loss]
   active_labels = labels.view(-1)[active_loss]

   loss = F.cross_entropy(active_logits, active_labels)
   ```

   实际上，如果数据格式正确（例如，使用特定数据集格式或提供数据收集器），Hugging Face的`transformers.Trainer`或TRL (`trl.SFTTrainer`) 等库会在内部处理这种掩码逻辑。
4. **反向传播与优化：** 标准反向传播根据计算出的损失来计算梯度。优化器（通常是AdamW）更新模型权重。

   ```python
   # 使用标准PyTorch优化步骤
   optimizer.zero_grad()
   loss.backward()
   # 可选：梯度裁剪
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   optimizer.step()
   if scheduler is not None:
       scheduler.step()
   ```

### 超参数 (parameter) (hyperparameter)考量

选择合适的超参数对SFT的成功很重要。由于SFT调整的是一个已很强大的预训练 (pre-training)模型，其设置与预训练时使用的不同。

- **学习率：** SFT通常使用比预训练小得多的学习率。$1 \times 10^{-5}$ 到 $5 \times 10^{-5}$ 范围内的值很常见。较小的学习率可以防止预训练期间获得的知识被灾难性遗忘，同时允许模型适应新的指令遵循目标。学习率调度器，如带有短热身阶段（例如，总步数的0-10%）的余弦衰减，通常会带来好处。

  > 一个典型的SFT学习率调度，包括热身和余弦衰减。
- **批量大小：** 批量大小通常受限于GPU内存。更大的模型需要更多内存，从而限制了每个批次的序列数量。梯度累积常用于在不增加每GPU内存使用量的情况下实现更大的*有效*批量大小。典型的有效批量大小可能在64到1024之间，具体取决于模型大小和可用硬件。
- **训练轮次（Epoch）数量：** SFT通常只需要少数几个训练轮次（通常1-3个，有时最多5个）。训练时间过长可能导致在特定SFT数据集上过拟合 (overfitting)，从而可能降低模型对未见指令的泛化能力或降低其通用知识。监控验证集上的表现很重要。
- **优化器：** AdamW仍然是标准选择，类似于预训练。权重 (weight)衰减参数可以保持不变或略作调整（例如，0.01到0.1）。Beta参数（$\beta_1, \beta_2$）通常保持其默认值（例如，0.9，0.999）。
- **序列长度：** 最大序列长度应能容纳SFT数据集中典型提示和响应的组合长度。它可能与预训练期间使用的序列长度不同。将多个短示例打包到一个序列中或使用动态填充可以提高效率。
- **梯度裁剪：** 应用梯度裁剪（例如，将梯度的L2范数裁剪到1.0）有助于稳定训练，尽管与大规模预训练相比，SFT中的不稳定性通常较少发生。

### 实际实施要点

- **计算资源：** 虽然SFT对计算资源的要求低于预训练 (pre-training)数万亿词元 (token)，但大型模型的SFT仍需要大量计算，通常涉及多个高内存GPU（如A100或H100）。
- **分布式训练：** 对于更大的模型或更快的迭代，通常采用数据并行（DP），使用PyTorch的DistributedDataParallel (DDP) 或DeepSpeed（尤其是ZeRO Stage 1或2）等框架。
- **数据集质量：** SFT数据集的质量和多样性可以说比纯粹的数量更重要。几千个高质量、多样化的示例通常能比数百万个嘈杂或重复的示例产生更好的结果。
- **评估：** 使用损失指标评估SFT。通过在保留的指令集上评估表现，使用自动化指标（如用于摘要的ROUGE），或采用人工评估，是了解对齐 (alignment)目标是否达到所必需的。

SFT过程微调 (fine-tuning)模型的能力，将模型的预训练知识引导至指令数据集定义的特定交互模式和任务执行格式。对训练循环和超参数 (parameter) (hyperparameter)的细致管理可确保这种调整有效进行，同时不损害模型内在优势。

## 参考资料

- [Supervised Fine-tuning (SFT) Trainer](https://huggingface.co/docs/trl/main/en/sft_trainer) — Hugging Face (2024)
  Publisher: Hugging Face
  TRL 库中 SFTTrainer 的官方文档，涵盖大型语言模型监督微调的实际实现细节，包括数据格式化和损失计算。
- [Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2203.02155) — Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, Ryan Lowe (2022)
  Journal: arXiv preprint arXiv:2203.02155; DOI: [10.48550/arXiv.2203.02155](https://doi.org/10.48550/arXiv.2203.02155)
  介绍了 InstructGPT 模型，详细说明了其初始监督微调（SFT）阶段、数据收集及其在使语言模型与人类指令和偏好对齐方面的作用。
- [Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416) — Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, Albert Webson, Shixiang Shane Gu, Zhuyun Dai, Mirac Suzgun, Xinyun Chen, Aakanksha Chowdhery, Alex Castro-Ros, Marie Pellat, Kevin Robinson, Dasha Valter, Sharan Narang, Gaurav Mishra, Adams Yu, Vincent Zhao, Yanping Huang, Andrew Dai, Hongkun Yu, Slav Petrov, Ed H. Chi, Jeff Dean, Jacob Devlin, Adam Roberts, Denny Zhou, Quoc V. Le, Jason Wei (2022)
  Journal: arXiv preprint arXiv:2210.11416; DOI: [10.48550/arXiv.2210.11416](https://doi.org/10.48550/arXiv.2210.11416)
  探讨了大规模指令微调的有效性，展示了在多样化指令数据集上训练如何显著提高大型语言模型在各种任务上的泛化能力。

---

[上一节](04-SFT%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F%EF%BC%88%E6%8F%90%E7%A4%BA%E8%AF%8D%EF%BC%8C%E5%9B%9E%E5%BA%94%EF%BC%89.md) · [下一节](06-%E8%AF%84%E4%BC%B0SFT%E6%A8%A1%E5%9E%8B%E5%AF%B9%E9%BD%90%E7%9B%AE%E6%A0%87.md)
