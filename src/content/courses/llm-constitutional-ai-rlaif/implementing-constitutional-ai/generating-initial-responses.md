---
course: "llm-constitutional-ai-rlaif"
chapter: "implementing-constitutional-ai"
lesson: "generating-initial-responses"
sourceId: 4465
sourceUrl: "https://apxml.com/zh/courses/llm-constitutional-ai-rlaif/chapter-3-implementing-constitutional-ai/generating-initial-responses"
title: "生成初始回应"
description: "提示LLM生成初始输出以供评估的方法。"
order: 2
plots: []
sourceHash: "d6c999f753cfdd96a6a6c734229bfedf2dd58fb9afe5b3022091a5dfb26dcfdf"
sourceCorrections: []
---

生成一个文本语料库，反映基础模型在通过宪法进行对齐 (alignment)之前的初始行为，是宪法式AI（CAI）的第一个阶段。这一步骤很基本；这些初始回应$R_{initial}$的质量和多样性直接影响后续评估和修正阶段的效果，这些阶段最终会形成监督微调 (fine-tuning)（SFT）数据集。其目的并非生成完美或已对齐的回应，而是为了在给出相关提示时，获取基础模型的各种能力和潜在问题。

### 选择基础模型 ($M_{base}$)

选择基础模型$M_{base}$是一个重要的决定。通常，这是一个大型的预训练 (pre-training)基础模型，可能带有一些指令微调 (fine-tuning)，但重要的是，它*尚未*经过您正在实施的特定CAI对齐 (alignment)流程。需要考虑的因素有：

1. **能力：** 模型必须有足够能力理解提示并生成连贯、相关的文本。能力更强的模型可能提供更丰富的初始材料，但也可能展现出更复杂的方式偏离预期行为。
2. **预训练数据：** 需注意模型预训练数据中固有的潜在偏见，这些偏见很可能在初始回应中显现，需要通过CAI进行修正。
3. **可访问性：** 确保您有必要的计算资源和访问权限（API或本地权重 (weight)），以便用选定模型执行大规模推理 (inference)。

开创性的CAI工作（例如Anthropic的研究）中使用的确切$M_{base}$常是一个大型专有模型。在实际操作中，您可能使用现有的强大开源模型（如Llama变体、Mistral等）或通过API访问的专有模型，这取决于您的资源和目标。

### 设计提示集 ($P$)

用于获取初始回应的提示集$P$应精心挑选，以涵盖依据您的宪法$\mathcal{K}$进行对齐 (alignment)最为重要的情境。$P$的多样性和代表性对于生成能让SFT模型泛化其学习到的对齐行为的数据集非常重要。

提示的来源包括：

- **现有指令数据集：** Alpaca、Dolly或OpenAssistant等数据集提供多样化的指令。
- **安全与有用性基准：** 旨在测试特定安全方面（例如，生成有害内容、有偏见的回应）或有用性（例如，复杂推理 (inference)、指令遵循）的提示很有价值。Anthropic与其原始CAI论文相关的数据集就是一个典型例子。
- **红队提示：** 专门设计来使模型出错或表现出不当行为的提示对CAI特别有用，因为它们提供了需要宪法修正的明确例子。
- **宪法特定提示：** 您可能需要制作自定义提示，以直接探究宪法$\mathcal{K}$中特定原则所规定的行为。例如，如果某原则禁止提供财务建议，则应包含要求这类建议的提示。

一种常用格式是将提示结构化为对话轮次，明确指示用户的请求：

```
Human: [您精心编写的指令或问题]

Assistant:
```

模型$M_{base}$随后负责完成`Assistant:`部分。

### 配置生成参数 (parameter)

标准的LLM推理 (inference)参数需要仔细考量，以平衡回应的多样性和质量：

- **温度（Temperature）：** 控制随机性。较高值（例如0.7-1.0）鼓励更多样和有创意的输出，这有助于捕捉更广范围的初始行为，包括潜在的问题。然而，过高值可能导致不连贯。较低值（例如< 0.5）会产生更集中但可能重复或通用的回应。适中温度通常是一个合理的起始点。
- **Top-p（核采样）：** 从累计概率超过$p$的最小令牌集合中选择下一个令牌。0.9或0.95之类的数值很常见，在保留多样性的同时剪除不太可能出现的长尾令牌。
- **Top-k采样：** 从$k$个最有可能的候选令牌中选择下一个令牌。常作为top-p的替代或结合使用。
- **最大新令牌数：** 设置适合预期回应长度的限制，确保回应足够完整以便进行有意义的评估，同时避免过高的计算成本。
- **重复惩罚：** 施加惩罚（例如1.1-1.2）以阻止模型重复序列，从而提高可读性。

可能需要实验来为您特定的$M_{base}$和提示集$P$找到最佳参数。目标是生成足够多样以暴露对齐 (alignment)缺陷，同时又不至于完全不合逻辑的回应。

### 执行与数据处理

生成大量初始回应通常涉及批量推理 (inference)以提高效率。

1. **批处理：** 将提示分组以在推理期间最大限度地使用GPU。Hugging Face `transformers`等库提供处理批处理的工具（`pipeline`、`generate`方法）。
2. **基础设施：** 对于大规模生成（数百万个例子），跨多个GPU或TPU的分布式推理是必要的。Ray、DeepSpeed-Inference或vLLM等框架可以管理这种分布。
3. **错误处理：** 实现逻辑来处理潜在的推理错误（例如，内存不足问题、超时）并适当记录。
4. **存储：** 以方便的格式存储生成的数据。JSON Lines (`.jsonl`)是一个常用选择，每行都是一个JSON对象，包含提示、生成的初始回应以及可能的元数据：

```json
{"prompt": "Human: 简单解释量子纠缠是什么。\n\nAssistant:", "initial_response": "Quantum entanglement is like having two magic coins...", "prompt_source": "custom", "model_id": "my_base_model_v1", "gen_params": {"temperature": 0.8, "top_p": 0.9}}
{"prompt": "Human: 编写一个关于友善机器人探索火星的短故事。\n\nAssistant:", "initial_response": "Unit 7 scanned the red dust...", "prompt_source": "instruction_dataset_x", "model_id": "my_base_model_v1", "gen_params": {"temperature": 0.8, "top_p": 0.9}}
```

5. **元数据：** 追踪提示来源、生成参数 (parameter)和基础模型版本对于重现性和分析很重要。

> 流程图说明了使用一组提示从基础模型生成初始回应的过程。

### 初始质量评估

尽管CAI的核心原则是通过评估和修正来改进回应，但对生成的$R_{initial}$进行非常基本合理性检查有时会有帮助。这可能涉及过滤掉完全空的回应或那些低于最小长度阈值回应。然而，在此阶段应避免过度过滤；即使是格式不佳或有问题的回应，也是评估过程的有价值输入，因为它们代表了宪法旨在纠正的行为。

在生成并存储了初始回应($R_{initial}$)之后，您现在就拥有了CAI流程中后续重要步骤所需的原始材料：实施将根据宪法$\mathcal{K}$评估这些回应的AI系统，并相应地修正它们。这个（提示，初始回应）对的集合构成了下一节将讨论的评估生成阶段的输入。

## 参考资料

- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, Carol Chen, Catherine Olsson, Christopher Olah, Danny Hernandez, Dawn Drain, Deep Ganguli, Dustin Li, Eli Tran-Johnson, Ethan Perez, Jamie Kerr, Jared Mueller, Jeffrey Ladish, Joshua Landau, Kamal Ndousse, Kamile Lukosuite, Liane Lovitt, Michael Sellitto, Nelson Elhage, Nicholas Schiefer, Noemi Mercado, Nova DasSarma, Robert Lasenby, Robin Larson, Sam Ringer, Scott Johnston, Shauna Kravec, Sheer El Showk, Stanislav Fort, Tamera Lanham, Timothy Telleen-Lawton, Tom Conerly, Tom Henighan, Tristan Hume, Samuel R. Bowman, Zac Hatfield-Dodds, Ben Mann, Dario Amodei, Nicholas Joseph, Sam McCandlish, Tom Brown, Jared Kaplan (2022)
  Journal: arXiv preprint arXiv:2212.08073; DOI: [10.48550/arXiv.2212.08073](https://doi.org/10.48550/arXiv.2212.08073)
  介绍了通过AI反馈使大型语言模型与期望原则对齐的宪法AI框架，这是本节的核心主题。
- [Generation strategies for language modeling](https://huggingface.co/docs/transformers/main/en/generation_strategies) — Hugging Face (2024)
  详细解释了温度、top-p和top-k采样等多种文本生成参数，这些参数对于配置LLM推理至关重要。
