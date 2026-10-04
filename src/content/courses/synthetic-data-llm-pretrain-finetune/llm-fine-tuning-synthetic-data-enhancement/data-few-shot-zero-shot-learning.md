---
course: "synthetic-data-llm-pretrain-finetune"
chapter: "llm-fine-tuning-synthetic-data-enhancement"
lesson: "data-few-shot-zero-shot-learning"
sourceId: 6137
sourceUrl: "https://apxml.com/zh/courses/synthetic-data-llm-pretrain-finetune/chapter-4-llm-fine-tuning-synthetic-data-enhancement/data-few-shot-zero-shot-learning"
title: "生成少样本和零样本学习场景的数据"
description: "如何生成合成示例以提升LLM在少样本或零样本环境下的表现。"
order: 4
plots: []
sourceHash: "89a7e2d5c9d0638d368a74b80e6e02df48e65ab3dc09bec16043f632ca862336"
sourceCorrections: []
---

大型语言模型展现出出色的零样本学习 (zero-shot learning)（ZSL）和少样本学习 (few-shot learning)（FSL）能力，分别使得它们能够在没有或只有极少示例的情况下完成任务。零样本学习依赖于模型理解任务描述并在没有预先特定训练实例的情况下执行任务的能力。少样本学习涉及在提示语中向模型提供少量演示（即“样本”），以引导其对新颖、类似输入的响应。合成数据的生成能够显著提升这些能力，特别是在针对新颖或专门任务的示例稀缺时。

### 为什么在少样本和零样本场景中使用合成数据？

对大型语言模型进行微调 (fine-tuning)，使其成为更好的零样本或少样本学习 (few-shot learning)器，通常需要一个数据集来教导它*如何*从指令中归纳或*如何*高效地运用示例。手动创建此类多样且高质量的数据集可能成本过高或难以实现。合成数据提供了一个可扩展的方案：

1. **应对数据稀缺：** 对于全新任务或高度特定的专业范围，真实示例可能很少，甚至没有。合成生成可以从头开始创建这些示例。
2. **提高指令泛化能力（ZSL）：** 为了使大型语言模型在零样本任务上表现更好，可以对其进行微调，使用大量合成生成的指令以及涵盖各种任务的相应输出。这教导模型更好地将任务描述映射到预期行为。
3. **制作高质量演示（FSL）：** 少样本学习的有效性很大程度上取决于示例“样本”的质量。合成数据方法可以用于生成清晰、简洁、多样化的演示，有效引导模型。
4. **受控的多样性：** 您可以控制合成数据的特性（例如，复杂性、风格、格式），以训练大型语言模型处理更广泛的零样本指令或少样本示例模式。

### 为改进零样本学习 (zero-shot learning)生成合成数据

这里的目标是提升模型理解和执行其未明确训练过的任务指令的能力。

1. **任务描述泛化与扩充：**
   从现有任务描述开始，或发明新的任务描述。使用一个强大的大型语言模型（一个“教师”模型）来重新措辞、抽象化或变化这些描述。例如，如果您希望模型擅长各种文本转换任务，您可以生成如下配对：

   - 指令：“将以下文本转换为被动语态：[input\_text]”
     输出：“[passive\_voice\_output]”
   - 指令：“使用主动语态重写此句：[input\_text]”
     输出：“[active\_voice\_output]”
   - 指令：“识别此段落中的主要动词：[input\_text]”
     输出：“[main\_verb]”
     对大量此类多样化的指令-输出配对进行微调 (fine-tuning)，有助于模型学习指令如何映射到动作的底层模式。
2. **生成新颖指令和任务：**
   提示一个有能力的大型语言模型，以发明新的、合理可行的任务并为其提供指令。例如：
   “生成一个文本处理任务的指令，该任务涉及识别古老词语并建议现代同义词。提供一个示例输入和输出。”
   此提示的输出成为一个合成训练实例。目的不一定是要模型精通这些特定任务，而是要学习如何有效处理*任何*新的、结构良好的指令。

### 为改进少样本学习 (few-shot learning)生成合成数据

对于少样本学习，合成数据生成过程侧重于创建有效的示例（样本），模型在推理 (inference)时作为提示的一部分呈现时可以从中学习。在这种情况下，微调 (fine-tuning)过程旨在使模型更擅长*运用*此类样本。

1. **制作高质量的输入-输出演示：**
   少样本提示中的“样本”非常重要。您可以使用大型语言模型来生成这些。对于给定任务类型（例如情感分析、短篇故事生成、代码解释），提示一个教师大型语言模型创建几个不同的输入-输出配对，这些配对能很好地说明该任务。

   - 给教师大型语言模型的示例提示：“生成三个清晰且不同的示例，包含关于历史事件的问题及其简洁答案。每个示例应展示略有不同的提问风格。”
     生成的示例随后可以作为合成微调数据集的一部分使用，其目标是在给定新查询和这些合成样本的情况下预测输出。
   > 一个大型语言模型根据种子任务描述生成合成示例（样本）。这些示例与新的输入一起，构成一个少样本提示，引导目标大型语言模型。
2. **生成思维链（CoT）示例：**
   对于需要多步推理的任务，展示推理过程（思维链）的少样本示例非常有效。您可以通过提示大型语言模型来合成生成这些示例，让它解决问题并阐明其逐步思考过程。

   - 给教师大型语言模型的提示：“解决以下数学应用题并逐步解释您的推理：[word\_problem]。然后，只提供最终答案。”
     完整的解释和最终答案成为一个合成的思维链示例。对这些示例进行微调有助于模型在得到适当提示时学习“逐步思考”。
3. **扩充稀缺的真实示例：**
   如果您只有非常少量的少样本演示，可以使用合成数据技术，如释义或基于大型语言模型的重写，来创建变体。这会扩展您的演示集合，帮助模型从有限的真实数据中更好地归纳。确保扩充后的示例保留了原始示例的核心意图和正确性。

### 用于微调 (fine-tuning)的合成数据结构

在微调大型语言模型以提升其零样本学习 (zero-shot learning)或少样本学习 (few-shot learning)能力时，合成数据通常遵循指令-响应对的格式，通常采用JSONL格式：

**用于零样本学习增强：**
每个微调示例都是一个直接指令及其理想输出。

```json
{"instruction": "将以下英文句子翻译成西班牙语：'Hello, how are you?'", "output": "Hola, ¿cómo estás?"}
{"instruction": "将此文档总结为三个要点：[长文档文本]", "output": "- 要点1\n- 要点2\n- 要点3"}
```

**用于少样本学习增强：**
微调数据本身旨在教导模型如何*使用*示例。“样本”是输入的一部分。

```json
{
  "instruction": "给定以下将主动语态转换为被动语态的示例：\n示例1 输入：The cat chased the mouse.\n示例1 输出：The mouse was chased by the cat.\n示例2 输入：The team celebrated their victory.\n示例2 输出：Their victory was celebrated by the team.\n\n现在，将此句转换为被动语态：The chef prepares delicious meals.",
  "output": "美味的饭菜由厨师准备。"
}
```

在这里，合成生成过程创建`指令`（包括样本）和相应的`输出`。当给定此类语境学习提示时，模型经过微调以生成正确的输出。

### 实际考量

- **质量非常重要：** 特别是对于少样本示例，合成“样本”的质量显著影响性能。质量差或具有误导性的示例可能会损害模型的能力。应投入精力为生成器大型语言模型设计强有力的提示，并考虑进行过滤/验证步骤。
- **多样性很重要：** 生成广泛的指令、任务类型和示例格式。这有助于模型更具适应性。
- **提示生成器大型语言模型：** 对创建合成数据的“教师”大型语言模型进行有效的提示工程 (prompt engineering)非常重要。清晰指定所需的格式、风格、复杂性以及任何限制。
- **评估：** 在真正未曾见过的任务（针对零样本学习 (zero-shot learning)）或新的样本和查询组合（针对少样本学习 (few-shot learning)）上测试微调 (fine-tuning)后的模型，这些任务或组合不应是合成训练数据的一部分。这能真实衡量改进效果。
- **迭代改进：** 为零样本学习/少样本学习生成合成数据通常是一个迭代过程。分析微调模型的性能，找出不足之处，并相应地完善您的合成数据生成策略。

通过深思熟虑地生成合成数据，您可以显著提升大型语言模型在极少或没有示例的情况下处理新任务的能力，使其成为一个更通用和强大的工具。当将模型应用于专业化范围或新颖应用，而大型标注数据集并不容易获得时，这一点尤为有益。

## 参考资料

- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) — Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, Dario Amodei (2020)
  Journal: Advances in Neural Information Processing Systems; DOI: [10.48550/arXiv.2005.14165](https://doi.org/10.48550/arXiv.2005.14165)
  本文介绍了GPT-3，并展示了大型语言模型的小样本和零样本学习能力，为上下文学习奠定了基础。
- [Finetuned Language Models Are Zero-Shot Learners](https://arxiv.org/abs/2109.01652) — Jason Wei, Maarten Bosma, Vincent Y. Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, and Quoc V. Le (2021)
  Journal: International Conference on Learning Representations; DOI: [10.48550/arXiv.2109.01652](https://doi.org/10.48550/arXiv.2109.01652)
  介绍了指令微调（FLAN），一种通过在以自然语言指令形式表达的任务集上微调语言模型的方法，以提高其零样本泛化能力。
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903) — Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, Denny Zhou (2022)
  Journal: Advances in Neural Information Processing Systems; DOI: [10.48550/arXiv.2201.11903](https://doi.org/10.48550/arXiv.2201.11903)
  本文提出了思维链提示，通过引导LLM输出中间推理步骤来增强其推理能力，这与为FSL生成CoT示例直接相关。
