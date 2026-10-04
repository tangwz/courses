---
course: "fine-tuning-adapting-large-language-models"
chapter: "llm-adaptation-foundations"
lesson: "need-for-fine-tuning"
sourceId: 2959
sourceUrl: "https://apxml.com/zh/courses/fine-tuning-adapting-large-language-models/chapter-1-llm-adaptation-foundations/need-for-fine-tuning"
title: "微调和适配的必要性"
description: "讨论预训练模型的局限性以及进行特定任务和特定专业适配的原因。"
order: 2
plots: []
sourceHash: "336080398804ab27c31ca4c950f24e669e5538d7e592823f3568902639a23a1f"
sourceCorrections: []
---

大型预训练 (pre-training)语言模型（LLM）通过对海量互联网文本和代码进行大量训练，具备出色的通用语言理解和生成能力。然而，这种通用性在将其应用于特定场景时常成为一种制约。LLM的预训练目标通常是基于一个庞大的语料库进行下一词元 (token)预测，这使得模型具备丰富的知识，但却缺少许多实际用途所需的专业能力、风格控制或任务对齐 (alignment)。

思考一下内在的权衡：一个对所有数据都进行过训练的模型，可能无法针对某个特定事务进行优化。仅仅使用提示技术（零样本或少样本学习 (few-shot learning)）来引导这些通用模型，经常会遇到以下几个重要障碍：

- **表现不一：** 模型的输出质量可能对提示词 (prompt)的确切措辞和结构高度敏感。微小的修改可能导致明显不同的结果，使得仅通过提示工程 (prompt engineering)难以实现可靠且一致的行为。
- **上下文 (context)窗口限制：** 少样本提示需要在输入提示中直接提供示例。这会占用宝贵的上下文窗口空间，从而限制了示例的数量或实际查询的长度，特别是对于上下文限制较小的模型。
- **知识缺失：** 预训练语料库虽然庞大，但可能不足以代表或完全遗漏小众术语、特定专业内容（例如，公司内部缩写、专有技术标准、近期法律判例）或私有数据集。一个通用模型无法有效处理其从未遇到或很少遇到的信息。
- **风格和角色不一致：** 基础模型的默认输出风格、语气或隐含角色可能不适合目标应用。仅通过提示词来强制指定一个角色（例如，正式的法律助理与一个风趣的聊天机器人）通常不可靠，并且在复杂交互中可能会失效。
- **指令遵循不足：** 尽管LLM在遵循指令方面越来越好，但若指令复杂、多步骤或详细，且与预训练期间所见的模式明显不同，它们可能会被忽略、误解或错误执行，除非进行明确的适配。
- **特定场景的幻觉 (hallucination)：** 当被问及其核心知识范围之外的专业话题时，LLM更容易生成听起来合理但事实不准确的信息（幻觉）。它们可能会虚构不存在的技术规范，误解特定行话，或编造与私有数据相关的细节。

微调 (fine-tuning)和适配通过使用精心挑选的、面向特定任务的数据集来调整模型的内部参数 (parameter)，直接解决了这些局限性。这一过程从仅仅依赖模型最初的预训练知识，转变为引导其行为趋向预期的结果。进行微调的主要原因包括：

1. **任务特化：** 在特定下游NLP任务上达到先进或接近先进的性能，例如专业分类（高精度分类金融情绪）、复杂摘要（总结冗长的医学研究论文）或结构化信息提取（从非结构化法律合同中提取特定数据点）。
2. **领域适配：** 通过在相关的特定专业语料库上进行训练，为模型注入某一特定方面（医学、法律、金融、特定软件工程专业）的深度知识。这能提升模型对该专业行话、上下文和相关实体的理解。
3. **风格、语气和角色对齐：** 可靠地调整模型的输出，使其符合特定的风格指南，保持一致的语气（例如，富有同情心、正式、客观），或采纳品牌或用户体验所需的特定角色。
4. **提升指令遵循能力：** 通过对包含指令和预期响应的数据集进行训练，大幅提升模型可靠地理解和执行与预期应用相关的特定命令或任务的能力（通常称作指令调优）。
5. **减少幻觉：** 通过微调将模型根植于特定专业的事实和预期响应模式中，可以降低其在该专业内运行时生成不准确或虚构信息的可能性。
6. **整合私有数据：** 使用专有或机密数据集对模型进行适配，使模型能够借助这些信息，而无需在推理 (inference)过程中反复在提示中暴露原始数据（尽管细致的数据整理和安全措施仍必不可少）。

例如，一个通用LLM可能难以针对新发布的内部软件库生成准确的代码片段。虽然提供少量示例可能有所帮助，但若对模型进行微调，使其学习该库的文档和现有代码库，将能得到一个更可靠、知识更丰富的助手。同样，一个对过往支持记录和内部知识库文章进行过微调的客户支持聊天机器人，在理解用户问题和提供相关、公司特定的解决方案方面，将远远优于通用模型。

因此，微调代表了迁移学习 (transfer learning)的一种强大应用。它利用预训练期间学到的丰富通用表示，并使用有针对性的数据和目标对其进行优化，以达到特定目的。尽管这种适配过程非常有效，但它需要仔细考虑数据质量、计算资源和评估方法，这些方面我们将在后续章节中细致研究。这是一个专业化的过程，使我们能够将大型预训练模型的惊人潜能塑造成面向特定用途的有效工具。

## 参考资料

- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://aclanthology.org/N19-1423/) — Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova (2019)
  Journal: Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers); Publisher: Association for Computational Linguistics; Pages: 4171–4186; DOI: [10.18653/v1/N19-1423](https://doi.org/10.18653/v1/N19-1423)
  介绍了BERT，这是自然语言处理领域预训练和微调的开创性模型，展示了迁移学习在特定下游任务中的有效性。
- [Language Models are Few-Shot Learners](https://proceedings.neurips.cc/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf) — Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, Dario Amodei (2020)
  Journal: Advances in Neural Information Processing Systems; Publisher: Curran Associates, Inc.; Volume: 33; Pages: 1877-1901; DOI: [arXiv:2005.14165](https://doi.org/arXiv%3A2005.14165)
  介绍了GPT-3，展示了大型语言模型强大的少样本学习能力，同时也间接指出了需要通过微调实现特定对齐和控制的局限性。
- [Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416) — Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, Albert Webson, Shixiang Shane Gu, Zhuyun Dai, Mirac Suzgun, Xinyun Chen, Aakanksha Chowdhery, Alex Castro-Ros, Marie Pellat, Kevin Robinson, Dasha Valter, Sharan Narang, Gaurav Mishra, Adams Yu, Vincent Zhao, Yanping Huang, Andrew Dai, Hongkun Yu, Slav Petrov, Ed H. Chi, Jeff Dean, Jacob Devlin, Adam Roberts, Denny Zhou, Quoc V. Le, Jason Wei (2022)
  Journal: arXiv preprint arXiv:2210.11416; DOI: [10.48550/arXiv.2210.11416](https://doi.org/10.48550/arXiv.2210.11416)
  探讨了指令微调，展示了在多样化指令格式任务上训练如何显著提高模型在各种下游自然语言处理任务上的性能和指令遵循能力。
- [Training Language Models to Follow Instructions with Human Feedback](https://cdn.openai.com/papers/Training_Language_Models_to_Follow_Instructions_with_Human_Feedback.pdf) — Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke E. Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Francis Christiano, Jan Leike, Ryan J. Lowe (2022)
  Journal: Advances in Neural Information Processing Systems; Pages: 4299-4307
  介绍了InstructGPT，详细说明了如何利用人类反馈的强化学习来使大型语言模型与人类偏好对齐，从而提高指令遵循、真实性和安全性。
