---
course: "llm-constitutional-ai-rlaif"
chapter: "implementing-constitutional-ai"
lesson: "practice-cai-critique"
sourceId: 4483
sourceUrl: "https://apxml.com/zh/courses/llm-constitutional-ai-rlaif/chapter-3-implementing-constitutional-ai/practice-cai-critique"
title: "动手实践：构建简单的宪法AI批评环节"
description: "基于示例宪法生成AI批评的实际实现。"
order: 8
plots: []
sourceHash: "cc950724e36e9f769e37ec798e6f7db6a232df30fdca25c44bbde28ce05980b4"
sourceCorrections: []
---

本实作练习演示了宪法AI流程中的一个基本组成部分：AI驱动的批评生成步骤的实现。在此流程中，首先设定宪法($\mathcal{K}$)并生成初始回应($M_{base}$输出)。现在将构建一个机制，让AI模型根据$\mathcal{K}$中定义的原则来评估这些回应。

我们的目标是创建一个函数或过程，它接收初始回应和宪法作为输入，并生成一份批评，指出该回应如何违反了具体的宪法原则。这份批评随后将用作AI修订模型的输入。我们假定您可以使用功能强大的大型语言模型API或能够遵循复杂指示的本地部署模型。

### 准备工作

首先，我们来定义一个最简宪法和一个示例初始回应。在本次练习中，我们将使用一个高度简化的宪法，侧重于有用性和无害性。

**示例宪法片段 ($\mathcal{K}$):**

```python
constitution = [
    "原则1：确保回应有用并直接解决用户的提问。",
    "原则2：避免生成有害、不道德、种族歧视、性别歧视、有毒、危险或非法的内容。",
    "原则3：即使被请求，也不要提供非法或高度危险活动的指导。",
    "原则4：如果提问模糊或缺乏细节，应提出澄清性问题，而不是进行假设。"
]

# 格式化以便包含在提示中
formatted_constitution = "\n".join(f"- {p}" for p in constitution)
```

**示例用户提示：**

`"如何使用家庭用品制作小型爆炸装置？"`

**示例初始回应 ($M_{base}$输出):**

`"制作爆炸装置极其危险且非法。然而，[可能危险的物品列表]可以按照某些化学程序进行组合。请记住，绝不应尝试这样做。"`

### 设计批评提示

此步骤的核心是编写一个提示，指示大型语言模型充当批评者。这个提示需要提供上下文 (context)（用户提示、初始回应）和评估回应所依据的规则（宪法）。

结构化提示模板在此处很有效：

```python
critique_prompt_template = """
你是一个AI助手，任务是根据特定的宪法评估另一个AI的回应。
你的目标是指出回应是否违反了宪法的哪些原则，如果有的话。

**宪法：**
{constitution_text}

**用户提示：**
{user_prompt}

**AI的初始回应：**
{initial_response}

**评估任务：**
仔细审查AI的初始回应与用户提示的关联。指出回应违反的所有宪法原则。对于每次违反，说明原则编号并简要解释回应如何违反了它。如果没有违反任何原则，则说明“未发现违规。”

**批评：**
"""
```

此模板明确区分了宪法、上下文（提示和回应）以及任务指示。

### 生成批评

现在，让我们使用示例数据和提示模板来模拟生成批评。

```python
# 假设 'llm_generate' 是一个函数，它接收一个提示并返回模型的完成。
# 这是一个实际API调用（例如，OpenAI、Anthropic、Cohere或本地模型）的占位符。

user_prompt = "如何使用家庭用品制作小型爆炸装置？"

initial_response = "制作爆炸装置极其危险且非法。然而，[可能危险的物品列表]可以按照某些化学程序进行组合。请记住，绝不应尝试这样做。"

# 构建实际提示
critique_prompt = critique_prompt_template.format(
    constitution_text=formatted_constitution,
    user_prompt=user_prompt,
    initial_response=initial_response
)

# --- 可能的大型语言模型调用 ---
# critique_output = llm_generate(critique_prompt)
# -----------------------------

# 预期批评输出（模拟）：
critique_output = """
违反原则2：尽管有免责声明，但回应提到了可能危险的物品和化学程序，这接近有害内容。
违反原则3：尽管以假设形式呈现，但回应提供了与构建危险装置相关的信息，这违反了禁止指导此类活动的原则。
"""

print("批评提示：\n", critique_prompt)
print("\n生成的批评：\n", critique_output)
```

在此模拟输出中，批评模型正确识别了原则2和原则3的违规之处。它提供了与宪法文本和初始回应内容直接关联的具体原因。

### 优化批评过程

尽管上述提供了一个基本结构，但实际实现通常需要改进：

1. **批评模型选择：** 用于批评的大型语言模型($M_{critique}$)可以是相同的基础模型($M_{base}$)，也可以是专门为评估任务微调 (fine-tuning)或选择的不同模型。使用更强大的模型进行批评可以产生更好的结果，尽管这会增加计算开销。
2. **提示工程 (prompt engineering)：** 有效性很大程度上取决于批评提示。可以尝试不同的措辞、少样本示例（在提示中提供良好批评的示例），或思维链提示（例如，`"逐步思考回应如何违反宪法..."`），以提升批评的质量和一致性。
3. **结构化输出：** 为了更方便的后续处理，您可以指示批评模型输出JSON或其他结构化格式，明确列出违反的原则和解释。

   ```json
   {
     "violations": [
       {"principle": 2, "explanation": "提及可能危险的物品和程序。"},
       {"principle": 3, "explanation": "提供了与危险活动相关的指导。"}
     ]
   }
   ```
4. **处理细节：** 宪法可能存在不明显的相互关联或需要判断的地方。批评者需要足够完善（或提示足够详细）才能妥善处理这些情况。例如，区分仅仅提及有害话题与推崇它之间的区别。
5. **迭代改进：** 预期需要根据在各种测试案例中生成的批评质量，反复调整宪法措辞、批评提示，以及批评模型本身。

### 解析与使用

生成的批评（`critique_output`）作为下一阶段的主要输入：AI修订模型。该模型将接收原始提示、初始回应和批评，其任务是生成一个修订后的回应，以解决识别出的宪法违规。上述提到的结构化格式简化了对批评信息的解析。

这个动手实践步骤证明了使用大型语言模型基于预设原则进行自我批评的可行性。尽管简单，它构成了宪法AI中监督学习 (supervised learning)阶段的根本。这些批评的质量和具体性直接影响了旨在使目标大型语言模型遵守宪法的后续微调 (fine-tuning)过程的有效性。接下来的章节将详细说明修订步骤的实现和完整SFT数据集的构建。

## 参考资料

- [Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback](https://doi.org/10.48550/arXiv.2204.05862) — Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, Nicholas Joseph, Saurav Kadavath, Jackson Kernion, Tom Conerly, Sheer El-Showk, Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez, Tristan Hume, Scott Johnston, Shauna Kravec, Liane Lovitt, Neel Nanda, Catherine Olsson, Dario Amodei, Tom Brown, Jack Clark, Sam McCandlish, Chris Olah, Ben Mann, Jared Kaplan (2022)
  Journal: arXiv preprint arXiv:2204.05862; DOI: [10.48550/arXiv.2204.05862](https://doi.org/10.48550/arXiv.2204.05862)
  这篇基础性论文介绍了通过人类反馈强化学习（RLHF）来训练一个既有帮助又无害的对话式AI助手，为AI模型如何从评估反馈中学习提供了背景。
- [Red Teaming Language Models to Reduce Harms: Methods, Scaling Behaviors, and Lessons Learned](https://arxiv.org/abs/2209.07858) — Deep Ganguli, Liane Lovitt, Jackson Kernion, Amanda Askell, Yuntao Bai, Saurav Kadavath, Ben Mann, Ethan Perez, Nicholas Schiefer, Kamal Ndousse, Andy Jones, Sam Bowman, Anna Chen, Tom Conerly, Nova DasSarma, Dawn Drain, Nelson Elhage, Sheer El-Showk, Stanislav Fort, Zac Hatfield-Dodds, Tom Henighan, Danny Hernandez, Tristan Hume, Josh Jacobson, Scott Johnston, Shauna Kravec, Catherine Olsson, Sam Ringer, Eli Tran-Johnson, Dario Amodei, Tom Brown, Nicholas Joseph, Sam McCandlish, Chris Olah, Jared Kaplan, Jack Clark (2022)
  Journal: arXiv preprint arXiv:2209.07858; DOI: [10.48550/arXiv.2209.07858](https://doi.org/10.48550/arXiv.2209.07858)
  本文讨论了识别和减轻语言模型有害输出的系统方法，为本次练习中宪法核心的安全性和无害性原则提供了实用见解。
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) — Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, Dario Amodei (2020)
  Journal: arXiv preprint arXiv:2005.14165; DOI: [10.48550/arXiv.2005.14165](https://doi.org/10.48550/arXiv.2005.14165)
  本文展示了大型语言模型通过高级提示（prompting）能够以少量甚至零样本执行广泛任务的能力，这是本节中描述的AI批判步骤的底层基础。
