---
course: "llm-constitutional-ai-rlaif"
chapter: "implementing-constitutional-ai"
lesson: "implementing-ai-revision"
sourceId: 4472
sourceUrl: "https://apxml.com/zh/courses/llm-constitutional-ai-rlaif/chapter-3-implementing-constitutional-ai/implementing-ai-revision"
title: "实施AI修订模型"
description: "构建或微调模型以根据评论修订回复。"
order: 4
plots: []
sourceHash: "3588bda6fe1ac670649799c754bcca3b8b5fefef24bbcc977ab8049cc78ab350"
sourceCorrections: []
---

在根据章程生成评论之后，宪法AI（CAI）流程中的下一个顺理成章的步骤是修订初始LLM回复以解决这些评论中提出的问题。本部分详细介绍负责生成这些修订的组件的具体实施。与从头训练模型不同，这通常涉及运用一个强大的现有LLM，在精心构建的提示引导下，来执行修订任务。目标是生成一个修订后的回复，$R_{revised}$，该回复结合了评论$C$中的反馈，使其与章程$\mathcal{K}$的符合程度高于初始回复$R_{initial}$。

### 运用大型语言模型进行修订

核心思路是利用LLM的指令遵循能力来执行修订。在此背景下，“AI修订模型”通常指的是协调此修订生成的过程或系统，而不是一个单独、独立训练的模型架构（尽管那是一种可能但资源消耗更大的方法）。通常，您会为此任务使用一个有能力的（基础）LLM（$M_{base}$），或者如果规模或能力有要求，甚至可能是一个更强大的模型（$M_{prompt}$，也许是基于API的模型）。

此过程的输入是在前一步骤中生成的对 ($R_{initial}$, $C$)。我们旨在生成的输出是 $R_{revised}$。

### 编写有效的修订提示

生成修订的质量在很大程度上取决于提供给LLM（$M_{prompt}$）的提示。提示必须清楚地指示模型根据$C$中提出的问题*具体地*修改$R_{initial}$，同时理想情况下保留原始回复有帮助的部分。

这里有两种模板结构。请记住，最佳的提示工程 (prompt engineering)通常需要根据具体的LLM和任务细节进行迭代。

**模板1：直接修订指令**

```markdown
[INST] 您的任务是根据一份源自章程的评论来修订一个AI回复。
原始回复：
<response>
{initial_response}
</response>

评论（识别章程违规）：
<critique>
{critique}
</critique>

*仅*根据所提供的评论，请修订原始回复，以解决已识别的问题，并确保其符合评论中提及或暗示的原则。在可能的情况下，保持原始回复的意图和有用性，将修改重点放在解决评论上。*只*输出修订后的回复。
[/INST]
修订后的回复：
```

**模板2：强调特定原则（如果评论中提供）**

```markdown
[INST] 原始回复：
{initial_response}

评论（违反原则：{list_of_violated_principles}）：
{critique}

修订原始回复以解决评论中描述的问题，并特别注意违反的原则：{list_of_violated_principles}。修订后的回复应符合评论中体现的章程。
直接输出修订后的回复，无需前言。
[/INST]
修订后的回复：
```

**提示工程考量：**

- **清晰度：** 对任务要明确：基于*评论*进行修订。
- **约束：** 明确指示模型*只*输出修订后的回复，以简化解析。
- **忠实性：** 鼓励在评论未直接针对原始回复的情况下，保留其有用性。
- **少样本示例（进阶）：** 对于复杂的修订，在提示中包含1-3个高质量的（初始回复、评论、修订回复）示例可以显著提高LLM的性能。这需要策划有代表性的示例。
- **思维链（进阶）：** 您可以尝试让LLM首先解释它将*如何*应用评论，*然后*生成修订的提示。这个中间推理 (inference)步骤有时可以促成更周到的修订，尽管它增加了生成成本和复杂性。

### 修订生成工作流程

自动化修订生成涉及遍历您的 ($R_{initial}$, $C$) 对并使用LLM为每个对生成 $R_{revised}$。

> 使用LLM并以初始回复及其评论作为提示，生成单个修订后的回复的数据流。

以下是一个演示自动化的Python代码片段：

```python
import logging # 使用日志进行更好的追踪

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 假设 llm_service 是一个预配置的LLM API客户端或本地模型
# critique_data 是一个字典列表：[{'initial_response': str, 'critique': str, 'metadata': {...}}]

REVISION_PROMPT_TEMPLATE = """
[INST] Original Response:
<response>
{initial_response}
</response>

Critique:
<critique>
{critique}
</critique>

Revise the original response to specifically address the issues raised in the critique, ensuring alignment with constitutional principles reflected in the critique. Preserve helpfulness where appropriate. Output only the revised response.
[/INST]
Revised Response:
"""

def generate_revisions(critique_data, llm_service, prompt_template):
    """为评论列表生成修订。"""
    revised_data = []
    for i, item in enumerate(critique_data):
        prompt = prompt_template.format(
            initial_response=item['initial_response'],
            critique=item['critique']
        )
        try:
            # 示例参数，根据LLM提供商/模型进行调整
            response = llm_service.generate(
                prompt=prompt,
                max_new_tokens=len(item['initial_response']) + 512, # 启发式最大长度
                temperature=0.4, # 较低的温度可获得更集中的修订
                stop_sequences=["[INST]", "\n\nHuman:", "\n\nAssistant:"] # 防止生成过长内容
            )
            revised_text = response.strip() # 基本清理

            if not revised_text:
                 logging.warning(f"项目 {i} 生成了空修订。跳过。")
                 continue

            item['revised_response'] = revised_text
            revised_data.append(item)
            if (i + 1) % 100 == 0: # 定期记录进度
                 logging.info(f"已为 {i+1}/{len(critique_data)} 个项目生成修订。")

        except Exception as e:
            logging.error(f"处理项目 {i} 时出错：{e}", exc_info=True)
            # 根据需要实现重试逻辑或跳过有问题项目

    return revised_data

# 示例用法：
# 假设 critique_outputs 包含来自评论器步骤的数据
# revised_dataset = generate_revisions(critique_outputs, my_llm_client, REVISION_PROMPT_TEMPLATE)
# logging.info(f"成功生成了 {len(revised_dataset)} 条修订。")

# revised_dataset 现在包含三元组（如果保留元数据则包含更多信息）
# 适合在下一步中构建SFT数据集。
```

### 质量控制与迭代

$M_{prompt}$ 生成的修订不能保证完美。常见的失败模式包括：

- **修订不完整：** 修订未能解决评论中的所有点。
- **引入新问题：** 修订解决了原始问题，但引入了新的章程违规或事实错误。
- **有用性降低：** 修订变得过于谨慎或僵硬，失去了原始回复的价值。
- **过度修正：** 修订修改了评论未针对的回复部分。

缓解策略包括：

1. **提示优化：** 根据观察到的失败模式，迭代改进修订提示。
2. **模型选择：** 为修订任务使用可用的最有能力的LLM（$M_{prompt}$）。
3. **筛选：** 实施自动化检查。例如，您可以再次使用*AI评论器模型*，根据章程对$R_{revised}$进行评分。在某些维度上仍然得分低或得分低于$R_{initial}$的修订可能会被丢弃或标记 (token)以供审查。
4. **迭代修订：** 对于关键应用，可以实现一个循环，其中$R_{revised}$再次被评论，并进行第二次修订尝试。这会显著增加计算成本。

### 进阶考量

- **专用修订模型：** 尽管通常依赖于提示通用LLM，但您可以专门为修订任务微调 (fine-tuning)一个单独的模型。这需要一个高质量的($R_{initial}$, $C$, $R_{target}$)示例数据集，其中$R_{target}$是人类验证过或明显更优的修订。这需要大量资源，但可以产生一个更高效和专用的修订组件。
- **保持一致性：** 确保修订模型不大幅改变原始回复的人格或核心功能，需要仔细的提示，如果训练了专用模型，可能还需要在微调期间施加特定约束。

### 准备进行微调 (fine-tuning)

此修订生成过程的输出是一个数据集，至少包含元组($R_{initial}$, $C$, $R_{revised}$)。这个数据集是CAI监督学习 (supervised learning)阶段下一个重要步骤的依据：即构建最终数据集并微调目标LLM（$M_{SFT}$），使其内化章程原则，详细内容将在下一节中说明。此处生成的修订质量直接影响最终微调阶段的有效性。

## 参考资料

- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, Carol Chen, Catherine Olsson, Christopher Olah, Danny Hernandez, Dawn Drain, Deep Ganguli, Dustin Li, Eli Tran-Johnson, Ethan Perez, Jamie Kerr, Jared Mueller, Jeffrey Ladish, Joshua Landau, Kamal Ndousse, Kamile Lukosuite, Liane Lovitt, Michael Sellitto, Nelson Elhage, Nicholas Schiefer, Noemi Mercado, Nova DasSarma, Robert Lasenby, Robin Larson, Sam Ringer, Scott Johnston, Shauna Kravec, Sheer El Showk, Stanislav Fort, Tamera Lanham, Timothy Telleen-Lawton, Tom Conerly, Tom Henighan, Tristan Hume, Samuel R. Bowman, Zac Hatfield-Dodds, Ben Mann, Dario Amodei, Nicholas Joseph, Sam McCandlish, Tom Brown, Jared Kaplan (2022)
  Journal: arXiv; DOI: [10.48550/arXiv.2212.08073](https://doi.org/10.48550/arXiv.2212.08073)
  介绍了宪法式AI框架，详细说明了其管道，包括AI评论和修订过程。
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903) — Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, Denny Zhou (2022)
  Journal: arXiv; DOI: [10.48550/arXiv.2201.11903](https://doi.org/10.48550/arXiv.2201.11903)
  介绍了思维链提示，这是一种增强大型语言模型推理并生成更周到回复的先进技术，与质量控制和迭代有关。
