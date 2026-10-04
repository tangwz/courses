# 评估SFT模型对齐目标

来源：[原文](https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-25-supervised-fine-tuning-alignment/evaluating-sft-models)

[返回章节目录](README.md) · [返回课程目录](../README.md)

评估监督式微调 (fine-tuning)（SFT）模型的主要目标是验证模型是否已更好地与期望行为*对齐 (alignment)*。这项评估有别于单纯继续预训练 (pre-training)的目标，它侧重于评估模型遵循指令、提供有用回复和遵守指定限制的能力，而非原始的语言建模能力（如在通用语料库上的困惑度）。这一评估步骤非常重要，因为它决定了SFT阶段的成功，并常为后续的对齐阶段（如基于人类反馈的强化学习 (reinforcement learning)（RLHF））提供信息。

### 定义评估的对齐 (alignment)目标

在评估之前，需清晰定义SFT旨在实现的特定对齐目的。这些通常包括：

1. **指令遵循**：模型能否准确理解并执行提示中提供的各类指令？这是许多SFT数据集的核心目标。
2. **实用性**：模型生成的回复在提示语境中是否实用、相关且信息丰富？
3. **格式依从**：如果指令指定了特定的输出格式（例如，JSON、markdown列表、特定语气），模型是否遵从？
4. **安全与无害性**：虽然这常是RLHF的主要侧重，但SFT评估也应检查模型是否避免生成有害、偏见或不适当的内容，以及SFT是否在此方面引入了任何退步。

衡量这些品质常需超越标准自动化指标。

### 人工评估：真实情况

对于主观属性，如实用性、复杂任务上的指令遵循忠实度以及无害性，人工评估仍是最可靠的方法。建立有效的人工评估涉及几个考量：

- **任务设计**：创建专门探测目标对齐 (alignment)目的的评估提示。这些提示可以取自SFT数据的保留集、新创建的提示或既定的对齐基准。
- **评估标准**：制定清晰一致的评分准则（标准）。这可以包括李克特量表（例如，实用性1-5分评级）、成对比较（哪个回复更好，A还是B？）或分类判断（模型是否遵循了格式指令：是/否？）。
- **评估者培训**：确保评估者理解任务、标准和潜在偏见。评估者之间的一致性检查很要紧。
- **抽样**：评估每个可能的输入是不可能的。选择涵盖各类指令类型、复杂度和潜在失败模式的代表性提示样本。

虽然强大，但人工评估是资源密集型的（时间、成本），并可能受到注释者间分歧的影响。它常用于验证自动化指标或进行周期性深入评估。

> 使用人工和自动化方法评估SFT模型回复的工作流程。

### 自动化评估方法

为补充人工评估并实现更快迭代，会使用几种自动化方法：

1. **基于模型的评估**：借助一个强大、预先存在的LLM（常被称为“评估器模型”，例如GPT-4、Claude）来评估SFT模型回复的质量。

   - **过程**：为评估器模型设计一个提示，向其提供原始指令、SFT模型的回复，并可能包含一个参考答案或清晰的标准。评估器模型随后输出一个分数或判断。
   - **评估器LLM的示例提示**：

     ```
     你是一名公正的评判员，评估AI助手对用户指令回复的质量。
     指令：“将以下文本总结为三点：\n[此处插入长文本片段...]”
     助手回复：“[此处插入模型生成的摘要...]”

     请根据以下标准评估回复：
     1. 准确性：摘要是否准确反映了原文的主要观点？（1-5）
     2. 简洁性：摘要是否简短扼要？（1-5）
     3. 格式符合性：助手是否使用了正好三点？（是/否）

     请以JSON格式提供您的评分：{"准确性": <分数>, "简洁性": <分数>, "格式符合性": "<是/否>"}
     同时提供您评分的简短理由。
     ```
   - **优点**：可扩展，比人工评估快。可以捕捉到简单指标遗漏的细节。
   - **缺点**：取决于评估器模型的质量和潜在偏见。易受提示敏感性的影响。计算成本可能较高。常需与人工判断进行校准。
2. **基准数据集**：在专为指令遵循或实用性设计的既定基准上评估SFT模型。

   - **例子包括**：
     - **AlpacaEval**：使用GPT-4在Alpaca指令集上自动比较模型输出与参考回复（例如，来自`text-davinci-003`）。
     - **MT-Bench**：一个多轮基准，评估不同类别下的对话和指令遵循能力，常由强大的LLM进行评判。
     - **HELM (语言模型的整体评估)**：虽然范围更广，但HELM包含与指令遵循和鲁棒性等对齐 (alignment)目的相关的特定场景和指标。

   使用基准涉及在基准提示上运行SFT模型，然后使用基准规定的评估协议（常基于模型或基于人工）。
3. **基于参考的指标（谨慎使用）**：ROUGE（用于摘要）或BLEU（用于翻译）等指标可以用于*如果*SFT任务涉及生成应与参考文本（例如，针对特定摘要风格进行微调 (fine-tuning)）紧密匹配的文本。然而，它们常是衡量一般指令遵循或实用性的不佳指标，因为：

   - 它们会惩罚有效的转述和风格变化。
   - 它们不衡量事实正确性或逻辑推理 (inference)。
   - 高重叠分数不保正核心指令被遵循。

   考虑这个简单例子：
   指令：“用一句话解释万有引力。”
   参考：“万有引力是行星或其他物体将物体吸引向其中心的力量。”
   模型A：“万有引力是吸引有质量物体相互靠近的基本力。”（好，但BLEU/ROUGE得分低）
   模型B：“万有引力是一种力。物体被行星吸引到它们的中心。”（尚可，BLEU/ROUGE得分高但流畅度较低）

   在此处使用ROUGE可能会误导性地偏向模型B。

### 实现评估

我们来演示获取回复并准备评估。假设您已使用PyTorch和Hugging Face `transformers`加载了您的SFT模型和分词 (tokenization)器 (tokenizer)。

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 假设模型和分词器已加载
# model = AutoModelForCausalLM.from_pretrained(
#     "path/to/your/sft_model"
# )
# tokenizer = AutoTokenizer.from_pretrained(
#     "path/to/your/sft_model"
# )
# device = torch.device(
#     如果torch.cuda可用则为"cuda"，否则为"cpu"
# )
# model.to(device)
# model.eval() # 将模型设置为评估模式

# --- 实际模型加载的占位符 ---
class DummyModel: # 模拟已加载的模型
    def generate(
        self, input_ids, attention_mask, max_new_tokens, pad_token_id
    ):
        # 根据输入长度模拟生成
        new_tokens = torch.randint(
            100,
            1000,
            (input_ids.shape[0], max_new_tokens),
            device=input_ids.device
        )
        output_ids = torch.cat([input_ids, new_tokens], dim=1)
        return output_ids
class DummyTokenizer: # 模拟已加载的分词器
    def __init__(self):
        self.pad_token_id = 0
    def encode(self, text, return_tensors=None):
        # 非常简单的模拟
        tokens = [101] + [ i+1000 for i in range(len(text.split())) ]
        return torch.tensor([tokens], dtype=torch.long)
    def decode(self, ids, skip_special_tokens=False):
        # 非常简单的模拟
        words = [
            f"word{i-1000}" if i >= 1000 else "[CLS]"
            for i in ids[0].tolist()
        ]
        return " ".join(words)
    def __call__(
        self, text, return_tensors=None, padding=False, truncation=False
    ):
         # 模拟常用的__call__
        encoded = self.encode(text, return_tensors)
        return {"input_ids": encoded, "attention_mask": torch.ones_like(encoded)}

model = DummyModel()
tokenizer = DummyTokenizer()
device = torch.device("cpu") # 示例简化
# --- 占位符结束 ---

def generate_response(prompt_text, model, tokenizer, max_new_tokens=100):
    """从SFT模型生成回复。"""
    inputs = tokenizer(
        prompt_text, return_tensors="pt", padding=True, truncation=True
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id
        )

    # 仅解码新生成的token
    input_length = inputs.input_ids.shape[1]
    generated_ids = output_ids[:, input_length:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response

# 评估提示示例
eval_prompt = (
    "指令：编写一个计算阶乘的简短 Python 函数。"
    "\n回复："
)
# 注意：好的SFT格式包括清晰的分隔符，例如
# \n回复：

generated_text = generate_response(eval_prompt, model, tokenizer)

print(f"评估提示：\n{eval_prompt}")
print(f"\n生成的回复：\n{generated_text}")

# --- 接下来的步骤将是：---
# 1. 将 `eval_prompt` 和 `generated_text` 发送给人工评估者。
# 2. 或者，为基于模型的评估器（如GPT-4）格式化它们。
# 3. 或者，如果作为基准的一部分，使用基准的特定评估
#    脚本。
# 4. 或者，应用更简单的检查（例如，检查生成的文本中是否包含 'def' 和 'return'）——有限但快速。
```

### 评估方法比较

不同的评估方法提供不同的信号。将它们结合使用常有益处。例如，使用自动化指标/基准进行广泛覆盖和频繁检查，并定期使用人工评估来验证自动化结果并检查不明显的问题。



[交互图表：SFT评估分数](https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-25-supervised-fine-tuning-alignment/evaluating-sft-models#plot-ibogud)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "data": [
    {
      "type": "bar",
      "x": [
        "指令遵循",
        "实用性",
        "安全性（基本检查）"
      ],
      "y": [
        4.2,
        3.8,
        4.5
      ],
      "name": "人工评分（平均1-5）",
      "marker": {
        "color": "#7048e8"
      }
    },
    {
      "type": "bar",
      "x": [
        "指令遵循",
        "实用性",
        "安全性（基本检查）"
      ],
      "y": [
        3.9,
        3.5,
        4.1
      ],
      "name": "GPT-4评分（平均1-5）",
      "marker": {
        "color": "#15aabf"
      }
    },
    {
      "type": "bar",
      "x": [
        "指令遵循",
        "实用性",
        "安全性（基本检查）"
      ],
      "y": [
        0.65,
        0.55,
        0.9
      ],
      "name": "自动化检查（通过率）",
      "marker": {
        "color": "#f59f00"
      }
    }
  ],
  "layout": {
    "title": "SFT评估分数",
    "barmode": "group",
    "yaxis": {
      "title": "分数 / 评级"
    },
    "xaxis": {
      "title": "对齐目的"
    },
    "font": {
      "family": "sans-serif"
    },
    "legend": {
      "title": {
        "text": "评估方法"
      }
    }
  }
}
```

</details>



> 评估SFT模型对齐 (alignment)度的不同方法的分数比较。注意自动化检查与人工或基于模型的评分可能存在的差异。

归根结底，有效评估SFT模型需要清晰理解对齐目的，并周全结合人类洞察力和可扩展的自动化技术。评估结果指引进一步的微调 (fine-tuning)工作，助力创建不仅能干，而且真正实用可靠的LLM。

## 参考资料

- [Judging LLM-as-a-judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) — Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, Ion Stoica (2023)
  Journal: NeurIPS 2023 Datasets and Benchmarks Track; Publisher: NeurIPS; DOI: [10.48550/arXiv.2306.05685](https://doi.org/10.48550/arXiv.2306.05685)
  提出了MT-Bench，用于评估多轮指令遵循和对话能力，并严格分析了使用大型语言模型作为其他LLM评估者的有效性和局限性。
- [HELM: Holistic Evaluation of Language Models](https://arxiv.org/abs/2211.09110) — Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian Zhang, Deepak Narayanan, Yuhuai Wu, Ananya Kumar, Benjamin Newman, Binhang Yuan, Bobby Yan, Ce Zhang, Christian Cosgrove, Christopher D. Manning, Christopher Ré, Diana Acosta-Navas, Drew A. Hudson, Eric Zelikman, Esin Durmus, Faisal Ladhak, Frieda Rong, Hongyu Ren, Huaxiu Yao, Jue Wang, Keshav Santhanam, Laurel Orr, Lucia Zheng, Mert Yuksekgonul, Mirac Suzgun, Nathan Kim, Neel Guha, Niladri Chatterji, Omar Khattab, Peter Henderson, Qian Huang, Ryan Chi, Sang Michael Xie, Shibani Santurkar, Surya Ganguli, Tatsunori Hashimoto, Thomas Icard, Tianyi Zhang, Vishrav Chaudhary, William Wang, Xuechen Li, Yifan Mai, Yuhui Zhang, Yuta Koreeda (2023)
  Journal: Transactions on Machine Learning Research (TMLR); DOI: [10.48550/arXiv.2211.09110](https://doi.org/10.48550/arXiv.2211.09110)
  提出了一套全面的框架、场景和指标，用于在广泛维度上评估大型语言模型，包括指令遵循和鲁棒性。

---

[上一节](05-SFT%E8%AE%AD%E7%BB%83%E8%BF%87%E7%A8%8B%E4%B8%8E%E8%B6%85%E5%8F%82%E6%95%B0.md) · [下一节](../26-%E4%BA%BA%E7%B1%BB%E5%8F%8D%E9%A6%88%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%20%28RLHF%29/01-RLHF%20%E6%B5%81%E7%A8%8B%E6%A6%82%E8%BF%B0.md)
