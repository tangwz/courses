---
course: "how-to-build-a-large-language-model"
chapter: "analyzing-model-behavior"
lesson: "identifying-failure-modes"
sourceId: 6070
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-23-analyzing-model-behavior/identifying-failure-modes"
title: "找出失效模式"
description: "寻找模型在哪些输入类型或情境中表现不佳的方法。"
order: 5
plots: []
sourceHash: "2269d44cba84a67c0a63c84da5952473bf3eefcf689022efde52d41f4b2f8e25"
sourceCorrections: []
---

尽管困惑度等指标以及下游任务表现能提供关于大型语言模型能力的有用信息，但它们无法完整描绘模型的可靠性或潜在不足。平均表现良好的模型仍可能在特定情况下出现问题行为。找出这些“失效模式”，即模型产生不正确、有偏见、不安全或其他不良输出的情况，是理解和改进大型语言模型的重要组成部分。这个过程不止于总体分数，旨在找出具体弱点，从而能够进行有针对性的干预，并构建更值得信赖的系统。

失效模式不仅是学术上的好奇点；它们在部署大型语言模型时带来真实风险。生成事实不准确信息的模型可能会误导用户，而放大偏见的模型则可能延续社会危害。了解这些可能发生的失效，对于调试、改进对齐 (alignment)策略（例如稍后会讨论的SFT和RLHF），以及确保应用得到负责任的开发，都具有重大意义。

### 失效模式的常见类别

大型语言模型失效以多种方式表现出来。认识这些模式有助于设计有效的测试：

1. **事实不准确（幻觉 (hallucination)）：** 这也许是讨论最多的失效。模型生成的文本听起来合理且语法正确，但事实错误或毫无意义。这通常发生在模型缺乏特定知识或试图超出其训练数据范围进行推断时。

   - *示例：* 询问一个近期鲜为人知的科学发现，可能会导致模型编造细节或混淆不同背景的事实。
2. **偏见放大：** 在大量互联网文本数据集上训练的模型不可避免地会学习到数据中存在的社会偏见。它们可能会再现甚至放大与性别、种族、职业或其他特征相关的刻板印象。

   - *示例：* 涉及某些职业的提示可能会总是引发假设特定性别的回应，这反映的是历史偏见而非当前现实。
3. **逻辑不一致和矛盾：** 模型可能会在单个回复中或在对话的多轮中自相矛盾。它也可能无法完成对人类来说似乎微不足道的基本逻辑推理 (inference)任务。

   - *示例：* 在同一段解释中先说“所有鸟类都会飞”，然后又提到“企鹅是不会飞的鸟”。
4. **指令遵循错误：** 特别是在复杂或多部分的提示下，模型可能会忽略限制、误解否定词，或未能遵循要求的格式或角色。

   - *示例：* 要求模型“写一个关于猫的故事，*不要*使用字母‘e’”，但结果故事中却大量出现了字母‘e’。
5. **输入扰动的敏感性：** 对输入提示进行细微的、语义无关的改变（例如，添加一个空格、更换一个同义词、轻微改写）有时会导致截然不同的输出，显露模型的不稳定性。

   - *示例：* “告诉我马来西亚的首都是什么。”可能会得到一个好的答案，而“告诉我马来西亚的首都城市是什么？”则可能会让模型困惑或产生质量较低的回复。
6. **对抗性漏洞：** 模型可能容易受到专门制作的输入的影响，这些输入旨在绕过安全过滤器或引出不正确的输出。这些“对抗性攻击”以非预期的方式利用了已学习到的模式。

   - *示例：* 精心构建的提示（对人类来说有时毫无意义）可能会触发模型生成它通常会拒绝的有害内容。
7. **重复或无意义的输出：** 在某些条件下（例如，非常长的生成上下文 (context)、特定的采样设置或模糊的提示），模型可能会陷入重复循环或退化成不连贯的文本。

### 找出失效模式的方法

找出这些弱点需要使用更具针对性的方法：

#### 针对性测试套件

创建或使用专门设计用于检查已知弱点区域的数据集。这包括制作可能引出特定失效模式的提示。

- **偏见探测：** BBQ（QA偏见基准）或Winogender Schemas等数据集包含旨在显现刻板印象关联的提示。评估模型在这些数据集上的响应可以量化 (quantization)偏见。
- **事实核查：** 使用专注于特定知识类别（科学、历史、近期事件）的问答数据集，其中真实情况已知。将模型输出与事实数据库进行比较。
- **指令遵守测试：** 开发带有复杂限制（否定、格式要求、长度限制）的提示，并评估模型是否遵守。

这是一个PyTorch代码片段，说明了如何检查像生成禁用词这样的简单失效模式：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载您的模型和分词器
model_name = "gpt2" # 替换为您的模型
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.eval() # 设置为评估模式

def check_forbidden_word(prompt, forbidden_word, max_new_tokens=50):
    """
    给定一个提示，检查模型是否生成了特定的禁用词。
    如果找到禁用词则返回True，否则返回False。
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        # 使用模型生成文本
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False # 此处使用贪婪解码以确保可复现性
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 简单检查禁用词是否出现在生成部分
    generated_portion = generated_text[len(prompt):]
    print(
        f"提示: {prompt}\n生成: {generated_portion[:100]}..."
    ) # 打印以便检查
    return forbidden_word.lower() in generated_portion.lower()

# 示例测试用例
prompt_template = "Describe the following animal: {}"
animal = "penguin"
forbidden = "fly"
prompt = prompt_template.format(animal)

failure_detected = check_forbidden_word(prompt, forbidden)

if failure_detected:
    print(
        f"\n失效已识别: 模型在描述'{animal}'时生成了'{forbidden}'。"
    )
else:
    print(
        f"\n测试通过: 模型在描述'{animal}'时未生成'{forbidden}'。"
    )
```

这个简单示例检查一个特定的关键词，但更复杂的测试将涉及语义分析、检查逻辑一致性，或与事实数据库进行比较。

#### 对抗性测试（红队演练）

这涉及人工测试人员积极尝试让模型失效。红队成员利用他们的创造力以及对模型潜在弱点的理解，制作自动化测试可能遗漏的挑战性提示。他们可能会尝试：

- 规避安全准则。
- 在棘手的主题上诱发幻觉 (hallucination)。
- 通过场景显露偏见。
- 测试指令遵循的极限。

红队演练对于发现意想不到的失效模式以及理解模型能力和安全限制的*边界*来说价值很高。

#### 边缘情况压力测试

在统计上稀有或超出典型使用边界的输入上评估模型：

- **非常长或复杂的提示：** 模型是否保持上下文 (context)和连贯性？
- **包含冲突信息的提示：** 模型如何处理矛盾？
- **领域外请求：** 模型如何优雅地处理远超其训练数据的主题？
- **具有模糊要求的代码生成：** 它能处理复杂的编程逻辑或不熟悉的库吗？

#### 分析异常分布（OOD）行为

系统地测试与模型训练分布明显不同的输入。这可能包括：

- **不同语言或方言**（如果模型主要训练于一种）。
- **高度专业化术语**，来自训练数据中未充分体现的方面。
- **不同文本格式**（例如，表格、结构化数据），如果主要训练于散文。

#### 输出模式分析

有时，失效表现为输出中的统计异常。监控以下情况：

- **高重复率：** 使用N-gram重叠等指标来识别过度重复。
- **低多样性：** 回复是否变得过于通用或模板化？
- **异常的Token概率：** 检查模型为token分配异常高或低概率的序列。

一个简单的重复检查：

```python
from collections import Counter

def calculate_repetition_rate(text, n=3):
    """计算重复N-gram的比例。"""
    words = text.split()
    if len(words) < n:
        return 0.0
    ngrams = [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return 0.0
    counts = Counter(ngrams)
    repeated_ngrams = sum(1 for count in counts.values() if count > 1)
    return repeated_ngrams / len(ngrams)

# 假设 'generated_portion' 包含模型的输出
# 来自上一个示例
rep_rate = calculate_repetition_rate(generated_portion, n=4)
# 检查4-gram重复
print(f"4-gram重复率: {rep_rate:.2f}")

# 定义失效阈值
repetition_threshold = 0.1
if rep_rate > repetition_threshold:
     print("潜在失效: 输出中识别到高重复。")
```

#### 运用可解释性工具

尽管注意力可视化和探测等技术（本章其他部分讨论）主要目的是理解模型*如何*工作，但它们有时能帮助诊断失效*为何*发生。例如，异常的注意力模式或表明对某个特定想法存在困惑的探测结果，可能与相关输入上观察到的失效相关。

找出失效模式并非一次性任务，而是一个持续的过程。随着模型的演进并应用于新方面，需要持续的测试和分析，以了解它们的局限性，并确保它们安全有效地使用。从失效分析中获得的认识直接指导模型改进、数据管理策略以及更好的对齐 (alignment)技术的开发。

## 参考资料

- [Red Teaming Language Models to Reduce Harms: Methods, Scaling Behaviors, and Lessons Learned](https://arxiv.org/abs/2209.07858) — Ganguli, Deep, Lovitt, Liane, Kernion, Jackson, Askell, Amanda, Bai, Yuntao, Kadavath, Saurav, Mann, Ben, Perez, Ethan, Schiefer, Nicholas, Ndousse, Kamal, Jones, Andy, Bowman, Sam, Chen, Anna, Conerly, Tom, DasSarma, Nova, Drain, Dawn, Elhage, Nelson, El-Showk, Sheer, Fort, Stanislav, Hatfield-Dodds, Zac, Henighan, Tom, Hernandez, Danny, Hume, Tristan, Jacobson, Josh, Johnston, Scott, Kravec, Shauna, Olsson, Catherine, Ringer, Sam, Tran-Johnson, Eli, Amodei, Dario, Brown, Tom, Joseph, Nicholas, McCandlish, Sam, Olah, Chris, Kaplan, Jared, Clark, Jack (2022)
  Journal: arXiv preprint arXiv:2209.07858; DOI: [10.48550/arXiv.2209.07858](https://doi.org/10.48550/arXiv.2209.07858)
  详细介绍了对大型语言模型进行红队测试的方法和发现，旨在发现和减轻有害输出。
- [Holistic Evaluation of Language Models](https://doi.org/10.48550/arXiv.2211.09110) — Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian Zhang, Deepak Narayanan, Yuhuai Wu, Ananya Kumar, Benjamin Newman, Binhang Yuan, Bobby Yan, Ce Zhang, Christian Cosgrove, Christopher D. Manning, Christopher Ré, Diana Acosta-Navas, Drew J. Hudson, Eric Zelikman, Esin Durmus, Faisal Ladhak, Frieda Rong, Hongyu Ren, Huaxiu Yao, Jue Wang, Keshav Santhanam, Laurel Orr, Lucia Zheng, Mert Yuksekgonul, Mirac Suzgun, Nathan Kim, Neel Guha, Niladri Chatterji, Omar Khattab, Peter Henderson, Qian Huang, Ryan Chi, Sang Michael Xie, Shibani Santurkar, Surya Ganguli, Tatsunori Hashimoto, Thomas Icard, Tianyi Zhang, Vishrav Chaudhary, William Wang, Xuechen Li, Yifan Mai, Yuhui Zhang, Yuta Koreeda (2023)
  Journal: Transactions on Machine Learning Research (TMLR); DOI: [10.48550/arXiv.2211.09110](https://doi.org/10.48550/arXiv.2211.09110)
  提出了一个全面评估语言模型各项能力的框架，包含鲁棒性和公平性，与识别各类故障模式相符。
