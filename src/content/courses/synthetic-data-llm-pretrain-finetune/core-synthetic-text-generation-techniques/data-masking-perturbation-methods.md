---
course: "synthetic-data-llm-pretrain-finetune"
chapter: "core-synthetic-text-generation-techniques"
lesson: "data-masking-perturbation-methods"
sourceId: 6125
sourceUrl: "https://apxml.com/zh/courses/synthetic-data-llm-pretrain-finetune/chapter-2-core-synthetic-text-generation-techniques/data-masking-perturbation-methods"
title: "数据掩码和数据扰动技术"
description: "理解匿名化数据和应用扰动以创建合成变体的方法。"
order: 6
plots: []
sourceHash: "0b2d14fc4beccc275500c3585968d75e617eb859c86f04fa68773b70474bf8e4"
sourceCorrections: []
---

我们接下来看看改变现有数据以生成新变体的方法：数据掩码和数据扰动。这些技术在您需要扩充数据集、保护敏感信息或引入可控变动以降低模型对微小输入变化的敏感度时尤其有用。尽管有些生成方法是从头创建文本，但掩码和扰动通常是从现有数据或预定义模板开始。

### 数据掩码：信息匿名化和泛化

数据掩码是将文本数据中的敏感或可识别信息替换为通用占位符、匿名值或更广泛类别的方法。其主要目标通常是保护隐私，使您能够使用数据训练大型语言模型，而不会泄露机密信息。它也可以是生成合成数据的一种方式，即特定实体被抽象化，帮助模型关注模式而非具体内容。

#### 掩码技术

1. **使用占位符或通用值进行替换：**
   这是一种广泛使用的方法，其中姓名、地址、电话号码或社会安全号码等特定实体被预定义的标记 (token)或生成列表中的值替换。

   - **标记：** 常用占位符包括 `[姓名]`、`[位置]`、`[组织]`、`[日期]`。
   - **通用值：** 您可能将“Alice Johnson”替换为“Jane Doe”，或者使用为创建虚假数据设计的库随机生成一个姓名。

   例如，句子：
   "John Smith, residing at 123 Main St, Anytown, called us on 05/15/2024 regarding order #XN789."
   可以掩码为：
   "[NAME], residing at [ADDRESS], [CITY], called us on [DATE] regarding order #[ORDER\_ID]."
2. **泛化：**
   与精确替换不同，您可以将特定值泛化为更广泛的类别。这会降低精确度，但可以保持数据的结构完整性和在某些任务中的有效性。

   - “年龄：47”可以变为“年龄：40-50 岁范围。”
   - “SpecificModelX\_Version3.2”可以变为“[产品型号版本]。”
3. **删改：**
   在某些情况下，信息可能会被完全移除。虽然这更直接地关乎匿名化而非纯粹的合成，但由此产生的缺失部分数据可被视为一种合成变体。例如，在客户评论数据集中删改所有特定公司名称的提及。

#### 何时使用数据掩码

- **隐私敏感应用：** 在训练包含个人身份信息 (PII) 或其他机密信息的数据时必不可少。
- **抽象具体内容：** 如果任务需要理解一般模式而非特定实体（例如，客户支持意图分类中，客户姓名通常与意图本身无关）时很有用。
- **创建模板式数据：** 掩码数据可以作为模板，通过填充占位符（或许使用其他生成技术）来生成更多样化的合成样本。

保持平衡非常重要：过度掩码虽然能有效保护隐私，但可能会移除过多有用的上下文 (context)信息，从而可能降低模型性能。目的是仅掩码您的特定目标所需的内容，同时保留数据效用。

### 数据扰动：引入可控变动

数据扰动是指对现有文本样本进行微小且通常是随机的改动，以创建新的、略有不同的版本。目标是增加数据集的多样性，从而可能降低大型语言模型对微小输入变化的敏感度。可以将其视为一种“抖动”数据点的方式，以覆盖现有样本周围更多的输入空间。

#### 文本扰动常见策略

1. **同义词替换：**
   句子中的词语被替换为其同义词。这可以带来词汇多样性。

   - 原始："The *quick* brown fox *jumps* over the lazy dog."
   - 扰动后："The *fast* brown fox *leaps* over the lazy dog."
     这里需要注意，因为并非所有同义词都完全适用于所有语境，有时可能会细微或大幅改变含义。
2. **随机插入：**
   随机向句子中插入词语（通常是常用词或相邻词的同义词）。

   - 原始："The cat sat."
   - 扰动后："The *fluffy* cat sat *quietly*."
     应谨慎使用此方法，因为如果不加限制，它很容易导致语法错误或无意义的句子。
3. **随机删除：**
   随机从句子中移除词语。

   - 原始："The quick brown fox jumps over the lazy dog."
   - 扰动后："The quick fox jumps over lazy dog."
     同样，这会影响语法和含义，因此删除的概率通常应较低。
4. **随机交换：**
   随机交换句子中两个词语的位置。

   - 原始："The quick brown fox."
   - 扰动后："The brown quick fox."
     这更可能破坏自然语言结构，应谨慎应用，或许对交换词语的距离进行限制。
5. **字符级扰动：**
   在字符层面引入微小变化，例如：

   - **错别字：** 模拟常见打字错误（例如，“important” -> “imporant”）。
   - **字符交换/插入/删除：** 类似于词语级别，但粒度更细。
     这有助于使模型在处理嘈杂输入时表现良好，例如来自光学字符识别 (OCR) 或非正式用户生成内容的文本。

以下图示说明了在简单句子上的一些扰动可能性。

> 这是一个扰动技术的图示，如同义词替换、删除、词语交换和插入应用于原始句子。

#### Python 示例：简单同义词替换

我们来呈现一个非常基本的同义词替换。在实践中，您会使用更复杂的同义词库，可能与自然语言处理 (NLP) 库集成，或使用词嵌入 (embedding)模型根据语境找到合适的同义词。

```python
import random

def simple_synonym_perturbation(text, synonym_map, probability=0.1):
    words = text.split()
    perturbed_words = []
    for word in words:
        # 检查单词是否在映射中，以及是否应根据概率进行扰动
        if random.random() < probability and word.lower() in synonym_map:
            synonyms = synonym_map[word.lower()]
            # 如果可能，保留原始大小写，否则直接使用同义词
            chosen_synonym = random.choice(synonyms)
            if word.istitle():
                perturbed_words.append(chosen_synonym.title())
            elif word.isupper():
                perturbed_words.append(chosen_synonym.upper())
            else:
                perturbed_words.append(chosen_synonym)
        else:
            perturbed_words.append(word)
    return " ".join(perturbed_words)

# 一个非常小且具说明性的同义词映射（全部小写）
example_synonym_map = {
    "quick": ["fast", "swift", "rapid"],
    "happy": ["joyful", "pleased", "content"],
    "big": ["large", "huge", "enormous"]
}

original_sentence = "The Quick dog was very Happy with the BIG bone."
# 对符合条件的词语应用 30% 概率的扰动
perturbed_sentence = simple_synonym_perturbation(original_sentence, example_synonym_map, probability=0.3)

print(f"原始: {original_sentence}")
print(f"扰动后: {perturbed_sentence}")
# 可能的输出：
# Original: The Quick dog was very Happy with the BIG bone.
# Perturbed: The Swift dog was very Pleased with the HUGE bone.
```

> 此代码片段提供了一种执行同义词替换的基本方法，并尝试保留大小写。对于更高级的应用，建议使用提供更丰富同义词库或词嵌入相似度的自然语言处理 (NLP) 库，以便进行语境合适的替换。

#### 何时使用数据扰动

- **数据增强：** 当您拥有有限的数据集并希望用相似但不完全相同的示例来扩展它时。
- **提高模型鲁棒性：** 训练模型时，使其对输入措辞或常见错误中的微小变化不那么敏感。
- **模拟噪声：** 对于输入数据可能包含错别字、非正式语言或其他不规则的任务。

### 平衡之举：掩码、扰动和数据质量

掩码和扰动都是有用的工具，但它们需要谨慎应用。目标是生成对您的大型语言模型*有益*的合成数据。

- **主要保留语义：** 尽管扰动旨在引入变动，但您通常希望避免大幅改变原始含义或意图的修改，除非这是一个特定目标（例如，生成反事实，这涉及更高级的技术）。
- **保持可读性和语法：** 合成文本理想情况下应流畅且语法正确，特别是如果它旨在供人阅读，或者大型语言模型正在从中学习语言结构。
- **避免引入偏见：** 如果您的掩码或扰动规则存在偏见（例如，总是用特定的刻板占位符替换某些姓名，或仅扰动某些类型的短语），您的合成数据可能会继承并放大这些偏见。
- **与任务的相关性：** 掩码或扰动的类型和程度应与下游任务保持一致。例如，字符级扰动可能与拼写纠正模型更相关，而不是与摘要模型。

在实践中，您通常会将这些技术与本章讨论的其他生成方法结合使用。例如，您可以使用大型语言模型生成初始文本（如“使用大型语言模型生成合成样本”中涉及的），然后应用掩码或扰动来进一步丰富输出或为满足特定的隐私要求做准备。

随着我们继续，请记住这些核心技术是构建模块。其有效性来自于深思熟虑地结合它们并评估其对大型语言模型性能和行为的影响。本章后面的实践练习将为您提供使用大型语言模型 API 进行生成的机会，您可以考虑这些掩码和扰动方法如何补充此类输出。

## 参考资料

- [Anonymizing Health Data](https://link.springer.com/book/10.1007/978-3-319-94101-2) — Khaled El Emam, and Lydia Aysegul Neşeoğlu (2018)
  Publisher: Springer; DOI: [10.1007/978-3-319-94101-2](https://doi.org/10.1007/978-3-319-94101-2)
  本书全面介绍了去标识化方法，包括掩码、泛化和编辑，用于保护文本和其他数据类型中的敏感信息。
- [Data Augmentation in NLP: A Survey](https://eudl.eu/doi/10.4108/eetair.v5i2.27448) — Sanjana Gupta, R. M. K. R. Namburi, Sai Nikhil, M. Ramakrishna, and P. S. R. N. Praveen (2023)
  Journal: EAI Endorsed Transactions on AI and Robotics; Publisher: European Alliance for Innovation; Volume: 5; Pages: e2; DOI: [10.4108/eetair.v5i2.27448](https://doi.org/10.4108/eetair.v5i2.27448)
  一份最新的调查，广泛概述了自然语言处理中特有的数据增强技术，包括与生成合成文本变体相关的技术。
- [Guide to Protecting the Confidentiality of Personally Identifiable Information (PII)](https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-122.pdf) — Marianne Swanson, Amy Bell, Marshall Abrams, and David Phillips (2010)
  Journal: NIST Special Publication 800-122; Publisher: National Institute of Standards and Technology (NIST)
  这份NIST出版物提供了保护个人身份信息（PII）的重要指导，包括去标识化和掩码技术，以降低隐私风险。
