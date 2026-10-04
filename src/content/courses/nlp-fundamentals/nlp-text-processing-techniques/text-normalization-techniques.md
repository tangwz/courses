---
course: "nlp-fundamentals"
chapter: "nlp-text-processing-techniques"
lesson: "text-normalization-techniques"
sourceId: 2479
sourceUrl: "https://apxml.com/zh/courses/nlp-fundamentals/chapter-1-nlp-text-processing-techniques/text-normalization-techniques"
title: "文本标准化方法"
description: "应用大小写转换、处理缩写词和移除重音符号等文本标准化方法。"
order: 6
plots: []
sourceHash: "36764b414e4aa3d2e52c630515cabe2b6307a239443b0a9dfe72cdf7c5cdc208"
sourceCorrections: []
---

原始文本数据常不一致。大小写、标点符号、重音符号和缩写形式的差异，会让算法难以识别不同的字符串实际指代的是同一事物。文本标准化是一组将文本转换为更标准、统一格式的方法，旨在减少这些差异并提高数据的一致性。这一步对于准备文本进行后续的自然语言处理（NLP）任务，如特征提取和模型训练，非常重要。

让我们了解几种常见的文本标准化方法。

### 大小写转换

最简单也最常用的标准化方法之一是大小写转换，它通常是将文本中的所有字符转换为小写。

**为何使用？**
思考词语“Apple”（指公司）和“apple”（指水果）。如果你的分析不需要区分专有名词和普通名词（基于大小写），将两者都转换为“apple”可确保它们被视为相同的标记 (token)。这能减少总词汇量并简化特征空间。对于许多任务，如情感分析或主题建模，这种区分可能不是必需的。

**示例：**

```python
text = "The quick brown fox Jumps over the Lazy Dog."
normalized_text = text.lower()
print(normalized_text)
# 输出: the quick brown fox jumps over the lazy dog.
```

**注意事项：**
虽然通常有益，但不加区分地转换为小写有时会导致信息丢失。

- **专有名词：** 专有名词（如公司“Apple”）和普通名词（水果“apple”）之间的区别会丢失。
- **缩写词：** 缩写词（如“US”，指美国）可能与普通词（如“us”）变得无法区分。

在某些特定应用中，如命名实体识别（NER），保留大小写可能很重要。然而，对于许多一般的自然语言处理任务而言，词汇量减少的好处通常超过了潜在的信息损失。你也可以考虑更复杂的方法，如真值大写（truecasing），它会尝试恢复正确的大小写，但小写转换仍是最常见的起点。

### 处理标点符号和特殊字符

标点符号（如逗号、句号、问号、感叹号）和其他特殊字符（如#、@、%、\$）通常会增加干扰，且在某些上下文 (context)中不提供重要的语义。

**策略：**

- **移除：** 最常见的方法是移除所有标点符号。这能大大简化文本。
- **替换：** 标点符号有时可以用空格替换，以确保被标点符号分隔的词语仍被视为独立的标记 (token)（例如，“end-of-line”变成“end of line”）。
- **选择性处理：** 在某些情况下，某些标点符号可能具有相关性。例如，复合词中的连字符（“state-of-the-art”）或缩写中的撇号（“don't”）可能需要不同对待。社交媒体分析可能会保留井号标签（#）或提及符号（@）。

**示例（使用`string`和`re`进行移除）：**

```python
import string
import re

text = "Hello! This is text_with_punctuation #NLP @example.com $50."

# 使用 string.punctuation
translator = str.maketrans('', '', string.punctuation)
normalized_text_1 = text.translate(translator)
print(f"Method 1: {normalized_text_1}")
# 输出: Method 1: Hello world This is textwithpunctuation NLP examplecom 50

# 使用正则表达式（移除标点符号，保留字母数字和空格）
# 这个模式保留字母、数字和空格，移除其他字符。
normalized_text_2 = re.sub(r'[^\w\s]', '', text)
# 注意：\w 包含下划线，可能需要根据需求调整。
# 一个更精确的模式，只移除标准标点符号：
normalized_text_3 = re.sub(r'[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]', '', text)
print(f"Method 3: {normalized_text_3}")
# 输出: Method 3: Hello world This is textwithpunctuation NLP examplecom 50
```

仔细考虑移除标点符号的影响。移除连字符可能会合并复合修饰语（例如，“state-of-the-art”变成“stateoftheart”），而错误移除撇号则会不正确地处理所有格或缩写（“it's”变成“its”）。

### 移除重音符号和变音符号

重音字符（如 é, ñ, ü）在许多语言中很常见，但也可能出现在英文文本中（例如，“résumé”、“naïve”、“café”）。对这些字符进行标准化，需要将其转换为最接近的ASCII等效字符（例如，“é”转换为“e”，“ñ”转换为“n”）。

**为何使用？**
这确保了“resume”和“résumé”等词被视为相同，进一步标准化词汇。

**示例（使用`unicodedata`）：**

```python
import unicodedata

def remove_accents(input_str):
  nfkd_form = unicodedata.normalize('NFKD', input_str)
  # 只保留非组合字符（有效地移除重音符号）
  return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

text = "Résumé naïve café façade"
normalized_text = remove_accents(text)
print(normalized_text)
# 输出: Resume naive cafe facade
```

像 `unidecode` 这样的库提供了另一种便捷的方法来进行此音译。与D其他标准化步骤一样，如果重音所带有的区分对你的特定任务很重要，请注意，尽管这在主要处理英文文本的应用中不那么常见。

### 扩展缩写词

缩写词是词语或短语的缩短形式（例如，“don't”、“I'm”、“can't”、“it's”）。将它们扩展回原始形式（“do not”、“I am”、“cannot”、“it is”）有助于文本标准化。

**为何使用？**
扩展可确保一致的词元 (token)化和表示。例如，“don't”可能与“do”和“not”的词元化方式不同。扩展它能确保组成词语明确存在。

**示例（使用字典查找）：**

```python
# 一个小样本映射
contraction_map = {
    "don't": "do not",
    "can't": "cannot",
    "i'm": "i am",
    "it's": "it is", # 注意：小心所有格的 'its'
    "you're": "you are",
    "isn't": "is not"
    # ...添加更多缩写
}

text = "I'm sure it's okay, don't worry."
# 基于简单正则表达式的替换（需要仔细设计模式）
import re

# 为键创建正则表达式模式
contraction_pattern = re.compile(r'\b(' + '|'.join(contraction_map.keys()) + r')\b')

def expand_contractions(text, cmap):
    def replace(match):
        return cmap[match.group(0).lower()] # 使用 lower() 进行大小写不敏感匹配
    # 使用编译后的模式应用替换函数
    # 对输入文本使用 lower() 进行大小写不敏感匹配
    return contraction_pattern.sub(replace, text.lower())

normalized_text = expand_contractions(text, contraction_map)
print(normalized_text)
# 输出: i am sure it is okay, do not worry.
```

构建一个全面的缩写映射或使用预构建的库是常见做法。注意模糊情况，例如“it's”（it is）与“its”（所有格）的区分，这可能需要更复杂的上下文 (context)感知处理，如果需要高精度，可能会推迟到后续阶段，如词性标注。

### 处理数字

文本中的数字可以用多种方式处理，这取决于数字信息与任务的相关性：

- **移除：** 如果精确的数值不重要，可以完全移除数字。
- **用占位符替换：** 用 `<NUM>` 或 `#` 等通用标记 (token)替换数字。这保留了数字存在的信息，但不保留具体数值，如果数字的存在重要而其大小不重要，这种方法很有用。
- **转换为单词：** 将数字转换为其对应的单词形式（例如，“5”转换为“five”）。这会将数字整合到标准词汇中，但可能显著增加词汇量。

**注意事项：**
最好的方法取决于具体任务。对于金融情感分析，具体的数字可能非常重要。对于一般主题建模，用占位符替换可能就足够了。

**示例（使用正则表达式替换）：**

```python
import re

text = "Order 123 costs $49.99 for 2 items."

# 移除数字
removed = re.sub(r'\d+', '', text)
print(f"Removed: {removed}")
# 输出: Removed: Order costs $. for items.

# 用占位符 <NUM> 替换
placeholder = re.sub(r'\d+', '<NUM>', text)
print(f"Placeholder: {placeholder}")
# 输出: Placeholder: Order <NUM> costs $<NUM>.<NUM> for <NUM> items.

# 更具体的货币/数字替换
placeholder_refined = re.sub(r'\b\d+\b', '<NUMBER>', text) # 仅独立的数字
placeholder_refined = re.sub(r'\$\d+(\.\d+)?', '<PRICE>', placeholder_refined) # 价格
print(f"Refined Placeholder: {placeholder_refined}")
# 输出: Refined Placeholder: Order <NUMBER> costs <PRICE> for <NUMBER> items.

# 转换为单词通常需要外部库，如 'inflect'
# import inflect
# p = inflect.engine()
# converted = re.sub(r'\b\d+\b', lambda m: p.number_to_words(m.group(0)), text)
# print(f"Converted: {converted}")
# 输出: Converted: Order one hundred and twenty-three costs $49.99 for two items.
# (注意：$49.99 需要更复杂的处理才能完全转换)
```

### 组合标准化步骤

这些标准化方法很少单独使用。它们通常作为更大预处理流程中的顺序步骤应用。顺序可能很重要。例如，在小写转换前扩展缩写词可能比处理大小写混用的缩写词更容易。在扩展缩写词之前移除标点符号可能会破坏用于匹配的模式（例如，“don't”变成“dont”）。

典型的顺序可能是：

1. 将文本小写。
2. 扩展缩写词。
3. 移除重音符号。
4. 移除标点符号（或选择性处理）。
5. 处理数字（移除、替换或转换）。

然而，最佳顺序和所选的具体方法很大程度上取决于你的文本数据特征和NLP应用的目标。通常需要进行实验和评估来确定最佳标准化策略。

通过应用这些文本标准化方法，你可以将原始的、多变的文本转换为标准化格式。这种清理后的数据更适合我们在下一章中将要讨论的特征工程方法，从而产生更有效、更可靠的自然语言处理模型。

## 参考资料

- [Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition (3rd Edition Draft)](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  这本教材草稿全面涵盖文本规范化，包括分词、小写转换和其他对自然语言处理的预处理步骤。
- [Natural Language Processing with Python - Analyzing Text with the Natural Language Toolkit](https://www.nltk.org/book/) — Steven Bird, Ewan Klein, and Edward Loper (2009)
  Publisher: O'Reilly Media
  一本实用指南，展示了如何使用Python和NLTK库进行文本预处理技术，适用于应用规范化方法。
- [CS224N: Natural Language Processing with Deep Learning - Lecture Notes / Course Materials](https://cs224n.stanford.edu/) — Christopher Manning and Stanford University CS224N Staff (2023)
  Publisher: Stanford University
  大学课程材料，提供了文本预处理的学术背景和讨论，这是自然语言处理流程中的初始步骤。
