# 字节对编码 (BPE) 算法

来源：[原文](https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-5-tokenization-large-vocabularies/byte-pair-encoding-bpe)

[返回章节目录](README.md) · [返回课程目录](../README.md)

标准的词级分词 (tokenization)方法在处理用于训练大型语言模型的庞大且多样的文本语料库时遇到不少难题。主要问题在于处理词汇表 (vocabulary)外 (OOV) 词汇以及词汇表可能非常庞大。字节对编码 (BPE) 为应对这些难题提供了一种有效的数据驱动型方法。BPE 最初是一种数据压缩算法，后来成功调整用于文本分词，构建了子词 (subword)单元词汇表。

BPE 的核心思想十分简洁：它以训练语料库中所有单个字符构成的词汇表为起始，迭代合并出现频率最高的相邻符号对，形成新的、更长的子词符号。这个过程会持续预设的合并次数，有效地控制最终词汇表的大小。

### BPE 训练过程

让我们仔细看看 BPE 算法如何从语料库中学习其词汇表 (vocabulary)和合并规则。

1. **初始化**：

   - 设定目标最终词汇表大小 $V_{target}$。
   - 将训练语料库中的每个词表示为字符序列，并添加一个特殊的词尾符号（通常是 `</w>` 或 `</w>`) 以区分词 (tokenization)边界。例如，“lower” 变成 `l`、`o`、`w`、`e`、`r`、`</w>`。
   - 初始词汇表包含语料库中所有独有的字符。
2. **迭代**：重复以下步骤，直至达到目标词汇表大小 $V_{target}$ 或已执行预设数量的合并操作：

   - **统计词对**：在语料库当前表示中，找出所有相邻符号对（字符或已合并的子词 (subword)），并统计它们的频率。
   - **找出最频繁词对**：选择出现频率最高的词对 $(s_1, s_2)$。
   - **合并**：创建一个新符号 $s_{12}$ 来表示合并后的词对。将 $s_{12}$ 添加到词汇表。
   - **更新语料库表示**：将语料库表示中所有相邻词对 $(s_1, s_2)$ 的出现替换为新符号 $s_{12}$。记录此合并操作（例如：“将 $s_1$ 和 $s_2$ 合并为 $s_{12}$”）。

### 一个小示例

假设有一个小语料库，词频如下：`{'low': 5, 'lower': 2, 'newest': 6, 'widest': 3}`。

**步骤 0: 初始化**

- 将词表示为字符序列 + `</w>`：
  - `low`：`l o w </w>` (出现 5 次)
  - `lower`：`l o w e r </w>` (出现 2 次)
  - `newest`：`n e w e s t </w>` (出现 6 次)
  - `widest`：`w i d e s t </w>` (出现 3 次)
- 初始词汇表 (vocabulary)：`{l, o, w, </w>, e, r, n, s, t, i, d}`
- 目标词汇表大小：为演示目的，我们进行几次合并。

**步骤 1: 统计词对并合并**

- 统计整个语料库中的相邻词对（考虑频率）：
  - `(l, o)`：5 + 2 = 7
  - `(o, w)`：5 + 2 = 7
  - `(w, </w>)`：5
  - `(w, e)`：2 + 6 = 8
  - `(e, r)`：2
  - `(r, </w>)`：2
  - `(n, e)`：6
  - `(e, w)`：6
  - `(e, s)`：6 + 3 = 9 <- 最频繁
  - `(s, t)`：6 + 3 = 9 <- 频率相同（我们选择 `(e, s)`）
  - `(t, </w>)`：6 + 3 = 9 <- 频率相同
  - `(w, i)`：3
  - `(i, d)`：3
  - `(d, e)`：3
- 将 `(e, s)` 合并成新符号 `es`。词汇表大小增加。
- 更新语料库：
  - `l o w </w>` (5)
  - `l o w e r </w>` (2)
  - `n e w es t </w>` (6)
  - `w i d es t </w>` (3)
- 记录合并：`e + s -> es`

**步骤 2: 统计词对并合并**

- 在更新后的语料库中重新计算词对频率：
  - `(l, o)`：7
  - `(o, w)`：7
  - `(w, </w>)`：5
  - `(w, e)`：2 + 6 = 8
  - `(e, r)`：2
  - `(r, </w>)`：2
  - `(n, e)`：6
  - `(e, w)`：6
  - `(w, es)`：6
  - `(es, t)`：6 + 3 = 9 <- 最频繁（或与 `(t, </w>)` 频率相同）
  - `(t, </w>)`：6 + 3 = 9
  - `(w, i)`：3
  - `(i, d)`：3
  - `(d, es)`：3
- 将 `(es, t)` 合并成新符号 `est`。
- 更新语料库：
  - `l o w </w>` (5)
  - `l o w e r </w>` (2)
  - `n e w est </w>` (6)
  - `w i d est </w>` (3)
- 记录合并：`es + t -> est`

**步骤 3: 统计词对并合并**

- 重新计算频率：
  - ...
  - `(est, </w>)`：6 + 3 = 9 <- 最频繁（如果 `est` 在第 2 步中未被选中，则与 `(t, </w>)` 频率相同）
  - ...
- 将 `(est, </w>)` 合并成 `est</w>`。
- 更新语料库：
  - `l o w </w>` (5)
  - `l o w e r </w>` (2)
  - `n e w est</w>` (6)
  - `w i d est</w>` (3)
- 记录合并：`est + </w> -> est</w>`

此过程持续进行。如果我们执行更多合并，像 `(l, o)`、`(o, w)`、`(w, e)` 这样的词对也可能被合并，潜在地形成 `low` 或 `we`。最终词汇表将包含单个字符和常见的多个字符子词 (subword)，例如 `es`、`est`、`est</w>` 等，这些都是根据它们在训练数据中的频率形成的。

### 使用已学习的 BPE 分词 (tokenization)新文本

一旦从训练语料库中学习到 BPE 词汇表 (vocabulary)和有序的合并操作列表，对新文本进行分词涉及以下步骤：

1. 将输入词拆分为字符序列。添加词尾符号 `</w>`。
2. 迭代地应用在训练期间学习到的合并操作，*顺序相同*。对于每个学习到的合并 $(s_1, s_2) \rightarrow s_{12}$，在当前序列中找出所有相邻的 $(s_1, s_2)$ 出现，并用 $s_{12}$ 替换。
3. 继续此过程，直到没有更多学习到的合并可以应用于序列。
4. 生成的符号序列（字符和子词 (subword)）即为分词表示。每个符号对应最终词汇表中的一个 ID。

例如，如果我们学习到合并规则 `e + s -> es`，然后 `es + t -> est`，对“tests”进行分词的步骤如下：

1. 初始：`t`、`e`、`s`、`t`、`s`、`</w>`
2. 应用 `e + s -> es`：`t`、`es`、`t`、`s`、`</w>`
3. 应用 `es + t -> est`：不存在相邻的 `es`、`t` 词对。
4. 最终分词结果：`t`、`es`、`t`、`s`、`</w>`

如果在 `est` *之后* 也学习到 `t + s -> ts`，那么它现在可能适用：`t`、`es`、`ts`、`</w>`。合并的顺序很重要。

### 处理未知词

BPE 的一个显著优点是其固有的能力，可以处理训练期间未曾出现的词汇（OOV 词汇）。由于初始词汇表 (vocabulary)包含所有单个字符，任何词汇在必要时都可以分解为字符序列。如果词汇的部分对应于学习到的子词 (subword)，则会使用这些子词；否则，它会退回到单个字符。例如，如果“huggingface”不在训练数据中，但“hugg”、“ing”、“face”是已学习的子词（或可以通过合并形成），它可能会被分词 (tokenization)为 `hugg`、`ing`、`face`、`</w>`。如果不是，它可能会变成 `h`、`u`、`g`、`g`、`i`、`n`、`g`、`f`、`a`、`c`、`e`、`</w>`。从需要专用 `[UNK]` 标记 (token)的意义上讲，BPE 不存在真正的“未知”标记，尽管为了其他目的可能仍会包含一个。

### 实现说明

- **字节级 BPE**：一种常见变体在字节而非字符上操作。这特别有用，因为它保证了固定的初始词汇表 (vocabulary)大小（256 字节），并且自然地处理所有 Unicode 字符，无需特殊处理重音或未见字符。其逻辑保持不变，只是合并频繁的字节对。
- **库**：高效地实现 BPE 需要仔细处理数据结构和计数。幸运的是，Hugging Face 的 `tokenizers` 等库提供了优化的实现。训练 BPE 分词 (tokenization)器 (tokenizer)可能如下所示（伪代码）：

```python
# BPE 训练循环
corpus = ["low low low low low", "lower lower", ...] # 你的文本数据
word_counts = get_word_counts(corpus)
vocab = initialize_with_characters(word_counts)
splits = initial_split_words(word_counts) # e.g., {'l o w </w>': 5, ...}
num_merges = target_vocab_size - len(vocab)
merges = {} # 存储学习到的合并规则

for i in range(num_merges):
  pair_counts = count_adjacent_pairs(splits)
  if not pair_counts:
    break
  most_frequent_pair = find_most_frequent(pair_counts)
  new_token = merge_pair(most_frequent_pair)
  vocab.add(new_token)
  merges[most_frequent_pair] = new_token
  splits = apply_merge_to_splits(splits, most_frequent_pair, new_token)

# 保存词汇表和合并规则
```

实际使用中，通常会利用成熟的库。下面是如何在 PyTorch 环境中使用 Hugging Face `transformers` 库的预训练 (pre-training) BPE 分词器（如 GPT-2 的）的示例：

```python
import torch
from transformers import GPT2Tokenizer

# 加载预训练的 BPE 分词器
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

text = "lower newest widest"
encoded_input = tokenizer(text, return_tensors='pt') # pt 表示 PyTorch 张量

print("输入文本:", text)
print("分词ID:", encoded_input['input_ids'])
# 示例输出 (ID可能不同): tensor([[ 3224, 29976, 23564]])
# 注意在 GPT-2 的词汇表中，'lower'、'newest'、'widest' 各为一个分词

print(
    "解码分词:",
    tokenizer.convert_ids_to_tokens(encoded_input['input_ids'][0])
)
# 示例输出: ['lower', 'Ġnewest', 'Ġwidest']
# 'Ġ' (U+0120) 通常表示
# 一个词/分词前面有空格。

# 处理潜在的未知组合（尽管 BPE 处理 OOV 字符/字节）
text_oov = "supercalifragilisticexpialidocious"
encoded_oov = tokenizer(text_oov, return_tensors='pt')
print("\n类OOV文本:", text_oov)
print("分词ID:", encoded_oov['input_ids'])
print(
    "解码分词:",
    tokenizer.convert_ids_to_tokens(encoded_oov['input_ids'][0])
)
# 示例输出: ['super', 'cal', 'if', 'rag', 'il',
# 'istic', 'exp', 'ial', 'id', 'ocious']
# 该词被分解为 GPT-2 词汇表中已知的子词单元。
```

这段代码演示了训练后的 BPE 分词器如何将文本分段为由数字 ID 表示的子词 (subword)单元，以便输入到如 Transformer 这样的模型。它还展示了 OOV 处理，即一个未曾见过的词被分解为可识别的片段。

### 词汇表 (vocabulary)大小权衡

BPE 训练期间执行的合并操作次数直接决定了最终词汇表的大小。这带来一个权衡：

- **更大的词汇表**：可以用单个分词 (tokenization)表示常用词和语素，从而使常见文本的分词序列更短。但是，这会增加模型的嵌入 (embedding)层大小（$|V| \times d_{model}$），并可能包含非常稀有的子词 (subword)。
- **更小的词汇表**：由于词汇更常被分解为更小的单元，导致分词序列更长。这会增加序列处理期间的计算成本，但会减小嵌入层大小，并通过将罕见或形态复杂的词汇从更小片段组合起来，可能提供更好的泛化能力。

选择合适的词汇表大小是一个经验性的过程，通常受模型规模、训练数据特点以及下游任务表现的指导。BPE 提供了控制这种平衡的机制。

总之，BPE 是一种功能强大且广泛使用的子词分词技术。通过学习合并大型语料库中频繁出现的字符或字节对，它构建了一个词汇表，能有效处理大量文本，避免 OOV 问题，并允许控制词汇表大小和序列长度之间的平衡。

## 参考资料

- [A New Algorithm for Data Compression](https://drdobbs.com/a-new-algorithm-for-data-compression/184407767) — Philip Gage (1994)
  Journal: C/C++ Users Journal; Volume: 12; Pages: 23-28
  首次提出字节对编码作为通用数据压缩算法的论文。
- [Neural Machine Translation of Rare Words with Subword Units](https://aclanthology.org/P16-1162/) — Rico Sennrich, Barry Haddow, and Alexandra Birch (2016)
  Journal: Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers); Publisher: Association for Computational Linguistics; Pages: 1715-1725; DOI: [10.18653/v1/P16-1162](https://doi.org/10.18653/v1/P16-1162)
  将BPE应用于神经机器翻译的开创性工作，使其在自然语言处理领域得到广泛应用。
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever (2019)
  Publisher: OpenAI
  在GPT-2模型中引入字节级BPE（BBPE）进行分词，解决了任意Unicode文本处理的挑战。
- [Natural Language Processing with Transformers: Building Innovative Applications with 🤗 Transformers](https://www.oreilly.com/library/view/natural-language-processing/9781098103231/) — Lewis Tunstall, Leandro von Werra, and Thomas Wolf (2022)
  Publisher: O'Reilly Media
  在现代大语言模型和Hugging Face生态系统的背景下，提供了包括BPE在内的分词技术的实用和最新概述。

---

[上一节](01-%E5%AD%90%E8%AF%8D%E5%88%86%E8%AF%8D%E7%9A%84%E5%BF%85%E8%A6%81%E6%80%A7.md) · [下一节](03-WordPiece%20%E5%88%86%E8%AF%8D.md)
