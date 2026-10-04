---
course: "nlp-fundamentals"
chapter: "nlp-word-embeddings"
lesson: "using-pretrained-embeddings"
sourceId: 2501
sourceUrl: "https://apxml.com/zh/courses/nlp-fundamentals/chapter-4-nlp-word-embeddings/using-pretrained-embeddings"
title: "使用预训练词嵌入模型"
description: "了解如何加载和采用预训练的Word2Vec和GloVe模型用于下游任务。"
order: 7
plots: []
sourceHash: "972003f6418d90c2d9a869171d3f055d934145826e5967ac1c56298798f6182a"
sourceCorrections: []
---

从头训练Word2Vec或GloVe之类的词嵌入 (embedding)模型需要大量文本语料和可观的计算资源。虽然为高度专业化的应用场景训练自己的嵌入模型可能有益，但通常采用预训练 (pre-training)嵌入模型会更实用且高效。这些模型已经在海量数据集（如维基百科、Google新闻或Common Crawl）上训练过，能有效捕捉词语间普遍的语义关系。使用它们为许多NLP任务提供了有力的起点，特别是当您自己的数据集相对较小时。

### 使用预训练 (pre-training)嵌入 (embedding)模型的益处

采用预训练词嵌入模型有几项优势：

1. **资源高效性**：您节省了大量时间和计算能力，否则这些时间和能力将用于收集大量文本和训练嵌入模型。
2. **借助大规模知识**：这些模型包含了源自数十亿甚至数万亿词汇的词语知识，能捕捉细致的语义差别，而这些差别在训练小规模、特定场景数据集时可能会被忽略。
3. **提高下游任务性能**：对于许多常用NLP任务，如情感分析、文本分类或命名实体识别，使用预训练嵌入初始化模型通常会带来更好表现，相比于随机嵌入或纯粹基于频率的特征。这是因为模型在开始时就带有一种表示，这种表示已经理解“国王”和“王后”或“跑步”和“慢跑”之类的词语是相关的。
4. **对数据量有限情况有效**：当您没有足够数据自行训练有意义的嵌入时，预训练向量 (vector)能提供从外部知识中获得的表示。

### 常用预训练 (pre-training)嵌入 (embedding)模型

有几个广泛使用的预训练嵌入模型是公开可用的：

- **Word2Vec (Google News)**: 由Google在庞大的Google News数据集（约1000亿词）上训练。通常为300万词和短语提供300维向量 (vector)。
- **GloVe (Stanford)**: 由斯坦福大学研究人员开发，在包括维基百科和Common Crawl在内的多种语料库上训练。提供不同维度（例如，50、100、200、300）和词汇量大小的版本。
- **fastText (Facebook AI Research)**: 与Word2Vec类似，但也考虑子词 (subword)信息（字符n-gram）。这使得fastText能够为词汇表 (vocabulary)外（OOV）词生成向量，并且常用于形态丰富的语言。支持多种语言。

这些模型通常以文本文件形式分发。每行通常包含一个词，后面是其对应的向量分量（浮点数），以空格分隔。

### 加载和采用预训练 (pre-training)嵌入 (embedding)

通常的流程包括下载预训练的向量 (vector)文件，并将其解析成可用格式，通常是字典或映射，其中键是词语，值是其对应的嵌入向量（常表示为NumPy数组）。

让我们举例说明如何加载GloVe向量。假设您已下载了一个名为`glove.6B.100d.txt`的文件（在60亿词元 (token)上训练的100维向量）。

```python
import numpy as np

def load_glove_embeddings(file_path):
    """
    从文本文件加载GloVe词嵌入。

    Args:
        file_path (str): GloVe嵌入文件的路径。

    Returns:
        dict: 将词映射到其嵌入向量（NumPy数组）的字典。
    """
    embeddings_index = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                try:
                    # 将系数从字符串转换为浮点数
                    vector = np.asarray(values[1:], dtype='float32')
                    embeddings_index[word] = vector
                except ValueError:
                    # 处理文件中可能存在的解析错误或不规范情况
                    print(f"跳过词语: {word}的行。无法解析向量。")
                    continue
        print(f"已加载 {len(embeddings_index)} 个词向量。")
    except FileNotFoundError:
        print(f"错误：嵌入文件在 {file_path} 处未找到")
    except Exception as e:
        print(f"发生错误：{e}")

    return embeddings_index

# 使用示例：
# glove_file = 'path/to/your/glove.6B.100d.txt'
# word_vectors = load_glove_embeddings(glove_file)

# # 访问向量
# if 'computer' in word_vectors:
#    computer_vector = word_vectors['computer']
#    print(f"'computer'向量的维度：{computer_vector.shape}")
# else:
#    print("在嵌入中未找到'computer'。")
```

像`gensim`这样的库提供了便捷函数来加载各种预训练嵌入格式（Word2Vec二进制/文本、GloVe、fastText）。

```python
# 使用gensim的示例（需要安装：pip install gensim）
# import gensim.downloader as api

# 加载预训练的GloVe嵌入（例如，GloVe Twitter 25d）
# 如果模型尚未存在，这将下载它
# try:
#    glove_model = api.load("glove-twitter-25")
#    vector = glove_model['computer']
#    print(f"'computer'向量的维度（gensim）：{vector.shape}")
#    # 您可以访问底层的KeyedVectors对象以获得更多功能
#    # word_vectors = glove_model.key_to_index # 词 -> 索引的字典
#    # embeddings_matrix = glove_model.vectors # 所有向量的NumPy矩阵
# except ValueError as e:
#     print(f"使用gensim加载模型时出错：{e}")
# except Exception as e:
#     print(f"使用gensim时发生意外错误：{e}")
```

### 将嵌入 (embedding)整合到神经网络 (neural network)中

加载后，这些预训练 (pre-training)的嵌入通常用于初始化神经网络模型（如用于文本的LSTM或CNN）中的嵌入层。

1. **词汇映射**：创建一个映射（字典），将您特定数据集词汇中的词语映射到独有的整数索引。
2. **嵌入矩阵**：构建一个嵌入矩阵（一个NumPy数组），其中索引`i`处的行对应于您词汇映射中索引为`i`的词语的预训练向量 (vector)。此矩阵的大小将为`(vocabulary_size, embedding_dimension)`。
3. **处理词汇表 (vocabulary)外（OOV）词**：您的数据集可能包含预训练模型词汇表中不存在的词语。常见策略包括：
   - 给它们分配一个零向量。
   - 给它们分配一个随机初始化的向量（这可能在训练期间进行微调 (fine-tuning)）。
   - 使用一个专用的`<UNK>`（未知）标记 (token)，并为其分配一个向量（可以是零向量、随机向量，或从已知向量中取平均值）。
   - 如果采用fastText，它能根据字符n-gram为OOV词生成向量。
4. **嵌入层初始化**：使用构建的嵌入矩阵来初始化您的神经网络框架（例如，TensorFlow/Keras、PyTorch）中嵌入层的权重 (weight)。您通常最初将此层设置为不可训练以保留预训练知识，尽管微调也是一个选择。

```python
# 示例（使用Keras类似结构）

# 假设：
# tokenizer: 将词映射到整数（例如，tokenizer.word_index）
# word_vectors: 从预训练文件加载的字典（例如，GloVe）
# EMBEDDING_DIM: 向量维度（例如，100）
# VOCAB_SIZE: 数据集中词汇的独有词语数量

# 1. 创建嵌入矩阵
embedding_matrix = np.zeros((VOCAB_SIZE, EMBEDDING_DIM))
for word, i in tokenizer.word_index.items():
    if i >= VOCAB_SIZE:
        continue # 安全检查
    embedding_vector = word_vectors.get(word) # 从加载的GloVe字典获取向量
    if embedding_vector is not None:
        # 在嵌入索引中找到的词将被复制
        embedding_matrix[i] = embedding_vector
    # 否则：在嵌入索引中未找到的词将全部为零（默认的OOV处理）

# 2. 在模型中定义嵌入层
# from tensorflow.keras.layers import Embedding, Input, LSTM, Dense
# from tensorflow.keras.models import Model

# input_layer = Input(shape=(MAX_SEQUENCE_LENGTH,))
# embedding_layer = Embedding(
#     input_dim=VOCAB_SIZE,
#     output_dim=EMBEDDING_DIM,
#     weights=[embedding_matrix], # 使用预训练权重初始化
#     input_length=MAX_SEQUENCE_LENGTH,
#     trainable=False # 设置为False以固定嵌入，设置为True以进行微调
# )(input_layer)

# # ... 添加后续层（例如，LSTM，Dense） ...
# lstm_layer = LSTM(units=64)(embedding_layer)
# output_layer = Dense(1, activation='sigmoid')(lstm_layer) # 二进制分类示例
# model = Model(inputs=input_layer, outputs=output_layer)
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# print(model.summary()) # 显示模型结构
```

### 微调 (fine-tuning)嵌入 (embedding)

虽然通常保持固定，但您可以选择将预训练 (pre-training)的嵌入层设置为*可训练*（`trainable=True`）。这允许嵌入向量 (vector)在您的特定下游任务的训练过程中进行调整。

- **优点**：可以使通用嵌入更精准地适应您的特定数据集和任务的特点，可能会提高性能。
- **缺点**：需要更多训练数据以避免嵌入在您的任务上过拟合 (overfitting)，可能会失去一些宝贵的通用语义知识。增加要训练的参数 (parameter)数量。

微调的决定通常取决于您数据集的大小以及预训练语料库与您任务的应用场景之间的相似性。常见策略是从不可训练的嵌入开始，之后再尝试微调，也许为嵌入层设置较低的学习率。

通过采用预训练词嵌入，您可以更快地构建更有效的NLP模型，受益于从大规模文本语料库中捕获的广泛语言知识，而无需自行训练。这是现代自然语言处理中一种标准且高效的技术。

## 参考资料

- [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781) — Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean (2013)
  Journal: arXiv preprint arXiv:1301.3781; DOI: [10.48550/arXiv.1301.3781](https://doi.org/10.48550/arXiv.1301.3781)
  介绍了原始的Word2Vec模型及其两种架构（CBOW和Skip-gram），用于学习高效的词表示。
- [GloVe: Global Vectors for Word Representation](https://www.aclweb.org/anthology/D14-1162/) — Jeffrey Pennington, Richard Socher, Christopher Manning (2014)
  Journal: Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP); Publisher: Association for Computational Linguistics; Pages: 1532-1543; DOI: [10.3115/v1/D14-1162](https://doi.org/10.3115/v1/D14-1162)
  提出了GloVe模型，结合了全局矩阵分解和局部上下文窗口方法来学习词表示。
- [Enriching Word Vectors with Subword Information](https://aclanthology.org/Q17-1010/) — Piotr Bojanowski, Edouard Grave, Armand Joulin, Tomas Mikolov (2017)
  Journal: Transactions of the Association for Computational Linguistics (TACL); Publisher: MIT Press; Volume: 5; Pages: 135-146; DOI: [10.1162/tacl_a_00051](https://doi.org/10.1162/tacl_a_00051)
  描述了fastText，Word2Vec的扩展，它结合了子词（字符n-gram）信息，从而更好地处理罕见词和未登录词。
- [Speech and Language Processing (3rd ed. draft)](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky, James H. Martin (2025)
  Publisher: Stanford University
  一本全面的自然语言处理教材，包含词嵌入、其历史和应用的详细章节。第6章“向量语义与嵌入”尤为相关。
