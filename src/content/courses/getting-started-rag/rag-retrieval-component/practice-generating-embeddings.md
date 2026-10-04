---
course: "getting-started-rag"
chapter: "rag-retrieval-component"
lesson: "practice-generating-embeddings"
sourceId: 3644
sourceUrl: "https://apxml.com/zh/courses/getting-started-rag/chapter-2-rag-retrieval-component/practice-generating-embeddings"
title: "实践：生成文本嵌入"
description: "动手实践使用库为示例文本数据生成向量嵌入。"
order: 7
plots: ["plots/3644-0.json"]
sourceHash: "4d33052f87531966ec0a8dd43d867db7bbd806f7af7bbd41b9adbbbc64d74c34"
sourceCorrections: []
---

获得动手实践的经验，通过一个常用的 Python 库生成文本的数值表示。这种实践应用侧重于创建文本嵌入 (embedding)，这是为任何 RAG 系统的检索部分准备数据的主要步骤。

### 选择嵌入 (embedding)库和模型

有许多库和预训练 (pre-training)模型可用于生成文本嵌入。在本次练习中，我们将使用 `sentence-transformers` 库，它为许多先进的嵌入模型提供了易于使用的接口。它构建于广泛采用的 Hugging Face `transformers` 库之上。

我们将使用 `all-MiniLM-L6-v2` 模型。这是一个备受推崇的句子嵌入模型，以其在计算效率和语义相似性任务表现之间的平衡而闻名。对于许多 RAG 用途来说，它是一个很好的起点。

### 设置你的环境

首先，请确保你已安装 `sentence-transformers` 库。如果你尚未安装，可以使用 pip 来安装：

```bash
pip install sentence-transformers
```

这个命令会下载并安装该库及其依赖项，包括 PyTorch 或 TensorFlow（取决于你的配置）以及 `transformers` 库。

### 为单个句子生成嵌入 (embedding)

让我们从基础开始：为单个文本片段生成嵌入。

```python
from sentence_transformers import SentenceTransformer

# 加载预训练模型
# 如果模型未缓存，将自动下载
model = SentenceTransformer('all-MiniLM-L6-v2')

# 定义一个示例文本
sentence = "This is an example sentence demonstrating embedding generation."

# 生成嵌入
embedding = model.encode(sentence)

# 打印嵌入的形状和前几个维度
print(f"句子: {sentence}")
print(f"嵌入形状: {embedding.shape}")
print(f"嵌入（前5个维度）: {embedding[:5]}")
```

当你运行这段代码时，`SentenceTransformer` 库会首先下载 `all-MiniLM-L6-v2` 模型文件，如果你的系统上还没有这些文件。`model.encode()` 方法接收输入文本，并通过模型的层进行处理，生成一个数值向量 (vector)，即嵌入。

输出会像这样（具体的浮点数值可能会略有不同）：

```
句子: This is an example sentence demonstrating embedding generation.
嵌入形状: (384,)
嵌入（前5个维度）: [ 0.05483127  0.05900988 -0.00499174  0.07899283 -0.0135861 ]
```

请注意形状 `(384,)`。这表明 `all-MiniLM-L6-v2` 模型为输入句子生成了一个384维的向量。这384个数字中的每一个都捕捉了句子语义含义的某些方面，这是模型在训练过程中学习到的。

### 为多个句子生成嵌入 (embedding)（批量处理）

如果你有许多文档或文本块，一个一个地生成嵌入效率很低。`encode` 方法经过优化，可以批处理方式处理句子列表（或文本片段），运用并行计算能力。

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

# 定义一个句子列表
sentences = [
    "The retriever finds relevant documents.",
    "Vector databases store text embeddings efficiently.",
    "Large language models generate human-like text.",
    "RAG combines retrieval with generation."
]

# 为句子列表生成嵌入
embeddings = model.encode(sentences)

# 打印生成的嵌入数组的形状
print(f"句子数量: {len(sentences)}")
print(f"嵌入数组形状: {embeddings.shape}")

# 你可以像这样访问单个嵌入：
# print(f"第一个句子的嵌入（前5个维度）：{embeddings[0][:5]}")
```

输出会显示生成的 NumPy 数组的形状：

```
句子数量: 4
嵌入数组形状: (4, 384)
```

这个输出 `(4, 384)` 确认我们已为所有4个输入句子生成了嵌入，并且每个嵌入都是一个384维的向量 (vector)。与在循环中单独编码句子相比，这种批量处理方法对于大型数据集来说要快得多。

### 理解输出

你现在对如何将文本转换为密集向量 (vector) (dense vector)表示有了实际理解。每个向量，就像上面生成的那些，都位于一个高维空间 (high-dimensional space)中（在我们的例子中是384维）。模型在训练过程中学习到的主要特性是，含义相似的文本，其向量在这个空间中会彼此“接近”，通常使用余弦相似度（$cos(\theta)$）等度量来衡量。

很难直接可视化384个维度，但可以想象一个更简单的二维空间。含义相似的句子会聚在一起。



![二维嵌入空间](plots/3644-0.json)



> 图示含义相似的句子（“猫坐在垫子上”，“猫科动物在地毯上”）在简化的二维嵌入 (embedding)空间中，与不相关的句子（“天气很好”，“股票市场”）相比，可能会更紧密地聚在一起。

这些生成的嵌入是你将存储在向量数据库中的基本元素。在即将到来的部分和章节中，你将学习如何使用这些嵌入填充向量数据库，并执行相似性搜索，以找到给定查询最相关的文本块，这构成了 RAG 系统检索机制的核心。

## 参考资料

- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://aclanthology.org/D19-1410/) — Nils Reimers and Iryna Gurevych (2019)
  Journal: Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP); Publisher: Association for Computational Linguistics; Pages: 3982-3992; DOI: [10.18653/v1/D19-1410](https://doi.org/10.18653/v1/D19-1410)
  介绍了一种微调预训练语言模型以生成语义丰富的句子嵌入的方法，该方法是 `sentence-transformers` 库的基础。
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index) — Hugging Face (2024)
  Hugging Face `transformers` 库的资源，`sentence-transformers` 构建于其上，并提供 MiniLM 等预训练模型的详细信息。
