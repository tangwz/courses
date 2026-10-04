# 动手实践：使用LDA进行主题建模

来源：[原文](https://apxml.com/zh/courses/bayesian-machine-learning/chapter-5-advanced-probabilistic-graphical-models/practice-topic-modeling-lda)

[返回章节目录](README.md) · [返回课程目录](../README.md)

潜在狄利克雷分配（LDA）是一种概率模型，将文档视为主题的混合，主题定义为词语上的分布，通常采用贝叶斯框架进行建模。塌缩吉布斯采样和变分贝叶斯（VB）等推断方法常用于从文档集合中估计隐藏的主题结构。接下来，将逐步实现LDA，解释其输出，并评估其在真实文本数据集上的表现。

我们主要使用Python中的`gensim`库，这是一个用于主题建模的常用工具包。虽然我们论述了吉布斯采样和变分贝叶斯，但`gensim`的标准`LdaModel`实现依赖于一种高效的在线变分贝叶斯算法，它非常适合大型数据集。我们将侧重于这种实际实现和评估，同时记住之前论述的VB的理论性质。

### 环境设置

首先，确保您已安装所需的库。您需要`gensim`用于LDA，`nltk`用于文本预处理（如停用词移除），如果想使用像20 Newsgroups这样的常用数据集，可能还需要`scikit-learn`。对于可视化，`pyLDAvis`非常好用。

```bash
pip install gensim nltk scikit-learn pyldavis matplotlib seaborn
```

您可能还需要下载`nltk`数据：

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
# 如果尚未安装，您可能还需要'punkt'进行分词
# nltk.download('punkt')
```

### 数据准备：良好主题的根本

“输入垃圾，输出垃圾”这句话对主题建模尤其适用。仔细预处理文本数据对于发现有意义的主题很重要。我们来概述一个使用文档列表的典型流程。在实际应用中，您可能会从文件中加载数据，或者使用`scikit-learn`中的数据集加载器，例如`fetch_20newsgroups`。

```python
# 示例文档（替换为您的实际数据）
documents = [
    "Bayesian inference provides a framework for updating beliefs.",
    "Markov Chain Monte Carlo methods are used for sampling posterior distributions.",
    "Variational inference approximates complex distributions.",
    "Topic models like LDA discover latent themes in text data.",
    "Gaussian processes model distributions over functions.",
    "Preprocessing text data is important for topic modeling accuracy.",
    "We use sampling methods or variational approximations for inference."
]

# 1. 分词和小写转换
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    # 移除标点和数字
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    # 移除停用词并词形还原
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
    return tokens

processed_docs = [preprocess(doc) for doc in documents]
print("示例文档:", processed_docs[0])
# Output: Sample Processed Document: ['bayesian', 'inference', 'provides', 'framework', 'updating', 'belief']
```

**预处理步骤：**

1. **分词 (tokenization)：** 将句子拆分为单独的词语（词元 (token)）。
2. **小写转换：** 将所有文本转换为小写，确保“Bayesian”和“bayesian”等词被视为相同的词元。
3. **移除标点/数字：** 这些通常不包含重要的主题信息。
4. **移除停用词：** 常用词（如“the”、“a”、“is”）通常会被移除，因为它们频繁出现在所有主题中。
5. **词形还原（或词干提取）：** 将词语简化为其基本或词根形式（例如，“updating”->“update”，“distributions”->“distribution”）。词形还原通常优于词干提取，因为它产生实际的词典词，从而使主题更易于理解。

### 创建语料库和词典

LDA要求数据采用特定格式：一个将唯一词语映射到ID的词典，以及一个将每个文档表示为词袋（BoW）的语料库。BoW格式是每个文档的一系列元组`(word_id, word_count)`。

```python
from gensim import corpora

# 创建词典
dictionary = corpora.Dictionary(processed_docs)
# 可选：过滤极端词（在少于min_docs或多于max_fraction_docs的文档中出现的词）
# dictionary.filter_extremes(no_below=2, no_above=0.8)

# 创建语料库（词袋）
corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

print("\n示例文档条目:", list(dictionary.items())[0])
# Output: Sample Dictionary Entry: (0, 'bayesian')
print("示例文档条目（BoW格式）:", corpus[0])
# Output: Sample Corpus Entry (BoW format): [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)]
# 这对应于第一个处理过的文档的词ID和计数。
```

### 使用变分贝叶斯训练LDA模型

现在我们可以使用`gensim`训练LDA模型。我们需要指定主题数量（$K$）。选择$K$通常是一个迭代过程，涉及评估指标和领域知识。我们还设置超参数 (parameter) (hyperparameter)$\alpha$（文档-主题先验）和$\eta$（常称为$\beta$，主题-词语先验）。如前所述，这些狄利克雷先验影响主题分布的预期稀疏性。`alpha='auto'`和`eta='auto'`让`gensim`从数据中学习这些超参数，这是一个常见做法。

```python
from gensim.models import LdaModel

# 设置主题数量
num_topics = 3 # 从一个合理的猜测开始

# 训练LDA模型（使用变分贝叶斯/EM）
lda_model = LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=num_topics,
    random_state=42, # 用于重现性
    passes=10,        # 训练期间遍历语料库的次数
    alpha='auto',     # 从数据中学习非对称alpha
    eta='auto',       # 从数据中学习非对称eta
    iterations=100    # 每个数据块VB/EM收敛的最大迭代次数
)

print(f"\nLDA模型已用{num_topics}个主题训练完成。")
```

### 评估和解释主题

训练模型只是第一步。我们需要评估发现的主题是否有意义。

#### 1. 人工检查主题

评估主题最常见的方法是检查与每个主题相关的最常出现的词语。好的主题应该包含语义连贯的词语。

```python
# 打印每个主题最常出现的N个词
print("\n每个主题最常出现的词：")
topics = lda_model.print_topics(num_words=5) # 获取每个主题最常出现的5个词
for topic in topics:
    print(topic)

# 示例输出（将根据数据和K值而变化）：
# (0, '0.150*"inference" + 0.095*"variational" + 0.090*"method" + 0.085*"sampling" + 0.070*"distribution"')
# (1, '0.180*"topic" + 0.130*"data" + 0.110*"lda" + 0.090*"model" + 0.075*"text"')
# (2, '0.160*"bayesian" + 0.100*"distribution" + 0.095*"framework" + 0.080*"posterior" + 0.070*"belief"')
```

查看这些词语列表。它们是否似乎代表了您文档中存在明确且可解释的主题？如果主题0包含“inference”、“sampling”、“variational”、“distribution”等词，它可能代表“贝叶斯推断方法”的主题。如果主题1包含“topic”、“model”、“lda”、“text”、“data”等词，它可能与“主题建模应用”有关。

#### 2. 量化 (quantization)评估：困惑度和一致性

尽管人工检查必不可少，但量化指标提供了客观衡量标准。

- **困惑度：** 困惑度传统上用于衡量训练好的模型预测未见数据的能力。较低的困惑度通常表示更好的泛化能力。然而，困惑度并不总是与人类可解释性很好地关联。它在一个保留的测试集上计算。

  ```python
  # 假设您有一个保留的测试语料库：test_corpus
  # perplexity = lda_model.log_perplexity(test_corpus)
  # print(f"\n对数困惑度：{perplexity}")
  # 注意：需要单独的测试集。计算可能比较复杂。
  ```
- **主题一致性：** 衡量主题内高分词 (tokenization)语之间的语义相似性。更高的一致性分数通常与人类对主题质量的判断关联性更好。`gensim`提供了`CoherenceModel`。`c_v`指标通常是一个好的选择。

  ```python
  from gensim.models import CoherenceModel

  # 计算一致性分数 (c_v)
  coherence_model_lda = CoherenceModel(model=lda_model, texts=processed_docs, dictionary=dictionary, coherence='c_v')
  coherence_lda = coherence_model_lda.get_coherence()
  print(f'\n一致性分数 (c_v)：{coherence_lda:.4f}')
  # Example Output: Coherence Score (c_v): 0.5832 (通常越高越好)
  ```

您可以训练不同数量主题（$K$）的模型，并绘制一致性分数与$K$的关系图，以帮助找到合适的主题数量。通常，一致性分数最初会增加，达到峰值，然后可能下降或趋于平稳。



[交互图表：主题一致性与主题数量 (K) 的关系](https://apxml.com/zh/courses/bayesian-machine-learning/chapter-5-advanced-probabilistic-graphical-models/practice-topic-modeling-lda#plot-snvt1d)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "主题一致性与主题数量 (K) 的关系",
    "xaxis": {
      "title": "主题数量 (K)"
    },
    "yaxis": {
      "title": "一致性分数 (c_v)"
    },
    "template": "plotly_white"
  },
  "data": [
    {
      "x": [
        2,
        3,
        4,
        5,
        6,
        7
      ],
      "y": [
        0.45,
        0.58,
        0.55,
        0.51,
        0.48,
        0.46
      ],
      "type": "scatter",
      "mode": "lines+markers",
      "marker": {
        "color": "#228be6"
      }
    }
  ]
}
```

</details>



> 显示一致性如何随主题数量变化的示例图。对于本次运行，峰值表明一个合适的最优K在3或4左右。

#### 3. 使用pyLDAvis可视化主题

交互式图示可以极大地帮助理解。`pyLDAvis`创建一个基于网络的图示，显示主题、它们的普遍性以及与每个主题最相关的词语。

```python
# import pyLDAvis
# import pyLDAvis.gensim_models as gensimvis # 不要跳过这一行
# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning) # 抑制相关警告

# # 准备可视化数据
# vis_data = gensimvis.prepare(lda_model, corpus, dictionary)

# # 显示可视化（在Jupyter notebooks中效果良好）
# # pyLDAvis.display(vis_data)

# # 或者保存为HTML文件
# # pyLDAvis.save_html(vis_data, 'lda_visualization.html')
# print("\npyLDAvis可视化已准备。使用pyLDAvis.display(vis_data)或save_html。")
```

`pyLDAvis`的输出通常显示：

- **左侧面板：** 代表主题的圆圈，大小由普遍性决定。中心之间的距离近似表示主题相似性。
- **右侧面板：** 一个条形图，显示所选主题最相关的词语（由$\lambda$参数 (parameter)控制，平衡主题内的词频与提升度）。

### 使用训练好的模型

一旦对主题满意，您可以使用该模型来：

- **推断新文档的主题分布：**

  ```python
  new_doc_text = "Exploring advanced variational inference algorithms."
  new_doc_processed = preprocess(new_doc_text)
  new_doc_bow = dictionary.doc2bow(new_doc_processed)
  topic_distribution = lda_model.get_document_topics(new_doc_bow)
  print(f"\n新文档的主题分布：{topic_distribution}")
  # Example Output: Topic distribution for new document: [(0, 0.85), (1, 0.05), (2, 0.10)]
  # 表明新文档约85%属于主题0，5%属于主题1，10%属于主题2。
  ```
- **分析整个语料库中的主要主题。**

### 吉布斯采样与变分贝叶斯在实践中的比较

如前所述，`gensim`默认使用VB。实现塌缩吉布斯采样通常需要自定义代码或不同的库（例如`lda`包，尽管其维护可能较少）。

基于之前章节的理论论述：

- **变分贝叶斯（VB/EM）：**
  - *优点：* 通常快得多，更适合大规模数据集，通常提供好的点估计。
  - *缺点：* 倾向于低估后验方差，可能收敛到局部最优，逼近质量取决于所选择的变分族（此处为平均场）。
- **塌缩吉布斯采样：**
  - *优点：* 渐近地从真实后验中采样，可以捕捉复杂的依赖关系，通常更容易推导更新规则。
  - *缺点：* 收敛可能非常慢，需要仔细监测收敛诊断（轨迹图、R-hat，在第2章中论述），在高维空间 (high-dimensional space)中混合可能较差。

对于许多实际主题建模任务，变分贝叶斯的速度和可伸缩性使其成为首选，尤其是在处理大型文本语料库时。然而，如果获得后验不确定性的准确表示或避免平均场近似的潜在偏差很重要，那么吉布斯采样可能会被考虑，尽管计算成本较高。

本次实践练习展示了如何将LDA作为PGM的理论转化为其具体应用。请记住，主题建模通常是一个迭代过程，涉及改进预处理步骤，试验主题数量（$K$），并使用量化 (quantization)指标和定性的人工判断仔细评估结果。

## 参考资料

- [Latent Dirichlet Allocation](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf) — David M. Blei, Andrew Y. Ng, Michael I. Jordan (2003)
  Journal: Journal of Machine Learning Research; Volume: 3; Pages: 993-1022; DOI: [10.1162/jmlr.2003.3.4-5.993](https://doi.org/10.1162/jmlr.2003.3.4-5.993)
  这篇开创性论文介绍了潜在狄利克雷分配模型，为现代主题建模奠定了基础。
- [Online Learning for Latent Dirichlet Allocation](https://proceedings.neurips.cc/paper_files/paper/2010/file/208e438f44d1837acb798781744ce97d-Paper.pdf) — Matthew D. Hoffman, David M. Blei, Francis R. Bach (2010)
  Journal: Advances in Neural Information Processing Systems 23 (NIPS 2010); Publisher: Curran Associates, Inc.; Volume: 23; Pages: 856-864
  该论文提出了用于LDA的高效在线变分贝叶斯算法，是`gensim`实际实现的基础。
- [Machine Learning: A Probabilistic Perspective](https://probml.github.io/pml-book/) — Kevin P. Murphy (2012)
  Publisher: The MIT Press
  这本全面的教科书涵盖了概率机器学习，包括对LDA、贝叶斯公式和推断方法的详细处理。
- [Towards an Automatic Topic Coherence Model](https://www.aclweb.org/anthology/L16-1327.pdf) — Michael Röder, Andreas Both, Andreas Hinneburg (2016)
  Journal: Proceedings of the 5th International Conference on Language, Resources and Evaluation (LREC 2016); Pages: 1989-1996
  该文介绍了多种主题连贯性度量方法，包括`c_v`分数，用于评估主题建模中的主题质量。
- [pyLDAvis: A tool for interactive topic model visualization](https://aclanthology.org/W14-3606/) — Carson Sievert, Kenneth Shirley (2014)
  Journal: Proceedings of the ACL Workshop on Topic Models: Computation, Application, and Evaluation; Publisher: Association for Computational Linguistics; Pages: 12-17; DOI: [10.3115/v1/W14-3606](https://doi.org/10.3115/v1/W14-3606)
  该文阐释了`pyLDAvis`库，它提供了交互式可视化功能，以帮助解释主题模型。

---

[上一节](07-LDA%E7%9A%84%E6%8E%A8%E6%96%AD%EF%BC%9A%E5%8F%98%E5%88%86%E8%B4%9D%E5%8F%B6%E6%96%AF.md) · [下一节](../06-%E8%B4%9D%E5%8F%B6%E6%96%AF%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/01-%E8%B4%9D%E5%8F%B6%E6%96%AF%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E7%9A%84%E7%BC%98%E7%94%B1.md)
