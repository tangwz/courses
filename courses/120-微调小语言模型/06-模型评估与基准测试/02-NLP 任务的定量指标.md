# NLP 任务的定量指标

来源：[原文](https://apxml.com/zh/courses/fine-tuning-small-language-model/chapter-6-model-evaluation-and-benchmarking/quantitative-metrics-for-nlp-tasks)

[返回章节目录](README.md) · [返回课程目录](../README.md)

人工评估生成的文本通常速度较慢且带有主观性。为了客观地衡量多个训练轮次（epoch）的进展或对比不同的模型检查点（checkpoint），自动化的定量指标必不可少。这些数学评分方法用于评估模型预测标记（token）的置信度，以及其输出内容与参考文本的匹配程度。

### 使用困惑度（Perplexity）衡量模型置信度

困惑度是语言建模中最基础的指标。它衡量模型对词序列的“惊讶”程度。困惑度越低，说明模型为真实文本分配的概率越高，也就是说它对序列的预测越准确。

从数学角度看，它是序列平均负对数似然的指数。如果 $N$ 代表序列中的标记 (token)数量，困惑度通过以下公式衡量模型预测该样本的效果：

$PP(W) = \exp \left( -\frac{1}{N} \sum_{i=1}^{N} \log P(w_i | w_{<i}) \right)$

在这个等式中，$P(w_i | w_{<i})$ 是模型在给定所有前序标记的情况下，为第 $i$ 个标记分配的概率。如果模型非常有把握且预测正确，该概率将趋近于 1，对数趋近于 0，整体困惑度则趋近于 1。如果模型难以预测下一个标记，概率就会下降，导致对数项变得更负，从而推高困惑度。在微调 (fine-tuning)过程中，你通常会在验证数据集上计算困惑度，以确保模型正在学习特定任务的结构。



[交互图表：各轮次的验证困惑度](https://apxml.com/zh/courses/fine-tuning-small-language-model/chapter-6-model-evaluation-and-benchmarking/quantitative-metrics-for-nlp-tasks#plot-vvjc4p)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "各轮次的验证困惑度",
    "xaxis": {
      "title": "轮次",
      "gridcolor": "#e9ecef"
    },
    "yaxis": {
      "title": "困惑度",
      "gridcolor": "#e9ecef"
    },
    "margin": {
      "l": 50,
      "r": 50,
      "b": 50,
      "t": 50
    },
    "showlegend": false,
    "plot_bgcolor": "white"
  },
  "data": [
    {
      "x": [
        1,
        2,
        3,
        4,
        5
      ],
      "y": [
        45.2,
        22.1,
        15.4,
        12.8,
        11.5
      ],
      "type": "scatter",
      "mode": "lines+markers",
      "line": {
        "color": "#339af0",
        "width": 3
      },
      "marker": {
        "size": 8,
        "color": "#1c7ed6"
      }
    }
  ]
}
```

</details>



> 验证困惑度在五个训练轮次中持续下降，表明模型在预测评估数据集时变得更加笃定。

### 使用 ROUGE 对比输出结果

虽然困惑度能反映模型的内部概率分布，但它并不能直接衡量最终生成文本的质量。对于指令遵循、摘要生成或问答等任务，你需要将生成的输出与人工编写的参考文本进行对比。

ROUGE（面向召回的摘要评估辅助工具）是一套旨在评估自然语言处理中自动摘要和机器翻译软件的指标。它计算生成文本与参考文本之间 n-gram 的重合度。

评估小语言模型时会用到几种 ROUGE 变体：

- **ROUGE-1**：衡量一元组（单个词）的重合度。
- **ROUGE-2**：衡量二元组（双词短语）的重合度。这有助于评估模型是否捕捉到了短小的上下文 (context)短语。
- **ROUGE-L**：衡量最长公共子序列。这用于评估句子结构和词序，即便单词之间穿插了其他标记 (token)，只要保持了正确的顺序，模型也会获得评分奖励。

> 在参考字符串和生成字符串之间提取并匹配一元组，用于 ROUGE-1 计算。

对于每种 ROUGE 变体，得分通常分为三个部分。召回率（Recall）衡量模型成功生成的参考文本单词占比。精确率（Precision）衡量生成文本中实际相关且存在于参考文本中的单词占比。F1 分数是精确率和召回率的调和平均数，提供了一个平衡的单项指标。

$F1 = 2 \times \frac{\text{精确率} \times \text{召回率}}{\text{精确率} + \text{召回率}}$

### 实现自动化评估

在 Python 中，你可以使用 Hugging Face 生态系统中的 `evaluate` 库来计算这些指标。这种方式使计算过程标准化，确保你的结果可以与其他机器学习 (machine learning)项目直接对比。

```python
import evaluate

# 加载 ROUGE 评估模块
rouge = evaluate.load("rouge")

# 定义模型的输出和预期的参考文本
predictions = ["the small language model generates accurate text"]
references = ["the fine-tuned language model produces accurate text"]

# 计算得分
results = rouge.compute(predictions=predictions, references=references)
print(results)
```

运行此脚本会输出一个包含 ROUGE-1、ROUGE-2、ROUGE-L 和 ROUGE-Lsum 得分的字典。通过在专门的留出数据集上计算这些数值，你可以为模型建立具体的基准。如果你修改了超参数 (parameter) (hyperparameter)并训练了第二个适配器（adapter），对比 ROUGE 和困惑度得分将客观地告诉你新版本是否比原版有所提升。

## 参考资料

- [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky, James H. Martin (2024)
  Publisher: Stanford University; Pages: Chapter 3
  这是一本标准的教科书，详细解释了 N-gram 语言模型以及困惑度（Perplexity）的数学基础。
- [ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/) — Chin-Yew Lin (2004)
  Journal: Text Summarization Branches Out; Publisher: Association for Computational Linguistics; Pages: 74-81
  介绍用于评估自动文本生成的 ROUGE 指标套件的原始论文。
- [Holistic Evaluation of Language Models](https://arxiv.org/abs/2211.09110) — Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian Zhang, Deepak Narayanan, Yuhuai Wu, Ananya Kumar, Benjamin Newman, Binhang Yuan, Bobby Yan, Ce Zhang, Christian Cosgrove, Christopher D. Manning, Christopher Ré, Diana Acosta-Navas, Drew A. Hudson, Eric Zelikman, Esin Durmus, Faisal Ladhak, Frieda Rong, Hongyu Ren, Huaxiu Yao, Jue Wang, Keshav Santhanam, Laurel J. Orr, Lucia Zheng, Mert Yuksekgonul, Mirac Suzgun, Nathan Kim, Neel Guha, Niladri Chatterji, Omar Khattab, Peter Henderson, Qian Huang, Ryan Chi, Sang Michael Xie, Shibani Santurkar, Surya Ganguli, Tatsunori Hashimoto, Thomas Icard, Tianyi Zhang, Vishrav Chaudhary, William Wang, Xuechen Li, Yifan Mai, Yuhui Zhang, Yuta Koreeda (2022)
  Journal: Transactions on Machine Learning Research (TMLR); DOI: [10.48550/arXiv.2211.09110](https://doi.org/10.48550/arXiv.2211.09110)
  一项关于模型评估的全面研究，将定量指标置于更广泛的基准测试框架中进行分析。

---

[上一节](01-%E8%AF%84%E4%BC%B0%E6%96%87%E6%9C%AC%E7%94%9F%E6%88%90%E8%B4%A8%E9%87%8F.md) · [下一节](03-%E6%B5%8B%E8%AF%95%E6%8F%90%E7%A4%BA%E8%AF%8D%E6%B3%9B%E5%8C%96%E8%83%BD%E5%8A%9B.md)
