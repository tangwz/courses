# 评估LLM压缩与延迟的衡量标准

来源：[原文](https://apxml.com/zh/courses/llm-compression-acceleration/chapter-1-foundations-llm-efficiency-challenges/metrics-compression-latency)

[返回章节目录](README.md) · [返回课程目录](../README.md)

现代LLM的庞大体量要求进行优化以实现实际应用。但我们如何衡量这些优化工作的成效呢？如果模型性能下降到无法接受的程度，仅仅减小模型尺寸或提升速度是不够的。评估优化后的LLM需要采取多方面的方法，平衡效率提升与模型质量的潜在损失。我们需要精确的衡量标准来量化 (quantization)这两个方面。

### 衡量模型质量和忠实度

主要目标通常是在不显著损害其能力的前提下压缩或加速LLM。评估这一点需要仔细的衡量，通常结合使用自动化衡量标准和针对特定任务的基准。

- **标准语言模型衡量标准：** 困惑度（PPL）是衡量语言模型预测给定文本语料库能力的一种常见内在衡量指标。较低的PPL通常表明与数据有更好的统计拟合度。它的计算方法是每个单词平均负对数似然的指数：
  $PPL(W) = \exp\left( -\frac{1}{N} \sum_{i=1}^{N} \log p(w_i | w_1, ..., w_{i-1}) \right)$
  其中 $W = (w_1, ..., w_N)$ 是语料库， $p(w_i | ...)$ 是模型分配的概率。尽管有用，PPL并不总是与下游任务的性能完全相关，特别是复杂的推理 (inference)或生成任务。对于翻译或摘要等任务，BLEU、ROUGE和METEOR等外部衡量标准通过比较生成文本与参考文本来提供更直接的输出质量衡量。
- **下游任务基准：** 评估优化后LLM最具参考价值的方法通常是衡量其在特定预期任务上的性能。这包括使用既定的基准测试套件，例如：

  - **GLUE/SuperGLUE：** 旨在测试通用语言理解能力的各种NLP任务集合。
  - **MMLU（大规模多任务语言理解）：** 评估涵盖广泛主题的零样本和少样本性能。
  - **针对特定方面的基准：** 针对代码生成（HumanEval）、数学推理（GSM8K）或问答（SQuAD）等特定方面定制的评估集。
    在多个相关基准上跟踪性能非常必要，因为优化技术有时可能不成比例地影响某些能力。某项技术可能保持整体困惑度，但会降低推理任务的性能。

"\* **人工评估：** 对于生成模型，自动化衡量标准通常无法捕捉连贯性、创造力、事实准确性或安全性等方面。尽管人工评估资源消耗大，但它仍然是评估生成输出的可用性和质量的重要组成部分。"

- **校准：** 优化，特别是量化 (quantization)，有时会影响模型的校准——即其预测的置信度得分反映实际正确可能性的程度。评估校准（例如，使用预期校准误差）对于需要可靠置信度估计的应用程序来说是必要的。
- **稳定性与公平性：** 压缩和加速可能会在细微处无意中改变模型的行为，可能影响其对分布外输入的稳定性或放大现有偏见。尽管更详细的分析将在后面讨论，初步评估应包含检查这些方面是否受到明显负面影响。

### 衡量效率：压缩、延迟和成本

效率衡量标准量化 (quantization)通过优化技术获得的收益。这些通常分为与尺寸、速度和计算资源相关的类别。

- **压缩衡量标准：**

  - **模型大小（参数 (parameter)）：** 模型中可训练参数的原始数量。尽管有指示作用，但如果涉及不同数据类型，它并不能直接反映内存使用情况。
  - **模型大小（存储/内存）：** 存储模型权重 (weight)所需的实际磁盘空间（例如，以兆字节或千兆字节为单位）。这直接受到量化（例如，FP32对比INT8对比NF4）的影响。这一衡量标准与加载模型所需的RAM紧密相关。
  - **压缩比：** 比较优化后模型尺寸与原始尺寸的相对衡量标准：
    $\text{压缩比} = \frac{\text{原始模型大小（字节）}}{\text{压缩后模型大小（字节）}}$
  - **稀疏度：** 对于剪枝技术，这衡量的是被设置为零的权重的百分比。它的计算方法是：
    $\text{稀疏度} = \frac{\text{零权重数量}}{\text{总权重数量}} \times 100\%$
    稀疏度的*类型*（非结构化对比结构化）也相关，因为结构化稀疏通常能更直接地转化为硬件加速。
- **延迟和吞吐量 (throughput)衡量标准：** 这些衡量推理 (inference)速度。

  - **延迟（每token时间）：** 对于自回归 (autoregressive)模型，这是生成单个输出token所需的平均时间。对于交互式应用程序，越低越好。
  - **延迟（首token时间 - TTFT）：** 从发送输入提示到接收到第一个输出token所经过的时间。这极大地影响了系统感知的响应速度。
  - **延迟（总生成时间）：** 生成预定义长度（例如，512个token）的完整序列所需的总时间。
  - **吞吐量：** 衡量系统在给定时间内可以处理的操作数量。常用单位包括：
    - 每秒token数（总体生成速率）。
    - 每秒请求数（对于服务系统，通常取决于批次大小和序列长度）。
      更高的吞吐量表示更好的系统容量。测量时应指定批次大小和序列长度，因为这些会显著影响结果。
- **计算成本衡量标准：**

  - **FLOPs（浮点运算次数）：** 单次推理过程所需的浮点计算总数的理论衡量。尽管对于架构比较有用，但它通常与实际延迟不完全相关，因为它忽略了内存访问成本、并行性以及特定硬件优化。通常以GFLOPs（千兆浮点运算次数）或TFLOPs（万亿浮点运算次数）为单位。
  - **MACs（乘积累加运算）：** 类似于FLOPs，在神经网络 (neural network)中常可互换使用。
  - **能耗：** 以每次推理的焦耳数或运行期间的平均功率（瓦特）来衡量。这对于移动/边缘部署和环境可持续性变得越来越重要。

### 理解权衡

优化很少是免费的。模型忠实度（准确性、质量）与效率提升（尺寸、速度）之间几乎总是存在权衡。目标是推进帕累托前沿——在给定忠实度水平下实现尽可能好的效率，反之亦然。将这些权衡可视化对于为特定使用场景选择正确的优化策略来说非常必要。



[交互图表：LLM优化权衡：忠实度对比延迟](https://apxml.com/zh/courses/llm-compression-acceleration/chapter-1-foundations-llm-efficiency-challenges/metrics-compression-latency#plot-lcgwmz)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "LLM优化权衡：忠实度对比延迟",
    "xaxis": {
      "title": "推理延迟（毫秒/token）",
      "autorange": "reversed"
    },
    "yaxis": {
      "title": "任务准确度（例如，MMLU分数）"
    },
    "legend": {
      "title": "优化方法"
    },
    "margin": {
      "l": 50,
      "r": 20,
      "t": 40,
      "b": 40
    },
    "width": 600,
    "height": 400
  },
  "data": [
    {
      "x": [
        50,
        40,
        38,
        35,
        25,
        20
      ],
      "y": [
        75.0,
        74.5,
        74.0,
        73.0,
        72.0,
        68.0
      ],
      "mode": "markers+text",
      "type": "scatter",
      "name": "方法",
      "text": [
        "基线 (FP16)",
        "PTQ (INT8)",
        "QAT (INT8)",
        "剪枝 (50%)",
        "PTQ (INT4)",
        "蒸馏"
      ],
      "marker": {
        "size": 12,
        "color": [
          "#4263eb",
          "#1c7ed6",
          "#228be6",
          "#37b24d",
          "#15aabf",
          "#ae3ec9"
        ]
      },
      "textposition": "top right"
    }
  ]
}
```

</details>



> 任务准确度与推理 (inference)延迟之间的权衡，针对应用于LLM的不同优化技术。图中偏向右上方的点表示高准确度和低延迟的更好组合。

选择合适的衡量标准在很大程度上取决于目标应用程序和部署限制。实时聊天机器人优先考虑低TTFT和每token延迟，而批处理系统可能优先考虑吞吐量 (throughput)和能效。此外，延迟和吞吐量基准只有在与测试所用的特定硬件（CPU、GPU型号、内存）和软件堆栈（如TensorRT、vLLM、ONNX Runtime等推理库）相关联时才具有意义。严谨和标准化的基准测试对于有效比较技术来说非常必要。理解这些衡量标准为评估后续章节中讨论的先进优化技术奠定了基础。

## 参考资料

- [Holistic Evaluation of Language Models](https://arxiv.org/abs/2211.09110) — Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian Zhang, Deepak Narayanan, Yuhuai Wu, Ananya Kumar, Benjamin Newman, Binhang Yuan, Bobby Yan, Ce Zhang, Christian Cosgrove, Christopher D. Manning, Christopher Ré, Diana Acosta-Navas, Drew A. Hudson, Eric Zelikman, Esin Durmus, Faisal Ladhak, Frieda Rong, Hongyu Ren, Huaxiu Yao, Jue Wang, Keshav Santhanam, Laurel Orr, Lucia Zheng, Mert Yuksekgonul, Mirac Suzgun, Nathan Kim, Neel Guha, Niladri Chatterji, Omar Khattab, Peter Henderson, Qian Huang, Ryan Chi, Sang Michael Xie, Shibani Santurkar, Surya Ganguli, Tatsunori Hashimoto, Thomas Icard, Tianyi Zhang, Vishrav Chaudhary, William Wang, Xuechen Li, Yifan Mai, Yuhui Zhang, Yuta Koreeda (2023)
  Journal: Transactions on Machine Learning Research (TMLR); DOI: [10.48550/arXiv.2211.09110](https://doi.org/10.48550/arXiv.2211.09110)
  提出了一个评估语言模型的综合框架，涵盖准确性、鲁棒性、公平性和效率等多样化指标。
- [Measuring Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300) — Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, Jacob Steinhardt (2021)
  Journal: International Conference on Learning Representations (ICLR 2021); DOI: [10.48550/arXiv.2009.03300](https://doi.org/10.48550/arXiv.2009.03300)
  提出了一个基准，用于在零样本和少样本设置下评估语言模型在广泛知识和推理任务上的表现。

---

[上一节](03-%E5%AE%9E%E7%8E%B0%E6%95%88%E7%8E%87%E7%9A%84%E6%9E%B6%E6%9E%84%E8%80%83%E9%87%8F.md) · [下一节](05-LLM%20%E9%83%A8%E7%BD%B2%E7%9A%84%E7%A1%AC%E4%BB%B6.md)
