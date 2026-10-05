# 从MLOps到LLMOps的转变

来源：[原文](https://apxml.com/zh/courses/mlops-for-large-models-llmops/chapter-1-foundations-llmops/mlops-to-llmops-transition)

[返回章节目录](README.md) · [返回课程目录](../README.md)

机器学习 (machine learning)生命周期管理的传统MLOps原则提供了一个基础，但将其直接应用于大型语言模型（LLM）会显现出显著的差异。LLMOps不仅仅是MLOps的简单规模扩大；它涉及到工具、基础设施、方法和关注点的根本性变化，因为这些庞大模型具有独特的属性。这一转变主要在哪些方面需要调整将被审视。

### 规模：从兆字节到太字节

传统的ML模型通常大小从兆字节到数千兆字节不等。然而，LLM的参数 (parameter)数量 $P$ 常常达到数十亿甚至数万亿，这使得模型检查点占用数百千兆字节或太字节的空间。

- **模型大小**：像ResNet-50这样的标准计算机视觉模型可能约为100MB。像GPT-3这样的LLM拥有1750亿参数，仅以半精度（FP16）存储就需要约350GB。更大型的模型会超过太字节。这种差异会影响存储、传输、加载时的内存需求以及版本管理策略。
- **数据量**：训练LLM需要庞大的、通常是拍字节规模的文本和代码语料库。传统的MLOps处理大型数据集，而LLMOps则在完全不同的量级上运行，需要专门的数据存储、处理和治理方案，能够高效处理非结构化数据。



[交互图表：典型规模对比：MLOps与LLMOps](https://apxml.com/zh/courses/mlops-for-large-models-llmops/chapter-1-foundations-llmops/mlops-to-llmops-transition#plot-184f88f)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "典型规模对比：MLOps与LLMOps",
    "xaxis": {
      "title": "资产"
    },
    "yaxis": {
      "type": "log",
      "title": "大小（约对数刻度）",
      "tickformat": ".1e"
    },
    "barmode": "group",
    "height": 350,
    "legend": {
      "orientation": "h",
      "yanchor": "bottom",
      "y": -0.4,
      "xanchor": "center",
      "x": 0.5
    }
  },
  "data": [
    {
      "type": "bar",
      "name": "传统MLOps",
      "x": [
        "模型大小",
        "数据集大小"
      ],
      "y": [
        100000000.0,
        1000000000000.0
      ],
      "marker": {
        "color": "#4dabf7"
      }
    },
    {
      "type": "bar",
      "name": "LLMOps",
      "x": [
        "模型大小",
        "数据集大小"
      ],
      "y": [
        500000000000.0,
        1000000000000000.0
      ],
      "marker": {
        "color": "#be4bdb"
      }
    }
  ]
}
```

</details>



> 近似规模对比，显示了传统MLOps和LLMOps环境中典型模型和数据集大小的数量级差异。请注意y轴的对数刻度。

### 计算需求：训练与推理 (inference)

LLM所需的计算资源远超大多数传统模型的需求。

- **训练**：从头开始训练一个大型基础模型可能需要数千个高端GPU（如A100或H100）连续运行数周或数月，耗费数百万美元的计算成本。这需要复杂的分布式训练技术（数据并行、张量并行、流水线并行），通过DeepSpeed或Megatron-LM等框架管理，这些方法在标准MLOps中通常是不必要的。对这些庞大、长时间运行的任务进行编排、检查点保存并确保容错，需要专门的操作工具和专业知识。
- **推理**：部署LLM带来了独特的挑战。将数百千兆字节的模型加载到GPU内存并非易事。实现可接受的延迟和吞吐量 (throughput)通常需要模型优化（量化 (quantization)、蒸馏）、专门的推理服务器（例如vLLM、TensorRT-LLM）以及多GPU服务策略。仅仅部署一个容器化模型端点（在MLOps中很常见）通常是不够的，原因在于内存限制以及对连续批处理或分页注意力机制 (attention mechanism)的需求以提高效率。

### 方法论上的转变：从完整重新训练到持续适应

开发和更新周期存在显著差异。

- **微调 (fine-tuning)与重新训练**：鉴于预训练 (pre-training)的极高成本，LLM的完整重新训练很少见。重点显著转向微调（使预训练模型适应特定任务），并越来越多地倾向于参数 (parameter)高效微调（PEFT）技术，如LoRA或Adapters。将PEFT投入实际使用、管理适配器权重 (weight)并高效组合它们，成为LLMOps的常规任务，这与管理完全重新训练的模型有所不同。
- **提示工程 (prompt engineering)即代码**：LLM的性能对输入提示高度敏感。有效的LLMOps将提示工程纳入操作流程，包括提示版本管理、A/B测试提示变体以及监控提示效果，将提示几乎视为需要严格管理的代码或配置产物。
- **检索增强生成（RAG）**：许多LLM应用依赖于RAG系统，这引入了像向量 (vector)数据库这样的新操作组件。管理索引管道、确保向量存储中的数据时效性、监控检索器和生成器之间的配合，以及处理分块策略，这些都增加了标准模型部署的复杂性。

### 监控与评估：准确性

传统的MLOps监控准确性、精确度、召回率或AUC等指标，而LLMOps需要更广泛且更全面的监控功能集。

- **输出质量**：评估LLM的输出需要考量连贯性、相关性、安全性（毒性、偏见检测）以及事实一致性（幻觉 (hallucination)检测）等多个方面。这些通常需要复杂的评估流程，有时涉及其他模型、统计测量或集成到操作流程中的人工反馈环节。标准的分类/回归指标是不够的。
- **性能指标**：延迟（首令牌时间、每令牌延迟）和吞吐量 (throughput)（每秒令牌数）成为主要的性能指标，同时还有推理 (inference)过程中的GPU利用率、GPU内存使用情况和网络带宽。这些与典型的Web服务延迟或批处理任务吞吐量有显著不同。
- **成本追踪**：鉴于训练和推理的GPU资源运营成本高昂，针对LLM应用，按请求、按用户或按生成的令牌进行细粒度的成本监控和归属成为一项重要需求。
  "\* **漂移**：漂移检测不仅适用于输入数据分布（例如用户查询变化），也适用于生成输出随时间变化的相关性、风格和安全性，这需要根据不断变化的规范或调整进行持续评估。鉴于生成的开放性特点，语义漂移尤其具有挑战性。"

### 工具生态系统发展

MLOps工具链通常需要为LLMOps进行增强或替换。尽管用于实验追踪（MLflow、W&B）、CI/CD（Jenkins、GitLab CI）和基础设施监控（Prometheus、Grafana）的工具仍然适用，但专门的工具正在涌现 (emergence)或变得日益重要，用于以下方面：

- 分布式训练编排与管理（例如Ray、Slurm、带特定操作器的Kubernetes）。
- 分布式训练效率框架（例如DeepSpeed、Megatron-LM、FSDP）。
- 大型模型/数据版本管理（例如大规模Git LFS、DVC适配方案、LakeFS等专门平台或专有解决方案）。
- 优化推理 (inference)服务引擎（例如带TensorRT-LLM后端的NVIDIA Triton Inference Server、vLLM、Text Generation Inference）。
- PEFT库集成与实际应用工具。
- 向量 (vector)数据库管理与扩展（例如Weaviate、Pinecone、Milvus、专门的云服务）。
- LLM评估与可观测性平台，关注输出质量、毒性和幻觉 (hallucination)（例如LangSmith、Arize AI、WhyLabs、定制解决方案）。

总而言之，从MLOps到LLMOps的转变，意味着模型、数据和计算规模的显著增长；适应训练和推理的不同计算方式；采用侧重于微调 (fine-tuning)、提示和RAG的新开发与维护方法；扩大监控范围，超越传统ML指标；并集成一套快速变化的专门工具。理解这些差异，是成功将大型语言模型投入实际使用的核心。

## 参考资料

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (2021)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.2106.09685](https://doi.org/10.48550/arXiv.2106.09685)
  介绍了LoRA，一种参数高效微调技术，已成为在不完全重新训练的情况下适应LLM的核心方法，与LLMOps方法论直接相关。
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11477) — Patrick Lewis, Ethan Perez, Aleksa Gordić, Armand Joulin, and Sebastian Riedel (2020)
  Journal: Advances in Neural Information Processing Systems 33 (NeurIPS 2020); Publisher: NeurIPS; Volume: 33; Pages: 9444-9457; DOI: [10.48550/arXiv.2005.11477](https://doi.org/10.48550/arXiv.2005.11477)
  提出了检索增强生成（RAG）框架，结合了信息检索和生成模型，这是LLM应用中的一个关键架构模式。

---

[下一节](02-%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E5%9C%A8%E7%94%9F%E4%BA%A7%E4%B8%AD%E7%9A%84%E7%89%B9%E6%9C%89%E6%8C%91%E6%88%98.md)
