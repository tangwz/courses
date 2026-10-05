# RAG生产部署的基础设施考量

来源：[原文](https://apxml.com/zh/courses/optimizing-rag-for-production/chapter-1-production-rag-foundations/rag-production-infrastructure)

[返回章节目录](README.md) · [返回课程目录](../README.md)

选择恰当的基础设施是部署生产级检索增强生成（RAG）系统的首要步骤。您的选择将直接影响性能、可伸缩性、运营成本和可维护性。RAG系统并非一个单一的整体应用；它是一个由相互关联的服务组成的流水线，每个服务都有独特的基础设施需求。这些通常包含数据摄取和预处理、向量 (vector)生成、向量存储和搜索，以及语言模型推理 (inference)的组件。

### RAG系统的主要基础设施层面

从宏观层面看，您的RAG系统需要计算、存储和网络资源，并根据每个流水线阶段的特定需求进行调整。

#### 计算资源

RAG流水线中的几项操作都需要计算资源：

1. **向量 (vector)生成**：将您的知识库文档和传入查询转换为向量。这通常是计算密集型任务，特别是对于大型数据集或复杂的向量模型。虽然初始回填可以进行批处理，但增量更新可能需要响应更快的计算。GPU或其他机器学习 (machine learning)加速器可以明显加快此过程。
2. **检索**：在向量数据库中执行相似性搜索。这方面的计算可能由向量数据库本身管理（如果使用托管服务），或者如果自行托管则需要专用资源。对于许多向量搜索工作负载，CPU优化的实例通常已足够，但某些高级索引或搜索算法可能受益于GPU。
3. **重排序（可选但建议）**：如果您使用重排序模型在将搜索结果传递给生成器之前进行优化，此组件也需要自己的计算资源，通常根据模型的复杂性偏向CPU或较小的GPU。
4. **语言模型（LLM）生成**：运行生成器LLM，根据查询和检索到的上下文 (context)合成答案。这通常是RAG流水线中计算最密集的部分，特别是对于更大、能力更强的LLM。高性能GPU（例如NVIDIA A100、H100）通常是达到可接受延迟和吞吐量 (throughput)所必需的。选择自行托管LLM还是通过API使用模型将极大改变您在此处的基础设施需求。

#### 存储

RAG系统中的存储要求多种多样：

1. **源文档存储**：构成您知识库的原始文档。这可以是云对象存储（如Amazon S3、Google Cloud Storage、Azure Blob Storage），也可以是数据库或文档管理系统。
2. **处理后的数据存储**：分块文档、元数据和任何中间表示。对象存储因其可伸缩性和成本效益而常被选用。
3. **向量数据库**：存储文档块的向量。这是一种专门的数据库，针对高维空间 (high-dimensional space)中的快速相似性搜索进行了优化。我们很快将详细讨论向量数据库的选择。
4. **模型存储**：用于存储您的向量模型、重排序模型以及可能自行托管的LLM。通常使用Hugging Face Hub或云平台中的私有模型注册表（例如AWS SageMaker模型注册表、Google Vertex AI模型注册表、Azure ML模型注册表）等服务。
5. **缓存**：用于频繁访问的向量、检索到的文档甚至LLM生成的缓存层可以显著提高性能并降低成本。这可能涉及Redis或Memcached等内存缓存。

#### 网络

高效且安全的网络对于RAG系统组件间的通信非常重要：

- **内部通信**：服务之间（例如，应用前端到检索器，检索器到生成器）的低延迟通信。在虚拟私有云（VPC）内，服务网格或内部负载均衡器很常见。
- **外部通信**：处理用户请求，如果适用，还包括对外部LLM API的调用。API网关在此处对于管理流量、身份验证和速率限制非常重要。
- **数据传输**：用于摄取源文档、传输向量和移动模型工件的带宽。

### RAG组件的基础设施决策

我们来审视一些最有影响力的基础设施决策，这些是您需要做出的选择。

#### 向量 (vector)数据库：托管式 vs. 自行托管

向量数据库是检索阶段的中心。您在此处的选择对运营和成本有重要影响。

- **托管向量数据库服务**：示例包括Pinecone、Weaviate Cloud Services、Zilliz Cloud、带有向量搜索的Redis Cloud、Google Vertex AI Vector Search和带有k-NN的Amazon OpenSearch Service。
  - **优点**：降低运营负担（供应商负责资源配置、伸缩、维护、备份），通常伴随性能优化和SLA。
  - **缺点**：大规模使用时可能更昂贵，存在供应商绑定风险，可能对细粒度配置或底层硬件的控制较少。
- **自行托管向量数据库**：开源选项如Milvus、Weaviate（自行托管）、Qdrant，或者甚至在您自己的基础设施之上使用FAISS等库。
  - **优点**：对部署、硬件选择和配置有更大的控制权；如果您有能力高效管理，直接成本可能更低。
  - **缺点**：运营负担大（设置、伸缩、高可用性、补丁、监控、优化），需要对数据库技术和底层基础设施都有丰富的专业知识。

在选择时，请考虑数据集大小、查询吞吐量 (throughput)要求、期望延迟、更新频率、过滤需求、团队专业知识和预算等因素。对于需要高可用性和可伸缩性但运营资源有限的生产系统，托管服务通常是一个实际的起点。

#### LLM托管：API vs. 自行托管

对于生成组件，您基本有三种途径：

1. **第三方LLM API**（例如：OpenAI, Anthropic, Cohere）：
   - **基础设施影响**：LLM本身直接的基础设施负担很小。您的主要考量是可靠地网络访问API端点以及管理API密钥和配额。
   - **优点**：无需复杂的托管即可访问先进模型，按使用付费。
   - **缺点**：数据隐私问题（数据发送到第三方），网络延迟，潜在的速率限制，对模型更新的控制较少，与使用量相关的持续运营成本。
2. **自行托管开源LLM**（例如：Llama 3, Mistral, Mixtral）：
   - **基础设施影响**：显著。需要强大的GPU实例（大型模型通常需要多个），用于服务的设施（例如使用Triton Inference Server, vLLM或Text Generation Inference等推理 (inference)服务器），以及用于模型部署、伸缩和监控的MLOps专业知识。
   - **优点**：对数据有完全控制，模型可定制（微调 (fine-tuning)），如果优化得当，在极高规模下成本可能更低，核心LLM无外部API依赖。
   - **缺点**：前期和持续基础设施成本高，运营复杂性大，需要专业的机器学习 (machine learning)工程和MLOps技能。
3. **托管LLM服务**（例如：Amazon Foundation, Google Vertex AI Model Garden, Azure AI Studio）：
   - **基础设施影响**：一种折衷方案。云服务提供商管理选定模型的底层GPU基础设施和提供服务堆栈。您将模型部署到端点。
   - **优点**：比完全自行托管更容易访问各种模型，处理部分运营复杂性，数据通常保留在您的云环境中。
   - **缺点**：对于某些模型/使用模式来说，比纯API使用更昂贵，模型选择受供应商限制，控制权少于完全自行托管。

您的决定将取决于预算、数据敏感性、所需模型定制、团队能力和期望的控制水平等因素。许多团队从API开始，进行快速原型开发，然后随着规模扩大或需要更多控制时，评估自行托管或托管服务。

#### 向量模型的计算

尽管不如大型生成LLM要求高，向量模型仍然需要仔细的计算规划：

- **批量向量化 (quantization)**：对于初始知识库索引，您可以使用成本效益高的计算选项，如果您的工作负载具有容错性，可以考虑利用带有GPU的竞价实例。
- **增量/实时向量化**：对于新文档或查询，需要低延迟的向量化。这可能涉及专用实例（CPU或GPU，取决于模型大小和吞吐量）或带有GPU支持的无服务器函数（如果可用且对您的流量模式具有成本效益）。
- **模型大小**：较小的向量模型（例如E5-small、BGE-small）可以在CPU上高效运行，而较大、更准确的模型（例如Cohere-embed-v3、Voyage AI）则显著受益于GPU以提高吞吐量和降低延迟。

### 部署架构和模式

您如何打包和部署RAG组件也将影响您基础设施的形态。

> 生产RAG系统中基础设施组件的典型分解，说明了潜在的服务和数据流。

常见的部署模式包括：

1. **无服务器架构**：（例如，AWS Lambda, Google Cloud Functions, Azure Functions）
   - **适用性**：适用于RAG流水线中事件驱动的部分，例如检索器API端点、查询向量 (vector)化器或较小的重排序器，如果它们的执行时间和资源限制相符。一些平台提供无服务器GPU选项，可能适合较小的LLM推理 (inference)任务或向量生成。
   - **优点**：自动伸缩，按使用付费（对于可变工作负载具有成本效益），减少函数执行环境的运营管理。
   - **缺点**：对于面向用户的服务，冷启动延迟可能是一个问题，执行持续时间和包大小有限制，状态管理可能更复杂，无服务器环境中的GPU可用性和成本可能有所不同。
2. **容器化部署（例如，Kubernetes, Docker Swarm, Amazon ECS, Google Kubernetes Engine, Azure Kubernetes Service）**：
   - **适用性**：对于大多数RAG组件来说，这是一种非常常见且灵活的方法，包括用于检索和生成的API服务、自行托管的向量数据库（带有有状态集），甚至LLM推理服务器。
   - **优点**：可移植性，环境一致，伸缩和编排能力，资源利用高效，监控和日志记录生态系统成熟。
   - **缺点**：Kubernetes学习曲线较陡峭，管理集群的运营负担（尽管托管Kubernetes服务能显著减轻这一点），对于非常简单的部署可能过于复杂。
3. **托管式AI/ML平台**：（例如，Amazon SageMaker, Google Vertex AI, Azure Machine Learning）
   - **适用性**：非常适合部署和管理机器学习 (machine learning)模型，包括向量模型、重排序器和自行托管的LLM。它们通常提供用于创建端点、自动伸缩、监控以及与其他云服务集成的工具。
   - **优点**：简化MLOps任务，为机器学习工作负载提供内置的伸缩和监控功能，通常针对特定硬件（如GPU）进行优化。
   - **缺点**：可能导致供应商绑定，成本可能高于裸机计算，对于非机器学习组件而言，灵活性可能不如纯Kubernetes设置。
4. **混合方法**：许多生产系统采用组合方式。例如，API网关和轻量级处理使用无服务器函数，核心服务和自行托管模型使用Kubernetes，向量数据库或第三方LLM API使用托管服务。

Here's a comparative overview:



[交互图表：<b>RAG组件部署比较</b>](https://apxml.com/zh/courses/optimizing-rag-for-production/chapter-1-production-rag-foundations/rag-production-infrastructure#plot-o0agaf)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "data": [
    {
      "type": "table",
      "header": {
        "values": [
          "<b>维度</b>",
          "<b>无服务器（函数）</b>",
          "<b>容器（例如，Kubernetes）</b>",
          "<b>托管式AI平台</b>"
        ],
        "align": "left",
        "fill_color": "#1c7ed6",
        "font": {
          "color": "white",
          "size": 12
        }
      },
      "cells": {
        "values": [
          [
            "管理开销",
            "可伸缩性",
            "成本模型",
            "定制/控制",
            "冷启动风险",
            "ML专用工具"
          ],
          [
            "低（针对函数执行）",
            "自动，细粒度",
            "按调用/时长付费",
            "受运行时限制",
            "高",
            "基础到中等"
          ],
          [
            "中等（针对集群，托管K8s较低）",
            "高，可配置（Pod，节点）",
            "按底层计算/集群付费",
            "高",
            "低（如果热启动）",
            "可集成，但非主要功能"
          ],
          [
            "低到中等（针对平台服务）",
            "高，托管（端点）",
            "可变（按小时，按推理）",
            "中等到高（针对模型）",
            "低到中等",
            "出色，内置"
          ]
        ],
        "align": "left",
        "fill_color": [
          [
            "#e9ecef"
          ],
          [
            "#dee2e6"
          ],
          [
            "#ced4da"
          ],
          [
            "#dee2e6"
          ],
          [
            "#ced4da"
          ],
          [
            "#dee2e6"
          ]
        ],
        "font": {
          "size": 11
        }
      }
    }
  ],
  "layout": {
    "title": "<b>RAG组件部署比较</b>",
    "height": 400
  }
}
```

</details>



> 常见部署模式的比较，强调了它们在RAG系统组件方面的优劣权衡。

### 数据摄取流水线基础设施

不要忽视数据摄取和预处理流水线的基础设施。该系统负责：

- 获取或接收新的/更新的文档。
- 清洗、解析和分块文档。
- 为新数据块生成向量 (vector)（可以是独立的批量或流处理过程）。
- 将向量索引到向量数据库中。

这方面 Infrastructure 可能包括：

- **编排工具**：Apache Airflow、Kubeflow Pipelines、AWS Step Functions或Google Cloud Workflows，用于管理任务序列。
- **处理引擎**：Apache Spark、Dask或运行在虚拟机/容器上的自定义脚本，用于大规模数据转换。
- **流处理平台**：如果您需要近实时处理文档更新，可使用Apache Kafka或云原生流媒体服务（例如Kinesis、Pub/Sub）。

此处的基础设施选择取决于源数据的体量、速度和多样性。

### 基础设施与可伸缩性、可靠性和安全性的关系

您的基础设施决策与整个系统的非功能性要求密切关联：

- **可伸缩性**：设计每个组件独立伸缩。计算资源使用自动伸缩组，选择可水平或垂直伸缩的向量 (vector)数据库，并采用负载均衡器。
- **可靠性**：为重要组件实现冗余（例如，跨可用区的多个服务实例），酌情使用带有SLA的托管服务，并设计容错能力。（更多内容见第7章）。
- **安全性**：网络隔离（VPC、子网、安全组）、用于控制资源访问的身份和访问管理（IAM）、静态和传输中的加密，以及API密钥等敏感信息的安全管理都非常重要。（本章稍后将在“生产RAG中的安全考量”部分更详细地讨论）。

为您的RAG系统选择恰当的基础设施是一个权衡过程。它涉及衡量性能需求、成本限制、团队的运营能力以及期望的控制水平。通过仔细考虑每个组件的需求并了解不同部署模型的权衡，您可以为生产RAG系统构建一个强大且可持续的稳固架构。此处概述的原则将作为后续章节中讨论更高级优化方法的起点。

## 参考资料

- [Designing Machine Learning Systems: New Era of AI Infrastructure](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) — Chip Huyen (2022)
  Publisher: O'Reilly Media
  一本关于构建、部署和维护机器学习系统的指南，涵盖了基础设施、MLOps实践以及RAG等复杂AI应用的系统设计模式。
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://proceedings.neurips.cc/paper_files/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf) — Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela (2020)
  Journal: Advances in Neural Information Processing Systems; Publisher: Curran Associates, Inc.; Volume: 33; Pages: 9459--9474; DOI: [10.48550/arXiv.2005.11401](https://doi.org/10.48550/arXiv.2005.11401)
  这篇关于检索增强生成（RAG）的基础论文讨论了RAG系统的组件和流程，为基础设施决策提供了理论基础。
- [Google Cloud Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs) — Google Cloud (2024)
  Publisher: Google
  Google Cloud托管机器学习平台的官方文档，提供了部署和管理机器学习模型，包括RAG组件（如嵌入服务和LLM推理）的指导。

---

[上一节](04-RAG%E7%B3%BB%E7%BB%9F%E9%95%BF%E6%9C%9F%E7%BB%B4%E6%8A%A4%E7%9A%84%E9%9A%BE%E9%A2%98.md) · [下一节](06-RAG%E7%BB%84%E4%BB%B6%E7%9A%84%E7%89%88%E6%9C%AC%E6%8E%A7%E5%88%B6%E4%B8%8E%E5%AE%9E%E9%AA%8C%E8%BF%BD%E8%B8%AA.md)
