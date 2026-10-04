---
course: "getting-started-with-llm-toolkit"
chapter: "performance-and-cost-optimization"
lesson: "identifying-performance-bottlenecks"
sourceId: 7788
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-llm-toolkit/chapter-9-performance-and-cost-optimization/identifying-performance-bottlenecks"
title: "定位性能瓶颈"
description: "学习如何确定LLM应用中对延迟和成本影响最大的部分。"
order: 1
plots: []
sourceHash: "a05f1f66026666995af0cc6c6890e1f4debc114b697e1f085609e974e99f191d"
sourceCorrections: []
---

为了让应用程序运行更快或更省钱，了解其时间与资源主要花费在哪里非常重要。传统软件的常见瓶颈出现在数据库查询、复杂计算或I/O操作中。尽管LLM应用也有类似情况，但它们引入了两个主要的延迟和成本来源，这些来源通常比其他所有因素都更显著：对外部模型API的调用。

一个典型的应用，尤其是使用检索增强生成（RAG）的应用，会遵循一个多阶段流程。有些阶段在本地运行，通常速度很快，而另一些则涉及对第三方服务的网络请求，这会带来显著的延迟和费用。

> 一个典型的RAG应用工作流程。最主要的瓶颈通常出现在嵌入 (embedding)和生成阶段，这些阶段依赖外部API调用。

让我们分析一下性能问题常见于何处。

### LLM生成调用

最明显的瓶颈是对LLM的最终生成调用。当您的应用发送提示并等待响应时，几个因素会增加延迟：

- **网络延迟：** 您的请求发送到服务提供商服务器以及响应返回所需的时间。
- **模型推理 (inference)时间：** LLM处理您的提示并生成输出令牌所需的时间。对于非常复杂的查询和长响应，这可能需要几秒到一分钟以上。
- **排队：** 在高峰期，您的请求在被模型处理前可能会排队。

成本与使用量直接相关。如引言所述，成本 $C$ 是输入（提示）和输出（完成）令牌的函数：

$C = (P_{prompt} \times N_{prompt}) + (P_{completion} \times N_{completion})$

对于接收许多相同或相似查询的应用，这些成本会累积起来。例如，一个重复回答“你们的营业时间是什么？”的客户支持机器人，每次都会发起一个新的、昂贵的API调用。

### 嵌入 (embedding)API调用

第二个主要瓶颈是嵌入生成。在RAG系统中，每个文档块都必须转换为向量 (vector)嵌入，然后才能在向量数据库中进行索引。尽管这通常是一次性的“摄取”成本，但其金额可能很大。如果您有10,000个文档块，您必须向嵌入服务发起数千次API调用。这个过程可能既耗时又昂贵。

此外，如果您的应用频繁处理新文档或实时为用户查询生成嵌入，这些调用会增加持续的运营成本和延迟。重复调用以嵌入相同的文本，例如常见的搜索词或文档标题，是对资源的低效使用。

### 衡量应用中的瓶颈

优化的第一步是衡量。识别瓶颈的一个简单方法是测量应用工作流程中每个主要阶段的时间。

考虑一个模拟RAG查询过程的简化函数。通过为每个步骤添加计时逻辑，您可以精确找出时间主要花在哪里。

```python
import time

def mock_llm_api_call(prompt):
    """模拟一个缓慢的LLM API调用。"""
    time.sleep(2.5)  # 模拟2.5秒的延迟
    return f"This is a generated response to: {prompt[:50]}..."

def mock_embedding_api_call(text):
    """模拟一个较快但仍有显著耗时的嵌入API调用。"""
    time.sleep(0.1) # 模拟100毫秒的延迟
    return [0.1] * 384 # 返回一个虚拟向量

def run_rag_query(query: str):
    """模拟完整的RAG查询并测量每一步的时间。"""

    print(f"\n正在处理查询：'{query}'")

    # 步骤1：生成查询嵌入
    start_time = time.time()
    query_embedding = mock_embedding_api_call(query)
    embed_duration = time.time() - start_time
    print(f"  1. 嵌入生成：{embed_duration:.4f}秒")

    # 步骤2：检索文档（模拟）
    start_time = time.time()
    time.sleep(0.05) # 模拟本地向量搜索
    retrieved_context = "这里检索到了一些相关上下文。"
    retrieve_duration = time.time() - start_time
    print(f"  2. 文档检索：{retrieve_duration:.4f}秒")

    # 步骤3：调用LLM进行最终生成
    start_time = time.time()
    prompt = f"Context: {retrieved_context}\n\nQuestion: {query}"
    final_response = mock_llm_api_call(prompt)
    generate_duration = time.time() - start_time
    print(f"  3. LLM生成：{generate_duration:.4f}秒")

    total_duration = embed_duration + retrieve_duration + generate_duration
    print(f"  -------------------------------------")
    print(f"  总耗时：{total_duration:.4f}秒")

# 运行模拟
run_rag_query("Kerb工具包是什么？")
```

运行此代码会产生类似如下的输出：

```text
正在处理查询：'Kerb工具包是什么？'
  1. 嵌入生成：0.1002秒
  2. 文档检索：0.0501秒
  3. LLM生成：2.5003秒
  -------------------------------------
  总耗时：2.6506秒
```

结果很明确：LLM生成调用占总请求时间的94%以上。嵌入 (embedding)调用虽然快得多，但仍然比本地检索步骤慢两倍。这个简单的分析立刻告诉我们，优化API调用将带来最大的性能提升。

确定了这些瓶颈后，我们现在可以研究解决方案了。以下章节将向您展示如何使用 `cache` 模块实现缓存策略，通过避免对LLM响应和嵌入的重复API调用，从而大幅减少延迟和成本。

## 参考资料

- [Retrieval-Augmented Generation for Large Language Models: A Survey](https://dl.acm.org/doi/10.1145/3639089) — Yuan-Fang Li, Genggeng Hao, Chunyang Li, Jingyang Ding, Yutong Zhou, Yanmin An, Gang Chen, Jianxin Li, Jun Liu, Xiang Li, Huaijun Li, Yu Han, Haoran Chen, Weizhao Li, Guodong Long, Ruoyu Chen, Cheng Chen, Jie Xu, Chunjing Gan, Quan Z. Sheng, Lei Pan, Kun Xu, Chen Wang, Wei Luo, Shirui Pan, Lei Wang, Xiaohui Tao, Minjuan Zhu, Jie Hu, Faliang Huang, Yonghong Kang, Yi Hu, Jingjing Xu, Tongtong Li, Yuxin Li, Zaiyu Li, Jiawen Lin, Wei Chen, Xifeng Yan, Xiangliang Zhang, Hongzhi Yin, Kai Chen, Bo Li, Guanghua Wang, Quan Li, Zhicheng Dou, Yanyan Shen, Yiming Li, Feifei Li, Chuan Zhou, Pengfei Wang, Peng Zhang, Jinyang Li, Xiangyu Fan, Ruimao Zhang, Dong Guo, Wei Xu, Linzhang Wang, Zhenyu Wang, Yi Wu, Jiajin Li, Qiang Wei, Yang Yang, Xindong Wu, Jianshe Zhou, Zhaoyu Wang, Hao Wang, Xinzhi Gao, Yanchun Zhang (2024)
  Journal: ACM Computing Surveys; Publisher: ACM; Volume: 56; Pages: 1-46; DOI: [10.1145/3639089](https://doi.org/10.1145/3639089)
  一项关于LLM中检索增强生成（RAG）的全面调查，涵盖了架构、挑战和优化策略，对于理解RAG应用程序的性能考量非常相关。
- [Best practices for API usage](https://help.openai.com/en/articles/6614292-best-practices-for-api-key-safety) — OpenAI (2023)
  Publisher: OpenAI
  OpenAI的官方指南，提供了与API交互时降低延迟和成本的策略，包括缓存和批量处理等技术，以解决已识别的瓶颈问题。
- [Designing Data-Intensive Applications: The Big Ideas Behind Reliable, Scalable, and Maintainable Systems](https://www.oreilly.com/library/view/designing-data-intensive-applications/9781491903063/) — Martin Kleppmann (2017)
  Publisher: O'Reilly Media
  一本关于构建健壮、可扩展和高性能数据系统的基础书籍，提供了理解和优化分布式应用程序中延迟、吞吐量和成本的原则，适用于LLM系统的底层基础设施。
