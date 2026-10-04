---
course: "planning-optimizing-ai-infrastructure"
chapter: "cost-management-and-optimization"
lesson: "understanding-cloud-pricing-models"
sourceId: 7038
sourceUrl: "https://apxml.com/zh/courses/planning-optimizing-ai-infrastructure/chapter-6-cost-management-and-optimization/understanding-cloud-pricing-models"
title: "了解云定价模式"
description: "比较按需、Spot实例和预留实例等不同的云定价模式，以选择最具成本效益的选项。"
order: 2
plots: ["plots/7038-0.json"]
sourceHash: "53b6c4dafa4c64334128e754d8b058bfec74432234cf716734763fa9b20c4c92"
sourceCorrections: []
---

本地基础设施涉及大量前期投入和可预测的运营开销，而云服务则采用按需付费模式，这提供了很大灵活性，但如果管理不当，可能导致成本复杂且不断上涨。主要云服务提供商，如亚马逊网络服务（AWS）、谷歌云平台（GCP）和微软Azure，提供分层定价结构，旨在适应不同的工作负载模式。理解这些模式间的权衡，是构建经济高效的AI基础设施的核心。

主要想法是将您的工作负载要求、其可预测性、持续时间以及中断容忍度，与最合适的定价模式相匹配。我们来分析您会遇到的三种主要模式。

### 按需实例

按需模式是最直接的定价模式。您可以请求一个虚拟机，例如配备GPU的实例，并按其运行时间（每小时或每秒）支付固定费用。没有长期承诺或预付款。当您使用完毕后，停止实例，计费便会终止。

- **最适合：** 具有不可预测或短期需求的工作负载。这使得按需实例成为初始开发、原型制作和实验的绝佳选择。如果您不确定训练作业的持续时间，或者需要运行快速测试，按需模式的灵活性是其最大优势。
- **缺点：** 这种灵活性伴随着更高的成本。按需实例的每小时成本高于其他模式。将它们用于长期运行且稳定的工作负载在财务上不划算，也是预算超支的常见原因。

例如，您可以使用AWS上的按需`g5.xlarge`实例来调试新的训练脚本。您只需要使用它几个小时，因此为如此短的持续时间支付溢价是完全合理的。

### 预留实例和节省计划

对于具有可预测的长期使用模式的工作负载，云服务提供商会提供大幅折扣以换取承诺。这通过两种相似的机制实现：预留实例（RIs）和节省计划。

- **预留实例（RIs）：** 通过预留实例，您承诺在特定区域使用特定实例类型（例如AWS `p4d.24xlarge`），期限为一年或三年。作为回报，您可以获得高达75%的折扣，相较于按需定价。预留实例最适合极其稳定的工作负载，即您确定在整个期限内都需要该硬件。
- **节省计划：** 这是一种更灵活的承诺模式。您无需承诺特定实例类型，而是承诺在一年或三年期内，在计算服务上花费一定金额（例如每小时10美元）。在此承诺金额内的任何使用都会按折扣费率计费。如果您预计在承诺期内更改实例系列或类型，这会很有利，因为折扣适用范围更广。
- **最适合：** 稳定、长期运行的生产工作负载。一个常见应用场景是托管需要24/7可用的模型推理 (inference)API。为底层计算实例承诺一年的预留实例或节省计划可以大幅降低您的运营成本。同样，如果您有一个核心数据科学家团队持续使用一组训练机器，这些模式能提供大量节省。
- **缺点：** 主要缺点是锁定。您有义务支付整个期限内的承诺使用量，无论您是否使用。这需要仔细的容量规划。

### Spot实例和抢占式虚拟机

Spot实例（在AWS上）或抢占式虚拟机（在GCP上）代表了最具成本效益但波动最大的购买选项。这些实例来自云服务提供商的空闲、未使用的计算能力。您可以竞标此容量，价格比按需价格优惠高达90%。

问题在于，云服务提供商可以随时回收这些实例，几乎没有警告，通常只有两分钟通知。如果提供商需要将容量返还给按需或预留客户，您的Spot实例将被终止。

- **最适合：** 容错、无状态和可中断的工作负载。这使得Spot实例与许多大规模AI训练作业完美匹配。现代深度学习 (deep learning)框架支持检查点，可以定期保存模型状态。如果Spot实例被终止，训练管理器可以简单地请求一个新的实例，并从上一个检查点恢复训练。批量推理 (inference)作业、数据处理管道以及其他非交互式任务也是绝佳选择。
- **缺点：** 波动性。您必须将应用程序设计为能够优雅地处理中断。将Spot实例用于无法容忍突然终止的工作负载，例如面向用户的生产API，将导致无法接受的停机时间。

### 选择合适的模式

选择合适的定价模式直接取决于您工作负载的特性。您的目标是使成本结构与作业的技术和业务要求保持一致。决策过程可以简化为几个重要问题。

> 基于工作负载特性选择云定价模式的决策流程。

这些模式之间的成本差异不容小觑。对于GPU密集型工作负载，做出正确选择可能意味着项目在财务上可行与被放弃之间的区别。



![GPU实例按定价模式的相对每小时成本](plots/7038-0.json)



> GPU实例的相对成本比较。按需模式是100%的基准。1年节省计划可能将成本降至55%，而Spot实例可能将其降至原价的18%。

最终，最有效的策略通常涉及混合方法。您可以使用预留实例覆盖您的基线生产推理 (inference)负载，在Spot实例集群上运行大规模训练作业，并允许开发者使用按需实例实验新模型。通过积极分析您的使用模式并将其映射到这些定价模式，您可以在保持性能的同时控制您的云账单。

## 参考资料

- [Amazon EC2 Pricing](https://aws.amazon.com/ec2/pricing/) — Amazon Web Services (2024)
  Publisher: Amazon Web Services
  提供有关 EC2 按需、预留实例、节省计划和 Spot 实例的信息。
- [Compute Engine pricing](https://cloud.google.com/compute/pricing) — Google Cloud (2024)
  Publisher: Google Cloud
  详细说明 Google Cloud Compute Engine 的定价，包括承诺使用折扣和抢占式虚拟机。
- [Azure Virtual Machines pricing](https://azure.microsoft.com/en-us/pricing/details/virtual-machines/windows/) — Microsoft Azure (2024)
  Publisher: Microsoft Azure
  阐述 Azure 虚拟机的定价，包含预留实例和 Spot 实例。
- [Cloud FinOps: Collaborative, Real-Time Cloud Cost Management](https://www.oreilly.com/library/view/cloud-finops/9781492080829/) — J.R. Storment, Mike Fuller, Kevin Emmerich (2022)
  Publisher: O'Reilly Media
  一本关于 FinOps 实践的指南，提供管理云成本的策略方法。
