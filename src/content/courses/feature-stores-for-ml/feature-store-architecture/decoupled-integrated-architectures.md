---
course: "feature-stores-for-ml"
chapter: "feature-store-architecture"
lesson: "decoupled-integrated-architectures"
sourceId: 3353
sourceUrl: "https://apxml.com/zh/courses/feature-stores-for-ml/chapter-1-feature-store-architecture/decoupled-integrated-architectures"
title: "解耦架构与集成架构的比较"
description: "Analyzing the trade-offs between tightly integrated and loosely coupled feature store system designs."
order: 5
plots: []
sourceHash: "5fb4e19e48c689a91629d2702af5ee2f3a2eff3eb0189422e0c89a67209147a5"
sourceCorrections: []
---

特征存储的基本组成部分包括注册层、在线/离线存储、转换引擎和 serving 层。考虑到这些组件，一个主要的架构决定出现了：这些组件的耦合程度应该有多紧密？特征存储应该采用单一的、统一的系统，还是从不同的、专用组件来构建？将分析集成式和解耦式特征存储架构之间的权衡，从而为这一设计选择提供依据。

## 集成式特征存储架构

集成式架构通常在一个单一、协调的系统或平台内提供主要特征存储组件。这通常是托管云服务或完整的特征存储软件套件的特点。在此模式下，注册层、在线存储、离线存储，以及可能的转换层和 serving 层被设计为协同工作，常常共享一个公共控制平面、API 和元数据层。

### 特点

- **统一界面：** 用户通过单一的API或UI与各组件进行交互。
- **托管式互联：** 在线和离线存储之间的数据移动与同步，以及与注册层的交互，通常由平台内部处理。
- **一致的工具：** 特征定义、验证和监控通常使用集成平台提供的工具。

> 集成式架构通常将核心组件打包成一个统一的平台。

### 优点

- **操作简化：** 管理单一系统相比协调多个独立组件可以降低操作负担。设置和配置可能更快。
- **潜在的一致性保障：** 集成系统通常提供更强的内置机制，以管理在线和离线视图之间的一致性，并确保时间点正确性。
- **厂商支持：** 单个厂商为整个技术栈提供支持，简化了故障排除。
- **流畅的用户体验：** 跨不同功能的一致界面可以提升数据科学家和工程师的易用性。

### 缺点

- **灵活性降低：** 您通常受限于平台厂商为每个组件选择的特定实现（例如，在线存储的底层数据库技术）。更换单个组件通常很困难或不可能。
- **厂商锁定：** 脱离集成平台进行迁移可能复杂且代价高昂，因为多个重要功能都与它绑定。
- **潜在的瓶颈：** 集成系统中某个组件的性能或可扩展性限制可能限制整个特征存储系统。
- **“最小公分母”特性：** 平台可能提供普遍适用的功能，但可能缺乏同类最佳独立工具中提供的专业功能。
- **整体更新：** 对平台的升级可能同时影响所有组件，可能增加风险或强制全面采纳更改。

## 解耦式特征存储架构

解耦式或可组合架构涉及为不同特征存储功能选择和整合独立系统。您可以使用BigQuery、Snowflake或Redshift等数据仓库作为离线存储，使用Redis、Cassandra或DynamoDB等低延迟NoSQL或内存数据库作为在线存储，使用Spark或Flink等处理框架进行转换，并可能使用一个独立的系统进行特征注册和元数据管理（例如，使用Feast等开源框架作为协调层）。

### 特点

- **独立组件：** 特征存储的每个部分（存储、计算、注册层）都是一个独立的系统或服务。
- **明确的集成：** 组件之间的连接是明确构建和管理的，通常通过API、数据管道（例如Airflow、Kubeflow Pipelines）和事件流进行。
- **异构技术：** 允许使用最适合每个具体任务的专用工具。

> 解耦架构将专用、独立的系统组合起来，以实现每个特征存储功能。

### 优点

- **最大灵活性：** 根据具体的性能、规模、成本或功能要求选择最适合的工具。例如，选择一个针对尽可能低的$T_{retrieval}$进行优化的在线存储。
- **独立扩展：** 根据每个组件的特定负载独立地扩展（计算、在线存储、离线存储）。
- **使用现有基础设施：** 与组织内部已有的数据湖、数据仓库、计算集群和数据库进行整合。
- **避免厂商锁定：** 随着技术进步或需求变化，更容易替换或升级单个组件。
- **潜在的成本/性能优化：** 为实现最佳成本性能，对技术栈的每个部分进行精细调整。

### 缺点

- **集成复杂度增加：** 需要大量的工程投入来设计、构建和维护组件之间的连接和数据流。
- **一致性挑战：** 确保跨分布式、独立系统的数据一致性（例如，在线/离线对等性、时间点正确性）需要细致的设计和可靠的机制。
- **更高的操作开销：** 监控、管理和升级多个独立系统可能更复杂。
- **需要更广泛的专业知识：** 团队需要对为不同组件选择的各种技术都有所了解。
- **潜在的“胶水代码”维护：** 将组件连接在一起的自定义代码和配置需要持续维护。

## 混合方法

值得注意的是，选择并非严格的二元对立。混合模式很普遍，组织可能对某些核心功能（如在线/离线存储和基本注册层）使用托管服务，但将其与运行在独立计算平台（如Spark）上的自定义转换管道整合，或者使用外部系统进行高级元数据管理和血缘追踪。这可以在使用托管服务与在重要方面保持灵活性之间提供平衡。

## 如何选择：要考虑的因素

最适合的架构很大程度上取决于您的具体情况：

1. **团队专业能力和规模：** 您是否有工程资源和专业知识来整合并管理多个复杂的系统（倾向于解耦式），还是操作简便性是更高的优先级（倾向于集成式）？
2. **现有基础设施：** 您能否使用组织内数据仓库、数据湖或计算平台上的现有投入（倾向于解耦式）？
3. **规模和性能需求：** 您对在线服务是否有极低的延迟要求，或者是否有庞大的离线计算需求，这可能超出集成平台的能力（倾向于解耦式）？
4. **灵活性和演变：** 您的需求或底层技术变化的M可能性有多大？独立更换组件的能力重要吗（倾向于解耦式）？
5. **上市时间：** 您是否需要快速使特征存储投入运行（可能倾向于集成式）？
6. **预算：** 比较托管服务的成本（集成式）与构建和管理组合系统（解耦式）的工程和运营成本。
7. **治理与合规：** 集成系统可能提供更多的开箱即用治理功能，但解耦系统允许与专用的外部治理工具进行整合。

在集成式和解耦式架构之间进行选择是一个根本性决定，对灵活性、可扩展性、成本和操作复杂性具有长期影响。在确定具体路径之前，请仔细评估您组织的具体需求、技术能力和现有基础设施。

## 参考资料

- [Tecton: A Fully Managed Feature Store for Real-time ML](https://www.vldb.org/pvldb/vol13/p1618-sculley.pdf) — Mike Sculley, Paul O'Rourke, Greg Demitriades, Josh Lee, Nick Frosst, Jeremy Hylton, Mike S. Kim, Sam Charrington, Ben Van Durme, Kevin Jue, Matt Glickman (2020)
  Journal: Proceedings of the VLDB Endowment; Publisher: VLDB Endowment; Volume: 13; Pages: 1618-1631; DOI: [10.14778/3407790.3407817](https://doi.org/10.14778/3407790.3407817)
  描述了托管式集成特征存储的架构和优势，强调一致性和操作便捷性。
- [Designing Machine Learning Systems: An Iterative Process for Production-Ready Applications](https://learning.oreilly.com/library/view/designing-machine-learning/9781098107956/) — Chip Huyen (2022)
  Publisher: O'Reilly Media
  构建机器学习系统的指南，涵盖数据管理和组件集成的架构决策，适用于集成和解耦方法。
- [Feast Documentation: Architecture Overview](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHHC6AIoZttFEW-44z6lzgGE5MARTlw56QFW7fSV9Sjna2ZMmikaQiVZJVm-KgWV2rZ79sYaYRxzL5CwbtLG8dE6yF2MMOr28m4c4l4EqTKqM8qpgvbsJXUjttA3TNH43notQmCCUhRx5xx30-xfgfuup94YeO0eQ==) — Feast Project Maintainers (2025)
  Publisher: Feast Project
  解释了Feast如何作为独立选择组件的编排层，展示了解耦的特征存储架构。
- [MLOps: A Taxonomy and Survey](https://doi.org/10.1016/j.future.2022.06.012) — Mohamed Ibrahim and Maedeh Saeedi (2022)
  Journal: Future Generation Computer Systems; Publisher: Elsevier; Volume: 136; Pages: 44-64; DOI: [10.1016/j.future.2022.06.012](https://doi.org/10.1016/j.future.2022.06.012)
  提供了MLOps架构模式和挑战的综述，包括数据基础设施和系统集成方面的考虑。
