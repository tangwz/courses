---
course: "multi-agent-llm-systems-design-implementation"
chapter: "core-principles-multi-agent-llm"
lesson: "defining-multi-agent-systems"
sourceId: 6626
sourceUrl: "https://apxml.com/zh/courses/multi-agent-llm-systems-design-implementation/chapter-1-core-principles-multi-agent-llm/defining-multi-agent-systems"
title: "定义多智能体系统 (MAS)"
description: "学习多智能体系统的定义和特征，为大型语言模型智能体开发做好准备。"
order: 1
plots: []
sourceHash: "2bcffe98e84b88dd7a7db729059e80ca1b13c2b35fad4bec9f854d6f182eab55"
sourceCorrections: []
---

多智能体系统 (MAS) 是指一组自主互动、存在于共享环境中的计算实体，这些实体被称为智能体。这些智能体追求个体或集体目标，它们的互动决定了系统的整体行为和能力。与控制中心化的单一应用不同，多智能体系统常具备分布式控制和决策能力。多智能体系统 (MAS) 的定义为理解大型语言模型 (LLMs) 如何赋予智能体高级智能奠定了基础。

### 多智能体系统的核心组成部分

无论具体应用如何，每个多智能体系统通常由三个主要部分构成：

1. **智能体**：这些是系统内活跃的、负责决策的实体。每个智能体都具备一定程度的自主性，这意味着它能独立运作，感知其局部环境，并根据自身目标和可用信息做出选择。在本课程中，我们将侧重于核心智能源自大型语言模型的智能体。
2. **环境**：这是智能体存在、感知、行动和互动的工作背景。环境可以是物理的（例如，机器人智能体的工厂车间）或虚拟的（例如，软件系统、网络或模拟环境）。它提供资源，施加限制，并作为间接通信的媒介（例如，通过一个智能体修改共享资源，另一个智能体随后观察到）。
3. **互动**：多智能体系统中的智能体很少孤立。它们相互沟通、协调、协商或竞争。这些互动可以从简单的消息传递到复杂的协商协议。这些互动的模式和性质决定了多智能体系统的结构和涌现 (emergence)行为。

下图说明了智能体、其环境和互动之间的基本关系。

> 智能体 A、B 和 C 在共享环境中运作，彼此互动（例如沟通），并与共享资源互动。外部输入可以启动任务，系统可以产生集体输出。

### 多智能体系统的定义特征

这些组成部分使多智能体系统具备一些重要特征，这些特征使其有别于其他软件系统：

- **自主性**：每个智能体都掌控自身的行为和内部状态。它无需人类或其他智能体的直接、持续干预即可做出决策。这种自主性使智能体能够主动发起行动以达成目标，并反应式地响应环境变化或来自其他智能体的消息。
- **局部视角（部分可观察性）**：智能体通常对整个系统具有不完整或局部的视角。它只感知环境的一个子集，可能不完全了解其他智能体的状态或意图。这需要信息共享和推理 (inference)机制。
- **去中心化**：在许多多智能体系统中，没有中央控制器指挥所有智能体的行为。控制和数据分布在各个智能体之间。这可以增强系统的韧性（无单点故障）和可扩展性，但也给实现协调一致的整体行为带来了挑战。
- **协调与协作**：为完成复杂任务或系统整体目标，智能体通常需要协调它们的活动。这可能包含信息共享、任务分配、行动同步和冲突解决。协作是指智能体共同努力达成目标，而在某些系统中，智能体也可能争夺资源。
- **涌现 (emergence)行为**：多智能体系统的整体行为源于其组成智能体的局部互动。这种行为有时可能很复杂，并未明确编程到任何单个智能体中，这种特性被称为涌现。在设计期望的涌现行为同时避免不希望的结果，是多智能体系统工程的一个重要方面。

### 为何采用多智能体系统？

多智能体系统的架构选择，主要受益于一些潜在优势，尤其适用于复杂问题：

- **模块化与可重用性**：智能体可以设计为具有特定功能的独立模块，从而使系统更易于开发、调试和维护。专业智能体常可在不同应用中重用。
- **可扩展性**：系统可以通过增加更多智能体来提升规模，每个智能体贡献其能力或资源。
- **韧性与容错性**：由于去中心化，一个智能体的故障可能不会使整个系统瘫痪。其他智能体或许能适应或接管职责。
- **解决固有的分布式问题**：某些问题本身就是分布式的，例如供应链管理、分布式感知或管理复杂互联的系统。多智能体系统提供了一种自然的方式来为这些问题建立模型并加以解决。
- **处理复杂性**：通过将大型复杂问题分解为由专业智能体处理的更小、更易于管理的任务，多智能体系统能提供一种更易于处理的系统设计方法。

例如，考虑一个精巧的电商平台。不必使用单一的、整体式应用，而是可以设计一个多智能体系统，让不同智能体负责库存管理、客户推荐、定价优化、欺诈检测和订单履行。这些智能体将互动、共享数据（例如，通过共享数据库或消息队列）并协调，以提供顺畅的用户体验。每个智能体都可以独立开发和更新，并且可以通过引入新的专业智能体来增加新功能。

### 大型语言模型背景下的多智能体系统

传统上，开发具备所需智能水平、适应性和沟通能力的智能体是一个巨大挑战。强大大型语言模型 (LLMs) 的出现，为构建高能力智能体提供了新可能性。大型语言模型能够提供构成这些智能体认知核心的推理 (inference)、语言理解和生成能力。

随着我们继续，我们将检查大型语言模型如何作为这些自主实体的基本构成块。多智能体系统设计原则提供了框架，而大型语言模型则提供了填充该框架的智能，从而形成了能够复杂、协作解决问题的系统。本章通过明确多智能体系统是什么来奠定基础；后续章节将详细说明如何赋予它们大型语言模型驱动的智能。

## 参考资料

- [An Introduction to MultiAgent Systems](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFZ4GUbzl6gXcFeImWuXk09GFnUEuJLAl3RwpyzbnMY6zJaNhLt4URUH8ztATJLy7_4kLoJ3ppjgaqdSzukpkNMyydoSwjUREAnbIv6EUBQXWgUdMQriLjSXtxj_C_gCdLX4GZ6BGYZF3SQC3rB7UNjsg8ezMzlDwXzflkxIfr9p0rT8j5UhAVNsy3yhHr2bZLSmi_mOL6KEQwk) — Wooldridge, Michael J. (2023)
  Publisher: John Wiley & Sons
  本书提供了多智能体系统的全面基础知识，包括定义、架构和交互协议。
- [Artificial Intelligence: A Modern Approach](https://www.pearson.com/us/higher-education/program/Russell-Artificial-Intelligence-A-Modern-Approach-4th-Edition/PGM335520.html) — Stuart J. Russell, Peter Norvig (2020)
  Publisher: Pearson
  这是一本人工智能标准教材，在更广泛的人工智能背景下包含了智能体和多智能体系统的内容。
- [Generative agents: Interactive simulacra of human behavior](https://arxiv.org/abs/2304.03442) — Joon Sung Park, Joseph C. O'Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, and Michael S. Bernstein (2023)
  Journal: arXiv preprint arXiv:2304.03442; DOI: [10.48550/arXiv.2304.03442](https://doi.org/10.48550/arXiv.2304.03442)
  这项研究介绍了由大型语言模型驱动的生成式智能体，展示了它们在模拟多智能体环境中创建可信行为的能力。
