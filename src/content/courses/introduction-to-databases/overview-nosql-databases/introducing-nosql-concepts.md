---
course: "introduction-to-databases"
chapter: "overview-nosql-databases"
lesson: "introducing-nosql-concepts"
sourceId: 1609
sourceUrl: "https://apxml.com/zh/courses/introduction-to-databases/chapter-4-overview-nosql-databases/introducing-nosql-concepts"
title: "NoSQL 基本知识介绍"
description: "阐明 NoSQL（“不仅仅是 SQL”）及其常见属性，如灵活的模式和可伸缩性。"
order: 2
plots: []
sourceHash: "9155205d030fefaa64cc54cb724f1090d7419116a86e1de1029d3450131813aa"
sourceCorrections: []
---

关系数据库凭借其结构化表和预定义模式，对许多应用程序来说非常有用。然而，随着应用需求的增长，特别是伴随互联网的兴起，开发人员遇到了关系模型严格性带来挑战的情况。处理巨量数据、实现极高可用性以及管理无法整齐放入行和列的数据，都需要不同的方法。这促使了一组多样化数据库的产生，它们通常归类在 **NoSQL** 这一术语下。

### NoSQL 是什么意思？

“NoSQL”这个术语可能有点令人困惑。它通常被解释为“不允许 SQL 语言”，但这并不完全正确。对许多系统而言，更准确的解释是\*\*“不仅仅是 SQL”\*\*。它表明摆脱了关系模型及其相关查询语言 SQL 的*单一*主导地位。

NoSQL 并非单一类型的数据库或特定产品。相反，它是一个总称，涵盖了各种数据库技术，旨在解决传统关系数据库可能难以应对的特定需求，例如：

1. **海量数据:** 存储和处理PB乃至EB级别的数据。
2. **高吞吐量 (throughput):** 处理每秒数千或数百万的读写操作。
3. **灵活的数据结构:** 适应没有固定模式或模式快速演变的数据。
4. **高可用性:** 确保即使部分服务器出现故障，系统也能保持运行。

### 为何需要其他方案？

NoSQL 数据库出现的主要驱动因素，通常与在为某些类型的大规模分布式应用扩展传统关系数据库时遇到的局限性有关：

- **可伸缩性难题:** 关系数据库传统上通过向单个服务器添加更多资源（CPU、内存、磁盘）来进行*垂直*扩展（向上扩展）。虽然在一定程度上有效，但存在物理和成本限制。许多现代应用程序需要通过将数据和负载分布到多台通用服务器上来进行*水平*扩展（向外扩展）。许多 NoSQL 数据库从一开始就考虑了水平扩展而设计。

> 比较数据库扩展方法：垂直扩展涉及使单个服务器更强大，而水平扩展涉及增加更多服务器来分摊负载。

- **模式严格性:** 关系数据库强制执行预定义模式。在插入数据之前，您必须定义表、列和数据类型。这提供了数据一致性，但如果数据结构频繁变化，或者您正在处理半结构化或非结构化数据（例如不同用户可能具有不同属性的用户配置文件，或变化的传感器读数），这可能会很麻烦。想象一下，如果试图将每个用户的个人资料信息强制放入完全相同的电子表格列中；如果某些用户拥有而其他用户没有某些信息，这将变得很困难。
- **成本:** 垂直扩展大型关系数据库可能变得非常昂贵，因为需要高端硬件和可能复杂的许可。使用更便宜、标准机器组成的集群进行水平扩展有时会更具成本效益。

### NoSQL 数据库的常见特点

虽然 NoSQL 数据库多样化，但与传统关系系统相比，它们通常有一些共同的一般属性：

- **灵活模式:** 许多 NoSQL 数据库允许您存储没有预定义结构的数据。例如，在文档数据库中，一个记录（文档）可能包含字段 A、B 和 C，而同一集合中的另一个记录包含字段 A、B 和 D。这种灵活性在处理多样或演变的数据时非常有用。这有时被称为“读取时模式”（应用程序在读取数据时解释结构），与关系型数据库的“写入时模式”（写入数据时严格执行结构）形成对比。
- **水平可伸缩性:** 构建为在机器集群上运行，通过增加更多服务器而非无限期升级单个服务器，使它们能够处理大型数据集和高流量负载。数据通常被分区或分散在多台机器上。
- **多样化数据模型:** 它们不仅仅依赖于表/行/列模型。如我们将在后续章节中看到，它们使用键值对、文档、宽列和图等模型，每种都针对不同类型的问题和数据访问模式进行优化。
- **宽松一致性（有时）:** 为实现分布式系统（分布在多台计算机上的系统）的更高可用性和性能，某些 NoSQL 数据库提供*最终一致性*。这意味着如果您将数据写入数据库，在更新传播到所有地方之前，系统不同部分的读取可能会暂时返回旧数据。最终，所有读取都将返回更新后的数据。与关系数据库通常保证的*即时一致性*相比，这是一种权衡，即成功写入意味着后续读取将立即看到该写入。

将 NoSQL 视为 SQL 数据库的补充工具集，而非替代品。SQL 和 NoSQL 之间的选择很大程度上取决于您的应用程序的特定要求、数据的性质以及您在一致性、可用性和可伸缩性方面的优先考虑。在以下章节中，我们将了解特定类型的 NoSQL 数据库，以理解它们独有的模型和应用场景。

## 参考资料

- [NoSQL Distilled: A Brief Guide to the Emerging World of Polyglot Persistence](https://martinfowler.com/books/nosql.html) — Pramod J. Sadalage and Martin Fowler (2012)
  Publisher: Addison-Wesley Professional
  本书简明介绍了NoSQL数据库，阐述了其发展动因、特点和多种数据模型。
- [Designing Data-Intensive Applications: The Big Ideas Behind Reliable, Scalable, and Maintainable Systems](https://dataintensive.net/) — Martin Kleppmann (2017)
  Publisher: O'Reilly Media
  深入探讨了分布式系统、可伸缩性、一致性模型（包括最终一致性）以及现代数据管理的其他基本原理。
- [NoSQL databases in modern applications: A survey](https://www.sciencedirect.com/science/article/pii/S0169023X2030132X) — Yousif H. H. Al-Hajji, Ahmed Al-Hajri, Mohammed Al-Yahyaei (2020)
  Journal: Data & Knowledge Engineering; Publisher: Elsevier B.V.; Volume: 130; Pages: 101768; DOI: [10.1016/j.datak.2020.101768](https://doi.org/10.1016/j.datak.2020.101768)
  这篇综述文章提供了NoSQL数据库的最新概述，涵盖了其特点、分类和应用。
