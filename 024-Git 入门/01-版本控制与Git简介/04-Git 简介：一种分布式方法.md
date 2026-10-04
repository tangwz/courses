# Git 简介：一种分布式方法

来源：[原文](https://apxml.com/zh/courses/getting-started-with-git/chapter-1-intro-version-control-git/introducing-git)

[返回章节目录](README.md) · [返回课程目录](../README.md)

分布式版本控制系统是现代软件开发中的主流方法。这些系统解决了集中式版本控制中存在的局限性。在分布式系统中，Git 处于领先地位。

Git 由 Linus Torvalds 于 2005 年创建，他也是 Linux 操作系统内核的创造者。他需要一个更好的工具来管理 Linux 内核高度分散的开发工作，一个比当时现有选项更快、更可靠、更适合非线性开发（想象一下数千名开发人员同时处理不同功能的情形）的工具。其成果就是 Git，一个从设计之初就注重速度、数据完整性以及对分布式工作流程支持的系统。

Git 为何被称为“分布式”？与集中式系统不同，集中式系统的完整项目历史记录存放在单个服务器上，而像 Git 这样的分布式版本控制系统（DVCS）让每位开发人员都在其本地机器上拥有整个仓库的完整副本。这不仅仅是文件的最新版本；它包含更改的*完整历史记录*、每个提交、每个分支，以及所有内容。

可以这样理解：在集中式系统中，主图书馆保管着一本书的原始副本，您借出特定章节进行工作。而在像 Git 这样的分布式系统中，每个参与者在开始时都会获得整本书的完整、未经删减的副本。他们可以使用自己的个人副本阅读，在空白处做笔记（提交），甚至起草全新的章节（分支）。

> 一种分布式模式，其中每位开发人员都有仓库的完整本地副本，通常与中央远程服务器同步。

这个基本区别带来了几个重要优点：

1. **速度快：** 由于您在本地拥有整个仓库，大多数 Git 操作（如查看历史记录、比较版本、创建分支或提交更改）都非常快。它们直接在您的机器上进行，无需通过网络通信。只有当您明确地将本地仓库与远程仓库同步时，网络延迟才会成为影响因素。
2. **离线工作：** 由于仓库位于本地，即使断开网络连接，您也能有效工作。您可以进行提交、创建分支、浏览项目历史记录等。只有当您想与他人共享您的更改或获取他们的最新更新时，才需要网络连接。
3. **数据完整性和冗余：** 仓库的每个副本（每个“克隆”）都可作为完整的备份。如果您用于协作的主服务器崩溃或其数据损坏，通常可以从任何开发人员的本地副本恢复整个历史记录。Git 内部还使用校验和（特别是 SHA-1 散列）来确保数据的完整性；数据在 Git 未察觉的情况下损坏是非常困难的。
4. **灵活的工作流程：** 分布式特性支持多种协作模式。虽然使用中央“中心”仓库（例如在 GitHub 或 GitLab 上）很常见，但 Git 也支持开发人员之间的直接共享，或在需要时支持更复杂的层级工作流程。

需要注意的是，虽然 Git *本质上*是分布式系统，但许多团队仍将中央远程仓库用作项目的权威来源。开发人员从这个中央仓库克隆，将更改推送回它，并从中拉取更新。这为协作提供了一个方便的焦点，但并未改变 Git 本身底层的分布式特性。每位开发人员仍然在本地保留一个完整的、独立的仓库副本。

Git 的方法侧重于项目随时间变化的快照。当您提交时，Git 本质上是拍下您所有文件在那个时刻的样子，并存储对该快照的引用。这使得分支和合并等操作与跟踪单个文件更改的系统相比尤其高效。

理解这种分布式理念是有效使用 Git 的基本要求。它影响着您的工作方式、协作方式，以及某些命令行为的原因。在接下来的章节中，我们将了解这种设计如何转化为管理项目的实际命令。

## 参考资料

- [Pro Git](https://git-scm.com/book/en/v2/) — Scott Chacon and Ben Straub (2014)
  Publisher: Apress
  Git 的官方综合指南，涵盖其历史、分布式特性和核心概念。可在线免费获取。
- [Version Control (Git) - The Missing Semester of Your CS Education](https://missing.csail.mit.edu/2024/version-control/) — Anish Athalye, Jon Gjengset, Jose Javier Gonzalez Ortiz, Irena Huang, and Sanjana Wason (2024)
  Publisher: MIT Computer Science and Artificial Intelligence Laboratory
  从基础计算机科学角度对 Git 进行的实用介绍，解释其分布式特性和常见工作流。
- [Version Control with Git: Powerful tools and techniques for coordinating software development](https://www.oreilly.com/library/view/version-control-with/9781492091189/) — Prem Kumar Ponuthorai, Jon Loeliger (2022)
  Publisher: O'Reilly Media
  一本广受好评的 Git 深入书籍，涵盖其分布式架构和使用模式。

---

[上一节](03-%E9%9B%86%E4%B8%AD%E5%BC%8F%E4%B8%8E%E5%88%86%E5%B8%83%E5%BC%8F%E7%89%88%E6%9C%AC%E6%8E%A7%E5%88%B6%E7%B3%BB%E7%BB%9F%E5%AF%B9%E6%AF%94.md) · [下一节](05-Git%E6%A0%B8%E5%BF%83%E8%A6%81%E7%82%B9%EF%BC%9A%E4%BB%93%E5%BA%93%E3%80%81%E6%8F%90%E4%BA%A4%E3%80%81%E5%88%86%E6%94%AF.md)
