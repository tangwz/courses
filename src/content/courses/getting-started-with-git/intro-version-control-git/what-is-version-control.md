---
course: "getting-started-with-git"
chapter: "intro-version-control-git"
lesson: "what-is-version-control"
sourceId: 985
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-git/chapter-1-intro-version-control-git/what-is-version-control"
title: "什么是版本控制？"
description: "了解版本控制的基本思想，以及它为何对软件开发及其他项目很重要。"
order: 1
plots: []
sourceHash: "05e41e9ad685e1d3410c917e5962dca2db0f7307b109edd1d49574ace6decbe1"
sourceCorrections: []
---

设想你正在进行一个项目，也许是为新应用编写代码，创建网站，甚至是起草研究论文。随着项目进行，你会做些修改。你会添加新功能，修复错误，重写段落，或更新配置。

当你需要回到之前的版本时，会发生什么？也许你刚做的改动引入了意想不到的问题，或者你只是想比较当前状态和上周的样子。如果没有一个系统，这可能会一团糟。你可能会保存多个文件副本，命名类似 `project_v1.py`、`project_v2_fixed.py`、`project_final.py`、`project_final_really_final.py`。这种方式很快就会变得难以管理，特别是当多人共同处理一个项目时。你如何合并不同协作者的改动而不覆盖彼此的工作？你如何知道 *why* 做出某项改动的原因？

这就是 **版本控制** 要解决的问题。

本质上，版本控制系统（VCS）是一种软件，可以帮助你管理文件随时间发生的变动。可以把它看作是你项目的细致档案管理员。它记录了对文件所做的每一次修改，生成详细的历史记录。这份历史记录能让你做到：

1. **恢复文件：** 如果出现错误或需求改变，可以轻松回溯到单个文件或整个项目的早期版本。
2. **追踪修改：** 清楚地查看谁修改了什么，何时修改，以及（理想情况下，通过好的提交信息）修改的原因。
3. **了解差异：** 比较文件的不同版本，精确地查看它们之间发生了哪些修改。
4. **高效协作：** 允许多人同时处理同一个项目，而不会相互干扰进度。VCS 提供将这些不同贡献合并的机制。
5. **安全试验：** 为你的项目创建隔离副本（通常称为分支），以尝试新想法或开发功能，而不影响主版本或稳定版本。如果试验成功，改动可以合并回去；如果失败，该分支可以丢弃而没有负面影响。

你可以把 VCS 看作文本编辑器中的“撤销”功能，但它远比那强大得多。VCS 不仅仅是撤销上一个操作，它让你能够浏览项目的整个历史，查看在不同时间点所做的快照（这些快照通常称为“提交”或“修订版”）。

VCS 将这段历史存储在一个特殊的数据库中，通常位于你的项目目录内，常被称为 **仓库**。这个仓库包含了重现你项目任何先前状态所需的所有信息。

尽管通常与软件开发相关联，版本控制对于管理任何随时间变化的数字文件集合都有益处，比如文档、配置文件、设计素材、数据集等等。

掌握版本控制是现代开发实践和高效协作的根本。Git 是本课程的重点，它是一种特定、功能强且被广泛使用的版本控制系统，以独特而高效的方式实现了这些思想，我们将在后续章节中对此进行讲述。

## 参考资料

- [Pro Git](https://git-scm.com/book/en/v2/Getting-Started-About-Version-Control) — Scott Chacon, Ben Straub (2014)
  Publisher: Apress; Pages: 1.1 About Version Control
  这本基础书籍对 Git 进行了全面介绍，其初始章节清晰地解释了核心版本控制概念。
- [What is version control?](https://www.atlassian.com/git/tutorials/what-is-version-control) — Atlassian (2023)
  一份来自受尊敬行业领导者的实用在线指南，以易于理解的方式解释了版本控制系统的基础知识和优势。
