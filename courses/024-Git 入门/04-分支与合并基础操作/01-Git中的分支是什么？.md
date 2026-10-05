# Git中的分支是什么？

来源：[原文](https://apxml.com/zh/courses/getting-started-with-git/chapter-4-branching-merging-basics/what-are-git-branches)

[返回章节目录](README.md) · [返回课程目录](../README.md)

Git将项目历史记录为一系列快照或提交。通常，这些记录形成一条线性序列，每个提交都基于前一个提交。但是，软件开发很少会严格按照这种线性方式进行。你经常需要同时处理不同的事务。设想你正在开发一个重要的新功能，但突然发现一个紧急错误需要修正已发布给用户的代码。你需要一种方法来暂停功能开发，在稳定的代码库上修正错误，然后返回到功能开发，而不会让这两部分工作互相干扰。

这时，Git分支就发挥作用了。把分支看作不是项目文件的完整副本，而是一个**轻量级可移动指针**，指向某个特定的提交。当你开始工作时，Git会自动创建一个默认分支，通常命名为`main`（在旧项目中可能叫`master`）。这个`main`分支代表主要的开发线，通常跟踪稳定或可用于生产的代码。

每次你进行提交时，这个分支指针都会自动向前移动，指向你刚刚创建的最新提交。所以，一个简单的历史可能看起来像这样：`main`分支指针指向最新提交，而该提交本身又指向上一个父提交，以此类推，构成项目历史。

> 一个简单的Git历史图。提交（C1, C2, C3）形成一个序列。`main`分支指向最新提交（C3），`HEAD`表明`main`是当前活跃的分支。

创建一个新分支意味着创建一个**新指针**，该指针从当前分支所指向的同一提交开始。比如，当你在`main`分支上，且该分支指向提交`C3`时，你创建了一个名为`new-feature`的新分支。最初，`main`和`new-feature`都将指向`C3`。

> 创建`new-feature`分支之后。`main`和`new-feature`这两个指针都引用相同的提交`C3`。`HEAD`仍指向`main`。

现在，如果你切换到`new-feature`分支（我们很快会讲到如何操作）并进行一次新提交，比如`C4`，会发生一些有趣的事情。`new-feature`指针向前移动指向`C4`，但`main`分支指针仍停留在原地，继续指向`C3`。你的`HEAD`指针现在表明你正在`new-feature`分支上处理事务。

> 切换到`new-feature`并进行提交`C4`后。`new-feature`分支指针推进到`C4`，而`main`停留在`C3`。`HEAD`现在指向`new-feature`。

这产生了从提交`C3`分叉的两条独立的开发线。你可以在`new-feature`分支上继续处理你的功能，而不影响`main`分支。如果需要，你可以切换回`main`，创建另一个分支（例如`bug-fix`），在那里进行提交，它不会影响`main`或`new-feature`，直到你明确决定合并它们（这称为合并，稍后会讲到）。

主要结论是，Git分支极其轻量。创建新分支不涉及复制文件或目录；它只是创建一个小文件，其中包含它所指向提交的40个字符的SHA-1哈希值。这种效率使得分支成为Git工作流程中一个常见且重要的部分，不像某些旧版本控制系统，在那些系统中，分支操作成本更高。

使用分支提供了几个重要优点：

1. **隔离：** 处理新功能或错误修正，而不干扰稳定代码库（`main`）。
2. **尝试：** 在单独分支上尝试新想法或重构代码。如果不成功，你可以直接丢弃该分支，而不影响主项目。
3. **并行开发：** 多个开发者（甚至一个开发者处理多个任务）可以同时处理不同的功能或修正。
4. **清晰历史：** 保持`main`分支整洁，只包含稳定、经过测试的代码，而开发发生在其他分支上。

理解这种指针想法是有效使用Git分支能力的重要基础，我们将在后续章节中进一步研究这一点。

## 参考资料

- [Pro Git](https://git-scm.com/book/en/v2/) — Scott Chacon, Ben Straub (2014)
  Publisher: Apress; Pages: Chapter 3: Git Branching - What a Branch Really Is
  Git 的官方综合指南。本章解释了分支作为指向提交的轻量级指针的核心内容。
- [Git Branching](https://www.atlassian.com/git/tutorials/using-branches) — Atlassian (2024)
  广受认可的教程，清晰解释了 Git 分支概念，包括指针类比和常见工作流程。
- [Lecture 5: Version Control (Git)](https://missing.csail.mit.edu/2020/version-control/) — Anish Athalye, Jon Gjengset, and Jose Javier Gonzalez Ortiz (2020)
  Publisher: MIT OpenCourseWare
  作为麻省理工学院备受推崇的课程的一部分，本讲座对 Git 及其分支模型提供了基本解释。

---

[上一节](../03-%E6%9F%A5%E7%9C%8B%E5%8E%86%E5%8F%B2%E8%AE%B0%E5%BD%95%E4%B8%8E%E6%92%A4%E9%94%80%E6%9B%B4%E6%94%B9/09-%E7%BB%83%E4%B9%A0%EF%BC%9A%E6%A3%80%E6%9F%A5%E5%92%8C%E4%BF%AE%E6%94%B9%E5%8E%86%E5%8F%B2.md) · [下一节](02-%E5%88%9B%E5%BB%BA%E6%96%B0%E5%88%86%E6%94%AF%20%28git%20branch%29.md)
