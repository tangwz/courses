---
course: "getting-started-with-git"
chapter: "intro-version-control-git"
lesson: "getting-help-with-git"
sourceId: 994
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-git/chapter-1-intro-version-control-git/getting-help-with-git"
title: "获取Git命令的帮助"
description: "了解如何使用 Git 的内置帮助系统，以获取关于特定命令和选项的更多信息。"
order: 10
plots: []
sourceHash: "8a22058c2389f9c5229f619bbce8941c6176501a3f1ec1abc9313a5908faa3fb"
sourceCorrections: []
---

当你开始使用 Git 时，会遇到各种命令，每个命令都有自己的选项和行为。记住所有细节是不必要的，因为 Git 提供了一个完善的内置帮助系统。学会如何有效使用此帮助，是熟练掌握 Git 的一个重要起步。

### 查看完整的命令文档

了解特定 Git 命令最直接的方式是使用 `git help` 命令。只需输入 `git help`，然后是你想了解的命令名称。

例如，如果你想了解用于初始设置的 `config` 命令，可以运行：

```bash
git help config
```

此命令通常会在你系统的默认查看器中打开 `git config` 的官方文档。在类 Unix 系统（Linux、macOS）上，这通常是 `man`（手册）页查看器。在 Windows 上，它可能会在网页浏览器中打开。这些手册页提供了关于命令的用途、概要、可用选项、配置变量和使用示例的详细信息。

另一种语法也能达到相同的效果：

```bash
git config --help
```

`git help <command>` 和 `git <command> --help` 都提供了特定命令最详细的可用文档。当你需要弄清楚命令的作用或可用选项时，请务必使用这些功能。

### 获取简要概览

有时，完整的手册页提供的细节可能超过你的需要。如果你只是想快速回顾命令最常用的选项，许多 Git 命令支持 `-h` 标志。

例如，要查看 `commit` 命令选项的简洁概览：

```bash
git commit -h
```

这通常会直接在你的终端上打印一条较短的帮助信息，列出命令的语法及其主要选项。请注意，并非每个 Git 命令都支持 `-h` 选项，但它适用于许多常用命令。

### 列出可用命令

如果你不确定确切的命令名称，或者只是想看看有哪些 Git 命令可用，可以不指定命令，直接使用 `git help`：

```bash
git help
```

或者，类似地：

```bash
git --help
```

运行这些命令中的任何一个，都会显示最常用的 Git 命令列表，通常按功能分组（例如，开始一个工作区、处理当前更改、检查历史记录）。当你试图为一项任务找到合适的工具时，这可以是一个有用的起点。

要获取 *所有* 可用 Git 命令的真正详尽列表，包括不那么常用的，你可以使用：

```bash
git help --all
```

### 内置指导

在使用过程中，请注意 Git 的输出。当你犯错或有更合适的下一步操作时，Git 通常会提供帮助。例如，如果你拼错了命令，Git 可能会建议正确的拼写。当你运行 `git status` 时，它通常会提供关于你可能想运行的后续命令的提示，例如 `git add` 或 `git commit`。这些上下文 (context)提示是学习过程中有价值的一部分。

### 在终端学习

虽然内置帮助对于命令细节非常有用，但请记住，官方在线文档（可在 [git-scm.com](https://git-scm.com) 网站找到）提供了全面的指南、教程和参考资料。像 Stack Overflow 这样的社区论坛也是解决特定问题和故障排除的有用资源。

掌握 `git help` 及其变体的使用方式很基本。它让你能够独立地了解 Git 的功能，并在你需要时精确地找到所需信息。每当遇到不熟悉的命令或选项时，养成查阅帮助系统的习惯。

## 参考资料

- [Git Documentation](https://git-scm.com/doc) — The Git Development Community (2024)
  提供所有 Git 命令的完整、权威文档（手册页），可通过 `git help` 直接访问。
- [Pro Git](https://git-scm.com/book/en/v2) — Scott Chacon and Ben Straub (2014)
  Publisher: Apress
  提供 Git 命令及其选项的广泛解释和实用示例，作为内置帮助的补充。
- [Git Tutorial](https://git-scm.com/docs/gittutorial) — The Git Development Community (2024)
  针对 Git 新用户的官方指南，强调基本命令以及如何访问其帮助文档。
