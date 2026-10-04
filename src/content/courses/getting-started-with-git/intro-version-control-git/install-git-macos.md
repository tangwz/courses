---
course: "getting-started-with-git"
chapter: "intro-version-control-git"
lesson: "install-git-macos"
sourceId: 991
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-git/chapter-1-intro-version-control-git/install-git-macos"
title: "在 macOS 上安装 Git"
description: "分步说明如何在 macOS 上使用 Homebrew 或官方安装程序等多种方法安装 Git。"
order: 7
plots: []
sourceHash: "54d4685fe5760a5334d7c15f24fcd88958b496b815506051acecc3b32cd7f724"
sourceCorrections: []
---

Git 是一种强大的分布式版本控制系统，广泛用于管理项目文件和协作代码。要开始使用 Git，首先需要在您的电脑上安装它。如果您使用的是 macOS 系统，有几种简单的方法可以让 Git 运行起来。通常，如果您已安装 Xcode 或其命令行工具，Git 可能已经存在。

### 检查现有安装

在进行新的安装之前，最好检查一下 Git 是否已在您的 Mac 上可用。打开终端应用程序（您可以在“应用程序”>“实用工具”中找到它，或使用 Spotlight（`Cmd + Space`）搜索它）。输入以下命令并按回车键：

```bash
git --version
```

如果 Git 已安装，您将看到类似于 `git version X.Y.Z` 的输出，显示已安装的版本号。如果您看到此信息，则可以跳过安装步骤，直接前往初始配置部分。

如果 Git 未安装，macOS 可能会提示您安装命令行工具，其中包含 Git。如果您收到类似 `command not found: git` 的消息，则需要使用以下方法之一进行安装。

### 安装方法

在 macOS 上安装 Git 有三种主要方式：

1. **使用 Xcode 命令行工具：** 如果您计划在 Mac 上进行任何软件开发，这通常是最简单的方法。
2. **使用 Homebrew：** macOS 上一个流行的软件包管理器，它使软件的安装和更新变得容易。
3. **使用官方安装程序：** 直接从 Git 网站下载安装程序。

#### 方法一：通过 Xcode 命令行工具安装

Xcode 命令行工具包包含 Git 以及开发者所需的其他必要工具。即使您没有安装完整的 Xcode 应用程序，也可以只安装命令行工具。

1. 打开终端应用程序。
2. 运行以下命令：

   ```bash
   xcode-select --install
   ```
3. 将出现一个对话框，询问您是否要安装这些工具。点击“安装”并同意条款和条件。
4. 这些工具将被下载并安装。这可能需要几分钟，具体取决于您的网络连接。
5. 安装完成后，通过再次在终端中运行 `git --version` 来验证它。

有时，只需在未安装 Git 的终端中尝试运行 `git --version`，就会自动触发安装命令行工具的提示。

#### 方法二：通过 Homebrew 安装

Homebrew 是 macOS 上一个广泛使用的软件包管理器。如果您没有安装 Homebrew，可以按照官方网站（brew.sh）上的说明获取它。通常，这包括在您的终端中运行其主页上提供的一个命令。

安装 Homebrew 后，您可以使用一个简单命令来安装 Git：

1. 打开终端应用程序。
2. 运行以下命令：

   ```bash
   brew install git
   ```
3. Homebrew 将下载并安装最新稳定版的 Git。
4. 通过运行 `git --version` 来验证安装。

使用 Homebrew 的一个优点是它使更新 Git 变得容易。您以后可以通过运行 `brew upgrade git` 来更新 Git。

#### 方法三：通过官方安装程序安装

您总是可以从 Git 网站直接下载最新的官方 macOS Git 安装程序。

1. 前往 Git 官方网站：<https://git-scm.com/download/mac>
2. 网站通常会检测您的操作系统并提供相应的下载链接。点击链接下载最新的安装程序，这将是一个 `.dmg` 文件。
3. 下载完成后，从您的“下载”文件夹中打开 `.dmg` 文件。
4. 在磁盘映像中，您会找到一个 `.pkg` 安装程序文件。双击它以开始安装过程。
5. 按照屏幕上的说明操作。您可能需要同意许可协议并输入您的管理员密码。
6. 安装程序完成后，关闭它。
7. 打开一个新的终端窗口（确保路径更新生效很重要），并通过运行 `git --version` 来验证安装。

### 验证安装

无论您选择哪种方法，最后一步都是确认 Git 已正确安装。打开一个新的终端窗口并执行：

```bash
git --version
```

如果安装成功，该命令将输出已安装的 Git 版本，例如：

```
git version 2.41.0
```

具体的版本号可能不同，但看到此消息确认 Git 已可在您的 macOS 系统上使用。您现在已准备好进行 Git 的初始配置，这包括设置您的用户身份。

## 参考资料

- [Git - Downloads](https://git-scm.com/downloads) — The Git Project (2024)
  Git 在 macOS 上下载和安装的官方页面，详细介绍了所有主要方法，包括 Xcode 命令行工具、Homebrew 和独立安装程序。
- [Pro Git](https://git-scm.com/book/en/v2) — Scott Chacon and Ben Straub (2014)
  Publisher: Apress
  一本全面而权威的 Git 指南，涵盖了基础概念、高级用法，以及初始设置和安装。在线版本会持续更新。
