---
course: "getting-started-with-git"
chapter: "intro-version-control-git"
lesson: "install-git-linux"
sourceId: 992
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-git/chapter-1-intro-version-control-git/install-git-linux"
title: "在 Linux 上安装 Git"
description: "使用软件包管理器在各种 Linux 发行版上安装 Git 的说明。"
order: 8
plots: []
sourceHash: "fb0aab68fb48768b9bf21dcfaeb09e7e18daee852b96af1e422c303e4ad2ebb9"
sourceCorrections: []
---

本节将指导你如何在 Linux 系统上安装 Git。Git 是一种分布式版本控制系统，它提供工具来跟踪项目变更并促进团队协作。对于 Linux 用户而言，安装过程通常很简单，可以通过发行版自带的软件包管理器来完成。

Linux 发行版有很多种，但大多数属于使用特定软件包管理系统的类别。我们将介绍最常见的。

### 使用 APT (Debian、Ubuntu 及衍生版本)

如果你的 Linux 发行版基于 Debian（例如 Ubuntu、Linux Mint、Pop!\_OS 等），它很可能使用高级软件包工具 (APT)。

1. **更新你的软件包列表：** 在安装新软件之前，最好确保你的系统可用软件包列表是最新的。打开你的终端并运行：

   ```bash
   sudo apt update
   ```

   你可能会被要求输入密码。`sudo` 命令临时授予管理系统软件所需的管理员权限。
2. **安装 Git：** 软件包列表更新后，你可以使用以下命令安装 Git：

   ```bash
   sudo apt install git
   ```

   APT 将计算依赖项，要求你确认（通常是输入 `Y`），下载所需文件，并将 Git 安装到你的系统上。

### 使用 DNF 或 YUM (Fedora、CentOS、RHEL 及衍生版本)

对于 Red Hat 家族的发行版（例如 Fedora、CentOS Stream、Rocky Linux、AlmaLinux 或 Red Hat Enterprise Linux），你将使用 `dnf`（较新系统，例如 Fedora、RHEL 8+）或 `yum`（较旧系统，例如 CentOS 7）。这些命令非常相似。

1. **更新你的软件包（可选但建议）：** 你可以先更新你的系统软件包。
   对于 `dnf`：

   ```bash
   sudo dnf update
   ```

   对于 `yum`：

   ```bash
   sudo yum update
   ```

   同样，`sudo` 提供必要的权限，你可能会被要求输入密码。
2. **安装 Git：**
   对于 `dnf`：

   ```bash
   sudo dnf install git
   ```

   对于 `yum`：

   ```bash
   sudo yum install git
   ```

   与 APT 类似，这些软件包管理器将处理依赖项并在安装 Git 之前要求你确认。

### 验证安装

无论使用哪种方法，你都可以通过询问 Git 的版本来验证它是否安装正确。打开终端并输入：

```bash
git --version
```

如果安装成功，你将看到类似于这样的输出（具体的版本号可能不同）：

```
git version 2.40.1
```

看到版本号确认 Git 已安装并可以从命令行访问。

### 从源代码安装（高级）

虽然使用软件包管理器是大多数用户的推荐方法，你也可以直接从 Git 的源代码编译和安装它。这确保你拥有绝对最新的版本，但这是一个更复杂的过程。从源代码构建的说明可在 Git 官方网站上找到 (<https://git-scm.com/>)。对于本入门课程，软件包管理器方法就足够了。

Git 安装完成后，下一步是进行一些初始设置，以将你识别为你未来更改的作者。我们将在下一节讨论这一点。

## 参考资料

- [Installing Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) — The Git Development Community (2024)
  提供在各种操作系统上安装 Git 的官方说明，包括针对 Linux 包管理器和从源代码构建的详细步骤。
- [Pro Git](https://git-scm.com/book/en/v2) — Scott Chacon and Ben Straub (2014)
  Publisher: Apress
  一本完整的参考资料，涵盖 Git 的所有方面，从初始设置和命令到高级工作流程，适合初学者和有经验的用户（当前在线版本）。
