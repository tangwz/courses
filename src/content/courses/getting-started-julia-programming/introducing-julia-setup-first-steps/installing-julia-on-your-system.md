---
course: "getting-started-julia-programming"
chapter: "introducing-julia-setup-first-steps"
lesson: "installing-julia-on-your-system"
sourceId: 6760
sourceUrl: "https://apxml.com/zh/courses/getting-started-julia-programming/chapter-1-introducing-julia-setup-first-steps/installing-julia-on-your-system"
title: "在您的系统上安装 Julia"
description: "在 Windows、macOS 和 Linux 操作系统上下载和安装 Julia 的分步说明。"
order: 4
plots: []
sourceHash: "9187f97184bda8659e37511859a3b2d3fe7ec0aba570f71b0e983721ba1b8604"
sourceCorrections: []
---

在您的电脑上安装 Julia 是您编程之路的第一步。Julia 社区已使此过程相当容易，您有几种主要方式进行设置：使用官方的 Julia 版本管理器 `juliaup`，或者直接从 Julia 语言网站下载安装程序或压缩包。对于大多数用户，特别是 Julia 新手，`juliaup` 是建议的方式，因为它不仅简化了最初的安装，也使管理不同 Julia 版本和更新它们变得更简单。

### 推荐方法：使用 `juliaup` 安装 Julia

`juliaup` 是一个命令行工具，作为一个安装器和版本管理器。您可以将其视为一个有用的助手，为您获取 Julia，如果您日后需要，它甚至可以在同一台机器上管理多个 Julia 版本。它还确保 Julia 正确添加到您系统的 PATH 中，这样您就可以从任何终端窗口运行 Julia。

**针对 Windows 用户：**

在 Windows 10（版本 1903 或更高）或 Windows 11 上获取 `juliaup` 的最简单方法是通过 `winget`（Windows 包管理器）或直接从 Microsoft Store。

1. **使用 `winget`（推荐给命令行用户）：**
   打开 PowerShell 或命令提示符并输入：

   ```sh
   winget install julia -s msstore
   ```

   此命令将从 Microsoft Store 下载并安装 `juliaup`。
2. **使用 Microsoft Store：**

   - 在您的 Windows 电脑上打开 Microsoft Store 应用。
   - 搜索“Julia”。
   - 找到官方的 Julia 应用（它可能由 JuliaLang 组织管理，或在其描述中显示 `juliaup`），然后点击“安装”或“获取”。

`juliaup` 安装后，通常会默认安装 Julia 的最新稳定版。如果未安装，或者您想确认，您可以打开一个新的终端（PowerShell 或命令提示符）并输入：

```sh
juliaup add release
```

此命令指示 `juliaup` 安装 Julia 的最新稳定版本。

**针对 macOS 和 Linux 用户：**

在 macOS 和 Linux 上，您可以通过在终端中运行一个命令来安装 `juliaup`。

1. 打开您的终端应用。
2. 执行以下命令：

   ```sh
   curl -fsSL https://install.julialang.org | sh
   ```

   此脚本将下载并安装 `juliaup`。它可能会请求您的许可来修改您的 shell 配置文件（例如 `.bashrc`、`.zshrc`、`.profile`），以将 `juliaup` 添加到您的 PATH 中。通常允许这样做是安全的。
3. 脚本完成后，您可能需要重新启动终端或来源（source）您的 shell 配置文件（例如 `source ~/.bashrc` 或 `source ~/.zshrc`），以使更改生效。
4. `juliaup` 安装后，通常会安装 Julia 的最新稳定版本。如果未安装，或为确保您已安装，请运行：

   ```sh
   juliaup add release
   ```

使用 `juliaup` 可以让您以后轻松更新 Julia。您只需运行 `juliaup update`。

### 备选方法：从 julialang.org 手动安装

如果您不想使用版本管理器，或者 `juliaup` 不适合您的环境（例如，离线系统或 `juliaup` 不易管理的特定旧版本），您可以直接下载 Julia。

1. **访问下载页面：**
   打开您的网页浏览器，访问 Julia 官方网站：<https://julialang.org>。查找“下载”部分。
2. **选择您的版本：**
   您将看到不同 Julia 版本的选项：

   - **当前稳定版：** 这通常是大多数用户和本课程推荐的版本。它包含经过充分测试的最新功能和错误修复。
   - **长期支持 (LTS) 版：** 此版本会在更长时间内获得错误修复，但不会那么快地获得新功能。在生产环境中，它通常因稳定性而更受欢迎。
   - **即将发布的版本（beta/RC）或每夜构建版：** 这些是为希望试用最新、未发布功能的开发者和测试者准备的。如果您刚开始学习，请避免使用这些版本。

   对于本课程，请下载适合您操作系统的**当前稳定版**。

**手动安装的平台特定说明：**

- **Windows：**

  1. 下载安装文件，其扩展名为 `.exe`（例如，64 位系统为 `julia-1.X.Y-win64.exe`）。大多数现代计算机都是 64 位的。
  2. 运行下载的安装程序。您可能需要管理员权限。
  3. 按照屏幕上的指示操作。一个重要选项通常是“将 Julia 添加到 PATH”。**确保此复选框已选中。** 这将使您能够轻松地从命令提示符运行 Julia。
     > 如果您看到此选项，选中它将省去您之后的一个手动配置步骤。
- **macOS：**

  1. 下载磁盘映像文件，其扩展名为 `.dmg`（例如，`julia-1.X.Y-mac64.dmg`）。
  2. 双击 `.dmg` 文件以挂载它。将打开一个窗口，显示 Julia 应用程序包（例如 `Julia-1.X.app`）。
  3. 将 `Julia-1.X.app` 包拖到您的 `Applications` 文件夹中。
  4. 要从终端运行 Julia，您需要将其添加到系统的 PATH 中。如果安装程序未处理此项，我们将在“验证您的安装和 PATH”部分介绍如何执行此操作。一些 macOS 包可能会自动创建符号链接；如果 `julia` 在终端中立即工作，则表示您已设置完成。
- **Linux（通用二进制文件）：**

  1. 下载通用的 Linux tarball，其扩展名为 `.tar.gz`（例如，64 位系统为 `julia-1.X.Y-linux-x86_64.tar.gz`）。
  2. 打开您的终端并导航到您下载文件的目录（例如，`cd ~/Downloads`）。
  3. 解压档案。放置 Julia 的常见位置是 `/opt`（用于系统范围访问，需要 `sudo`）或直接在您的主目录中（例如 `~/julia`）。
     例如，要解压到您的主目录：

     ```sh
     mkdir ~/julia-1.X.Y # 或者您下载的任何版本
     tar -xvzf julia-1.X.Y-linux-x86_64.tar.gz -C ~/julia-1.X.Y --strip-components=1
     ```

     将 `1.X.Y` 替换为实际的版本号。`--strip-components=1` 部分会从档案中移除顶层目录。
  4. 然后，您需要将 Julia 的 `bin` 目录添加到您的系统 PATH 中。例如，如果您将 Julia 解压到 `~/julia-1.X.Y`，要添加的目录将是 `~/julia-1.X.Y/bin`。

### 验证您的安装

使用 `juliaup` 或手动方法安装 Julia 后，现在是时候检查一切是否正常运行了。

1. **打开您的终端或命令提示符：**

   - **Windows：** 搜索“命令提示符”、“PowerShell”或“Windows 终端”。
   - **macOS：** 打开“终端”（您可以在应用程序 > 实用工具中找到它，或通过 Spotlight 搜索）。
   - **Linux：** 打开您喜欢的终端模拟器（例如 GNOME Terminal、Konsole、xterm）。
2. **输入 Julia 命令：**
   在终端窗口中，输入 `julia` 并按回车键：

   ```sh
   julia
   ```
3. **检查输出：**
   如果 Julia 已正确安装并添加到您的 PATH 中，您应该会看到 Julia REPL（Read-Eval-Print Loop，读取-求值-输出循环）启动。它将显示一个带有 Julia 标志、版本信息和 `julia>` 提示符的横幅：

   ```
                  _
      _       _ _(_)_     |  文档： https://docs.julialang.org
     (_)     | (_) (_)    |
      _ _   _| |_  __ _   |  输入 "?" 获取帮助, "]?" 获取 Pkg 帮助。
     | | | | | | |/ _` |  |
     | | |_| | | | (_| |  |  版本 1.X.Y (YYYY-MM-DD)
    _/ |\__'_|_|_|\__'_|  |  官方 https://julialang.org/ 发布版
   |__/                   |

   julia>
   ```

   （版本号 `1.X.Y` 和日期将与您安装的版本相符。）

   如果您看到此内容，恭喜！Julia 已准备就绪。您可以输入 `exit()` 或按下 `Ctrl+D`（在某些终端的 macOS 上是 Cmd+D）以退出 Julia REPL 并返回到您的系统 shell。

   如果您收到类似“julia: command not found”或“julia 不是内部或外部命令”的错误消息，则表示您系统的 PATH 变量不包含 Julia 的安装目录。这在您选择手动安装且未选中“添加到 PATH”选项，或者 `juliaup` 对您的 shell 配置文件的修改尚未生效时（尝试重启您的终端）很常见。

### 关于系统 PATH 的简要说明（如果需要）

PATH 是您操作系统上的一个环境变量，它告诉 shell 在哪里查找可执行文件。如果 Julia 的安装目录不在 PATH 中，当您输入 `julia` 时，您的 shell 将不知道在哪里找到 `julia` 程序。

- **如果您使用了 `juliaup`：** `juliaup` 通常会自动处理 PATH 配置。如果找不到 `julia`，请尝试重启您的终端甚至计算机。如果仍然无效，请查阅 `juliaup` 文档以解决 PATH 问题。
- **如果您是手动安装：**

  - **Windows：** 如果您在安装过程中没有选中“将 Julia 添加到 PATH”，则需要手动操作。在 Windows 搜索栏中搜索“环境变量”，点击“编辑系统环境变量”。在“系统属性”窗口中，点击“环境变量...”。在“系统变量”（或仅针对您的帐户的“用户变量”）下，找到 `Path` 变量，选中它，然后点击“编辑...”。点击“新建”并添加 Julia 的 `bin` 目录的完整路径（例如 `C:\Users\YourUser\AppData\Local\Programs\Julia-1.X.Y\bin` 或其安装位置）。在所有窗口中点击“确定”。您需要打开一个新的命令提示符以使更改生效。
  - **macOS：** 如果您将 `Julia-1.X.app` 拖到 `/Applications`，可执行文件位于应用程序包内，路径为 `/Applications/Julia-1.X.app/Contents/Resources/julia/bin/julia`。为了使其可访问，您可以创建符号链接或将此目录添加到您的 PATH 中。一种常见方法是编辑您 shell 的配置文件（例如，新版 macOS 默认的 Zsh 的 `~/.zshrc`，或 Bash 的 `~/.bash_profile` 或 `~/.bashrc`）。添加类似以下的一行：

    ```sh
    export PATH="/Applications/Julia-1.X.app/Contents/Resources/julia/bin:$PATH"
    ```

    将 `1.X` 替换为您的 Julia 版本。保存文件，然后打开新的终端或运行 `source ~/.zshrc`（或相应的 G文件）。
  - **Linux：** 如果您将 Julia 解压到例如 `~/julia-1.X.Y`，其 `bin` 目录是 `~/julia-1.X.Y/bin`。编辑您 shell 的配置文件（例如 `~/.bashrc` 或 `~/.zshrc`）。添加类似以下的一行：

    ```sh
    export PATH="$HOME/julia-1.X.Y/bin:$PATH"
    ```

    （使用 `$HOME` 是您主目录的快捷方式）。将 `julia-1.X.Y` 替换为您的实际 Julia 目录名称。保存文件，然后打开新的终端或运行 `source ~/.bashrc`（或相应的 G文件）。
  > 对新手来说，将目录添加到 PATH 有时可能有点难办。如果您遇到问题，在线搜索“add to PATH [您的操作系统名称]”会提供很多详细的指南。使用 `juliaup` 可以避免这些手动步骤中的大部分。

### 保持 Julia 更新

使用 `juliaup` 的优点之一是简化了更新流程。

- **使用 `juliaup`：** 要更新到当前 Julia 版本的最新补丁版本，或更新 `juliaup` 本身以及您正在跟踪的任何通道（例如 `release` 通道），只需打开您的终端并运行：

  ```sh
  juliaup update
  ```
- **手动安装：** 如果您是手动安装 Julia，通常需要从 Julia 网站重复新版本的下载和安装过程。如果您愿意，通常可以将新版本与旧版本并存安装，或者替换旧版本。

成功安装 Julia 后，您现在可以开始使用该语言了。在接下来的部分中，我们将了解 Julia REPL 并编写我们的第一个简单 Julia 程序。

## 参考资料

- [Julia Documentation: Installing Julia](https://docs.julialang.org/en/v1/manual/getting-started/#Installing-Julia) — The Julia Language Community (2025)
  这份官方综合指南详细介绍了Julia的安装方法，涵盖了不同平台和途径。
- [juliaup - The Julia Installer and Version Manager](https://github.com/JuliaLang/juliaup) — The JuliaLang Organization (2024)
  Publisher: The JuliaLang Organization
  这是推荐的Julia版本管理器juliaup的官方文档，详细说明了其用法、功能和安装说明。
- [The Julia Language: Downloads](https://julialang.org/downloads/) — The Julia Language Community (2024)
  Publisher: The Julia Language Community
  Julia的官方下载页面，提供了适用于各种操作系统和Julia版本（稳定版、LTS版等）的安装程序和归档文件的直接链接。
