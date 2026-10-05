# 在 Windows 上安装 Git

来源：[原文](https://apxml.com/zh/courses/getting-started-with-git/chapter-1-intro-version-control-git/install-git-windows)

[返回章节目录](README.md) · [返回课程目录](../README.md)

现在您已了解版本控制的基础和 Git 的分布式方式，下一步就是将 Git 安装到您的计算机上。如果您使用 Windows 电脑，本节将为您说明安装步骤。

在 Windows 上安装 Git 最常见且推荐的方法是使用官方的“Git for Windows”软件包。此软件包包含 Git 本身，以及 Git Bash（一个提供类似 Linux 体验的命令行环境）和 Git GUI（一个图形用户界面替代方案）等实用工具。

### 下载安装程序

1. 打开您的网页浏览器，访问 Git 官方网站：<https://git-scm.com/>
2. 在首页上，您会看到一个显眼的下载按钮，通常标明适用于 Windows。点击此按钮。
3. 另外，您可以直接前往下载页面 (<https://git-scm.com/download/win>)，它会自动开始下载最新稳定版安装程序。
4. 将下载的 `.exe` 文件保存到易于查找的位置，例如您的“下载”文件夹。

### 运行安装程序

下载完成后，找到安装文件（例如 `Git-2.xx.x-64-bit.exe`），双击它以开始安装过程。您会看到 GNU 通用公共许可证。如果愿意，可以阅读它，然后点击“下一步”。

您将依次通过多个配置屏幕。虽然默认选项对于初学者来说通常是合理的，但我们还是来查看一些重要的选项：

1. **选择组件：** 此屏幕允许您选择要安装的组件。

   - 确保勾选“Git Bash”和“Git GUI”。Git Bash 尤其实用，因为许多 Git 教程和文档都使用它的命令。
   - 额外的图标和 Windows 资源管理器集成可能会有帮助，但并非必需。
   - 除非有特殊原因，否则请保留其他默认设置。点击“下一步”。
2. **选择 Git 使用的默认编辑器：** Git 有时需要您输入文本，例如提交消息。此步骤允许您选择 Git 将为这些任务打开的文本编辑器。

   - 默认通常是 Vim，它功能强大但对新手来说可能有难度。
   - 如果已安装，您可以从下拉列表中选择一个更熟悉的编辑器（例如 Notepad++、Visual Studio Code、Sublime Text）。如果您选择 VS Code，安装程序可能需要为您配置它。
   - 如果拿不准，选择记事本（Windows 基本编辑器）是一个简单的起点，尽管它在处理复杂任务时功能较弱。点击“下一步”。
3. **调整新仓库中初始分支的名称：** Git 中默认的分支名称历史上是 `master`。新的趋势倾向于 `main`。安装程序允许您选择。

   - 选项 1：“让 Git 决定”（目前为向后兼容性，默认为 `master`）。
   - 选项 2：“覆盖默认分支名称...” 您可以在此处输入 `main` 或其他名称。使用 `main` 正成为标准，特别是在 GitHub 等平台上。选择您偏好的选项，如果拿不准，则坚持使用默认选项。点击“下一步”。
4. **调整您的 PATH 环境变量：** 这是一个重要的设置，它决定了您如何运行 Git 命令。

   - **仅从 Git Bash 使用 Git：** 如果您担心冲突，这是最安全的选项，但这表示您只能在 Git Bash 窗口中使用 Git。
   - **从命令行和第三方软件使用 Git（推荐）：** 此选项将 Git 添加到您的系统 PATH 中，允许您直接从标准 Windows 命令提示符 (cmd.exe) 或 PowerShell 运行 Git 命令，以及从 Git Bash 和其他工具（如 VS Code）中使用。这通常是最方便的选择。
   - **从命令提示符使用 Git 和可选的 Unix 工具：** 这会将 Git 和一些 Unix 实用程序添加到 PATH 中，可能会覆盖一些 Windows 工具。除非您知道需要它，否则请避免使用此选项。
   - 选择推荐的中间选项，然后点击“下一步”。
5. **选择 HTTPS 传输后端：** 保持默认的“使用 OpenSSL 库”选项。点击“下一步”。
6. **配置行结尾转换：** 此设置处理 Git 如何处理行结尾，Windows (CRLF) 和 Linux/macOS (LF) 之间有所不同。

   - **检出时使用 Windows 风格，提交时使用 Unix 风格行结尾（推荐）：** Git 在检出代码时将 LF 结尾转换为 CRLF，并在提交时将 CRLF 转换回 LF。这是 Windows 开发的标准设置，尤其适用于跨平台团队。
   - 选择推荐的第一个选项，然后点击“下一步”。
7. **配置与 Git Bash 搭配使用的终端模拟器：**

   - **使用 MinTTY (MSYS2 的默认终端)（推荐）：** MinTTY 提供比标准 Windows 控制台更好的终端体验（可调整大小的窗口、Unicode 字体等）。
   - **使用 Windows 的默认控制台窗口：** 使用标准的 `cmd.exe` 窗口。
   - 选择推荐的 MinTTY 选项，然后点击“下一步”。
8. **选择 `git pull` 的默认行为：** 此项配置 `git pull` 如何集成拉取的更改。默认（快进或合并）适用于初学者。我们稍后会讨论 `pull`、`fetch` 和 `rebase`。保持默认选项，然后点击“下一步”。
9. **选择凭证助手：** Git 在与远程仓库交互时需要进行身份验证。

   - **Git Credential Manager Core (默认)：** 此功能可安全处理 GitHub 和 GitLab 等服务的凭据，通常与 Windows 凭据存储集成。这是推荐的选项。
   - 点击“下一步”。
10. **配置额外选项：**

    - **启用文件系统缓存：** 提高性能。保持启用状态。
    - **启用符号链接：** 符号链接在 Windows 上不那么常见，但如果需要可以启用。默认（禁用）通常即可。
    - 点击“下一步”。
11. **配置实验性选项：** 这些选项用于测试新功能，除非有特殊原因，否则通常最好保持禁用。点击“安装”。

Git 现在将安装到您的系统。安装完成后，您可以选择查看发布说明或启动 Git Bash，然后再点击“完成”。

### 验证安装

要确认 Git 已正确安装，您可以打开 Git Bash（如果您安装了它）或标准的 Windows 命令提示符/PowerShell，然后输入以下命令：

```bash
git --version
```

按 Enter 键。如果 Git 已正确安装并添加到您的 PATH 中，您应该会看到类似以下内容的输出（确切的版本号会有所不同）：

```
git version 2.45.1.windows.1
```

如果您看到版本号，恭喜您！Git 已安装并可以在您的 Windows 系统上进行配置。如果您收到错误消息，例如“'git' 不是内部或外部命令...”，请在安装过程中仔细检查“调整您的 PATH 环境变量”步骤，或尝试重新启动您的命令提示符/终端。

## 参考资料

- [Pro Git Book, Version 2 (online): Getting Started - Installing Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) — Scott Chacon and Ben Straub (2024)
  Publisher: git-scm.com
  官方的Git安装指南，包含详细的跨平台安装步骤和Windows特有的考量，如Git for Windows包组件。

---

[上一节](05-Git%E6%A0%B8%E5%BF%83%E8%A6%81%E7%82%B9%EF%BC%9A%E4%BB%93%E5%BA%93%E3%80%81%E6%8F%90%E4%BA%A4%E3%80%81%E5%88%86%E6%94%AF.md) · [下一节](07-%E5%9C%A8%20macOS%20%E4%B8%8A%E5%AE%89%E8%A3%85%20Git.md)
