# 初始Git配置 (user.name, user.email)

来源：[原文](https://apxml.com/zh/courses/getting-started-with-git/chapter-1-intro-version-control-git/initial-git-configuration)

[返回章节目录](README.md) · [返回课程目录](../README.md)

在你开始用Git记录项目改动之前，你需要进行一个简单的一次性设置：告诉Git你是谁。每当你保存项目的一个快照（我们很快就会学到，这被称为“提交”），Git会将你的姓名和电子邮件地址包含在这些保存的信息中。这对于跟踪历史和与他人协作非常重要，因为它清楚地标明了每次改动的作者。

可以把它想象成在你的工作上签名。当你回顾项目历史或与团队成员合作时，了解是谁做了具体的修改是很重要的。

幸运的是，使用`git config`命令配置这个身份信息很简单。你通常只需要在每台电脑上做一次。

### 设置你的用户名

首先，我们来告诉Git你的名字。打开你的终端或命令提示符（如果你按照之前的步骤操作，就是你用于安装的那个）。输入以下命令，将`"Your Name"`替换为你自己的名字：

```bash
git config --global user.name "Your Name"
```

按下回车键。如果命令成功运行，你不会看到任何确认消息，这在许多命令行工具中很常见。

### 设置你的电子邮件地址

接下来，配置你的电子邮件地址。使用你希望与你的代码贡献关联的电子邮件地址。这通常是你用于GitHub或GitLab等Git托管平台的电子邮件。将`"youremail@example.com"`替换为你的电子邮件：

```bash
git config --global user.email "youremail@example.com"
```

同样，按下回车键。

### 理解 `--global` 标志

你可能注意到了两个命令中的`--global`选项。这告诉Git将此配置设置应用于此计算机上你当前用户帐户下工作的所有Git仓库。它将你的姓名和电子邮件保存在一个全局配置文件中。这通常是你想要的。

如果你需要为特定项目使用不同的姓名或电子邮件（例如，公司项目使用工作电子邮件，个人项目使用个人电子邮件），你可以省略`--global`标志。要做到这一点，请在终端中使用`cd`命令进入特定项目的目录，然后运行不带`--global`的`git config`命令：

```bash
# 在你的特定项目目录中
git config user.name "Your Work Name"
git config user.email "work.email@company.com"
```

*不带*`--global`配置的设置（本地设置）仅覆盖该特定仓库的全局设置。

### 验证你的配置

你怎么能确定你的设置已正确保存？你可以要求Git显示其配置。要查看所有全局设置，包括你刚刚设置的姓名和电子邮件，请使用：

```bash
git config --list --global
```

或者，你可以检查特定的设置：

```bash
git config user.name
# 预期输出：你的名字

git config user.email
# 预期输出：youremail@example.com
```

如果你没有看到预期的输出，或者你打错了字，只需用正确的信息重新运行相应的`git config --global`命令。它将覆盖之前的值。

配置好你的身份后，Git现在知道你是谁了。你已经可以开始使用Git来管理你的项目了，从创建第一个仓库并进行首次提交开始，这些我们将在下一章介绍。

## 参考资料

- [Pro Git, Chapter 1.6: First-Time Git Setup](https://git-scm.com/book/en/v2/Getting-Started-First-Time-Git-Setup) — Scott Chacon and Ben Straub (2024)
  Publisher: Apress
  一份权威且全面的指南，涵盖 Git 初始配置，包括用户身份、全局与局部设置以及验证步骤。
- [git-config Documentation](https://git-scm.com/docs/git-config) — Git Community (2024)
  `git config` 命令的官方参考手册，详细说明其用法、选项以及 Git 如何在不同级别管理配置。

---

[上一节](08-%E5%9C%A8%20Linux%20%E4%B8%8A%E5%AE%89%E8%A3%85%20Git.md) · [下一节](10-%E8%8E%B7%E5%8F%96Git%E5%91%BD%E4%BB%A4%E7%9A%84%E5%B8%AE%E5%8A%A9.md)
