---
course: "prompt-engineering-llm-application-development"
chapter: "application-development-considerations"
lesson: "managing-api-keys-secrets"
sourceId: 3090
sourceUrl: "https://apxml.com/zh/courses/prompt-engineering-llm-application-development/chapter-8-application-development-considerations/managing-api-keys-secrets"
title: "管理 API 密钥和机密信息"
description: "在应用程序中处理敏感凭证的安全方法。"
order: 2
plots: []
sourceHash: "9d9107eefb70cd70f7f263d534001518ea567213145591b8af1a17bdb781206f"
sourceCorrections: []
---

与大型语言模型 API 交互，不可避免地要处理敏感凭证，主要是 API 密钥。这些密钥提供对强大且可能费用高昂的资源的访问权限。不当管理它们会带来重大的安全风险，并可能导致意外费用或未经授权使用您的 LLM 提供商账户。此处提供在应用程序中安全处理 API 密钥及其他机密信息的实用方法。

对待 API 密钥要像对待密码一样。如果它们落入不法分子之手，其他人可能会使用您的账户，这可能导致高额费用或访问您原本打算保密的功能。最常见的错误是将密钥直接硬编码到源代码中。

```python
# 不良做法：硬编码 API 密钥！
# 在实际应用程序中切勿这样做。
import openai

# 这在代码中直接暴露。
openai.api_key = "sk-this_is_a_fake_key_replace_me_immediately_abc123"

# ... 应用程序的其余逻辑
```

以这种方式存储密钥风险极高。如果您的代码被共享、发布到公共仓库（如 GitHub），甚至被不应拥有管理权限的人访问，您的密钥就会泄露。Git 等版本控制系统会追踪每一次更改，因此即使您稍后移除密钥，它仍可能存在于仓库的历史记录中。

### 安全管理方法

幸运的是，有几种成熟的方法可以安全地处理机密信息。最佳方法通常取决于您的开发工作流程、部署环境和团队规模。

#### 1. 环境变量

使用环境变量是管理机密信息最常用、最直接的方法之一。环境变量是存储在应用程序代码之外的键值对，由操作系统或执行环境（如 Docker 容器或云平台服务）管理。

您的应用程序代码在运行时从环境中读取 API 密钥。

```python
import os
import openai
from dotenv import load_dotenv

# 从 .env 文件加载环境变量（可选，适合本地开发）
load_dotenv()

# 从环境变量获取 API 密钥
api_key = os.environ.get("OPENAI_API_KEY")

if not api_key:
    print("Error: OPENAI_API_KEY environment variable not set.")
    # 适当处理错误，例如退出或抛出异常
else:
    openai.api_key = api_key
    # 继续使用 API
    print("API key loaded successfully.")
    # response = openai.Completion.create(...) # 示例用法
```

**使用 `.env` 文件进行本地开发：**

对于本地开发，将环境变量存储在项目根目录下的 `.env` 文件中很方便。此文件*绝不能*提交到版本控制系统。立即将 `.env` 添加到您的 `.gitignore` 文件中。

```
# .env 文件（放置在项目根目录，并添加到 .gitignore）
OPENAI_API_KEY="sk-your_actual_secret_key_for_development"
ANTHROPIC_API_KEY="sk-ant-your_other_secret_key"
```

您可以使用 `python-dotenv` 等库（通过 `pip install python-dotenv` 安装），以便在应用程序启动时自动将此文件中的变量加载到环境中，如上面的 Python 示例所示。

**部署：**

在部署环境（如云服务器、容器、无服务器函数）中，您通常通过平台的配置界面或部署脚本来设置环境变量。例如：

- **Docker：** 在 `docker run` 或 `docker-compose.yml` 中使用 `-e` 标志或 `env_file` 选项。
- **Kubernetes：** 使用 Secret 或 ConfigMap，通常作为环境变量挂载。
- **云平台（AWS、Azure、GCP）：** Elastic Beanstalk、App Service 或 Cloud Functions 等服务都有专门的部分，用于为部署的应用程序设置环境变量。

**优点：**

- 易于实施，并且在各种语言和平台中得到广泛支持。
- 使机密信息脱离代码库。

**缺点：**

- 在多个环境（开发、测试、生产）中管理变量可能会变得繁琐。
- 如果处理不当，变量可能会在进程列表或日志中显示。

#### 2. 配置文件（安全处理）

您可以将配置（包括 API 密钥）存储在文件中（例如 YAML、JSON、TOML）。但是，**包含实际机密信息的配置文件绝不能提交到版本控制系统**。

常见的做法是：

1. 在版本控制中保留一个模板配置文件（例如 `config.template.yaml`）。
2. 拥有一个单独的本地配置文件（例如 `config.local.yaml`），其中包含实际的机密信息，并且该文件已列入 `.gitignore`。
3. 您的应用程序从两者加载配置，如果本地文件存在，则优先使用它。
4. 在部署时，机密文件会安全地注入到环境中（例如，在构建/部署过程中从安全位置复制）。

这种方法通常与使用环境变量重叠，部署过程可能会根据环境变量或从专用服务获取的机密信息来填充 `config.local.yaml` 文件。

#### 3. 专用机密管理服务

对于需要更严格安全控制的复杂应用程序或组织，专用机密管理服务是更受欢迎的解决方案。例如：

- AWS Secrets Manager
- Google Secret Manager
- Azure Key Vault
- HashiCorp Vault（可自行部署或基于云）

这些服务提供集中式的、加密的机密信息存储，并具有以下功能：

- **精细访问控制：** 精确定义哪些用户或应用程序可以访问特定机密信息。
- **审计：** 追踪谁在何时访问了机密信息。
- **自动轮换：** 配置密钥定期自动轮换，减少密钥泄露后的风险时间。
- **集成：** 提供 SDK 和集成，方便在应用程序代码中检索。

检索机密信息通常涉及验证您的应用程序（例如，使用 AWS 上的 IAM 角色或 GCP 上的服务账户），然后向机密管理服务发起 API 调用。

```python
# 使用机密管理器 SDK 的示例
# （语法将根据具体服务和 SDK 而异）

# 假设 'secrets_client' 已正确初始化并完成身份验证
try:
    # 根据其标识符获取机密值
    secret_response = secrets_client.get_secret_value(SecretId="prod/openai/api_key")
    api_key = secret_response['SecretString'] # 或 'SecretBinary'，取决于存储方式

    # 使用检索到的 api_key
    # openai.api_key = api_key
    print("API key retrieved successfully from secrets manager.")

except Exception as e:
    print(f"Error retrieving secret: {e}")
    # 处理错误
```

**优点：**

- 最高级别的安全性、控制性和可审计性。
- 非常适合大型团队和复杂部署的扩展需求。
- 轮换等功能增强了安全状况。

**缺点：**

- 引入额外的基础设施组件和复杂性。
- 可能产生与机密管理服务相关的费用。
- 需要理解特定服务的认证和授权模型。

### 最佳实践清单

无论选择哪种方法，请遵循以下实践：

- **切勿硬编码机密信息：** 这是黄金法则。将机密信息从您的源代码和版本控制历史中移除。
- **使用 `.gitignore`：** 确保包含机密信息的文件（如 `.env` 或本地配置文件）绝不会意外提交。在创建文件*之前*将其添加到 `.gitignore` 中。
- **最小权限原则：** 生成 API 密钥时，仅赋予应用程序功能所需的最低权限。避免使用根账户密钥。
- **定期轮换密钥：** 定期更改您的 API 密钥。机密管理服务通常可以自动化此过程。如果手动操作，请制定一个时间表。
- **环境专用密钥：** 为您的开发、测试和生产环境使用不同的 API 密钥。这限制了低级环境密钥泄露时的影响范围。
- **监控使用情况并审计访问：** 定期检查您的 LLM 提供商仪表板，以发现意外的使用模式。如果使用机密管理服务，请利用其审计功能。

选择正确的方法取决于您的项目规模和安全需求。环境变量对于小型项目或简单部署通常已足够，而机密管理服务则为生产系统和大型团队提供解决方案。从一开始就实施安全的 API 密钥处理实践，对于构建可靠且安全的 LLM 应用程序不可或缺。

## 参考资料

- [HashiCorp Vault: Introduction to Vault](https://developer.hashicorp.com/vault/docs/concepts) — HashiCorp (2024)
  Publisher: HashiCorp
  解释了领先的秘密管理解决方案的基本概念和架构。
- [What is AWS Secrets Manager?](https://docs.aws.amazon.com/secretsmanager/latest/userguide/intro.html) — Amazon Web Services (2024)
  详细介绍了主要云秘密管理服务的特性和使用方法的官方指南。
- [Secrets Management Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html) — OWASP (2023)
  Publisher: The OWASP Foundation
  为应用程序中的安全秘密管理提供了全面的最佳实践和建议。
