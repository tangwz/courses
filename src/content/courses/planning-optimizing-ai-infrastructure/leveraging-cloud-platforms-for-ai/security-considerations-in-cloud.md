---
course: "planning-optimizing-ai-infrastructure"
chapter: "leveraging-cloud-platforms-for-ai"
lesson: "security-considerations-in-cloud"
sourceId: 7004
sourceUrl: "https://apxml.com/zh/courses/planning-optimizing-ai-infrastructure/chapter-3-leveraging-cloud-platforms-for-ai/security-considerations-in-cloud"
title: "云中的安全考量"
description: "审阅云上AI的安全最佳实践，包括身份与访问管理(IAM)、数据加密和网络安全。"
order: 7
plots: []
sourceHash: "01a8a61a2c29c8ba48a4fb330c61be0dbded62861e3543fa142e625ac2914f2a"
sourceCorrections: []
---

迁移到云平台简化了硬件管理，但也带来了不同的安全职责。与您掌控全部物理和网络堆栈的本地部署不同，在云中，安全是您与提供商之间的协作。这正式称为**共享责任模型**。提供商负责云*本身*的安全（物理数据中心、硬件和核心网络），而您负责云*内部*的安全（您的数据、配置、访问策略和应用代码）。

未能充分履行云环境中的安全职责可能导致数据泄露、对昂贵GPU资源的未授权访问或模型被窃。构建安全的AI环境需要纵深防御策略，在身份、网络和数据层面分层设置控制措施。

### 身份与访问管理 (IAM)

第一道防线是控制谁能做什么。每个主要的云提供商都有身份与访问管理 (IAM) 服务（例如，AWS IAM、Google Cloud IAM、Azure Active Directory）。这里的基本原则是**最小权限原则**：仅授予执行任务所需的权限。

避免将您的根账户或管理员账户用于日常任务。而是创建具有定制策略的特定IAM角色和用户。对于AI工作负载，常见模式是为不同职能创建角色：

- **数据科学家角色**：启动和停止特定训练实例的权限，以及对数据集存储桶的只读访问权。
- **MLOps工程师角色**：管理基础设施、配置部署管道和更新模型的权限。
- **训练服务角色**：分配给虚拟机实例的角色，授予其从对象存储桶读取数据和写入日志的权限。这比直接在训练脚本中嵌入 (embedding)访问密钥要安全得多。

以下是一个简化版的IAM策略（JSON格式），它允许训练实例访问特定的S3存储桶：

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::my-ai-datasets",
                "arn:aws:s3:::my-ai-datasets/*"
            ]
        },
        {
            "Effect": "Allow",
            "Action": "s3:PutObject",
            "Resource": "arn:aws:s3:::my-model-artifacts/*"
        }
    ]
}
```

此策略附加到IAM角色，然后该角色被分配给云实例。运行在实例上的应用程序会自动获得这些权限，无需处理任何密钥。

### 网络隔离与防火墙

您的AI基础设施不应完全暴露在公共互联网上。使用虚拟私有云 (VPC) 服务创建云中逻辑隔离的部分。在VPC中，您可以定义公共子网和私有子网。

- **公共子网**：包含需要直接互联网访问的资源，如推理 (inference)端点的负载均衡器或用于安全管理访问的堡垒主机（跳板机）。
- **私有子网**：包含您的核心基础设施，例如用于训练的GPU实例。这些实例不能直接从互联网访问，这大幅减少了它们的攻击面。它们可以配置为通过NAT（网络地址转换）网关访问互联网，用于下载软件包等任务，但不允许入站连接。

要控制实例的流量进出，您可以使用防火墙规则。在AWS中，这些称为**安全组**，而在GCP和Azure中，它们就是**防火墙规则**。这些是工作在实例层面的有状态防火墙。例如，您可以为您的训练实例配置一个安全组，使其只允许来自堡垒主机的安全组的SSH入站流量（端口22），而不允许任何其他入站流量。

> 典型的安全网络架构。工程师只能通过首先连接到公共子网中的堡垒主机来访问私有训练实例。训练实例使用安全的IAM角色从S3访问数据，而非通过公共互联网。

### 数据保护：加密与密钥管理

您的数据集和训练模型是宝贵的知识产权。保护它们是不可妥协的。

#### 传输中加密

您的组件之间传输的所有数据都应加密。这意味着所有连接都使用TLS（通常称为SSL）。当您的训练实例从Amazon S3或Google Cloud Storage等对象存储服务获取数据时，请确保您连接到HTTPS端点。这可以防止网络窃听。

#### 静态加密

存储在对象存储或虚拟机磁盘上的数据也应加密。大多数云提供商默认对其对象存储服务启用服务器端加密。这意味着提供商管理加密密钥，并在写入数据时自动加密，在访问数据时自动解密（假设您拥有正确的IAM权限）。为了提高安全性或满足合规性需求，您可以使用**客户管理加密密钥 (CMEK)**，通过AWS KMS或Google Cloud KMS等服务管理加密密钥。这使您能够通过密钥层面撤销对数据的访问。

#### 密钥管理

切勿在代码或配置文件中硬编码API密钥、数据库密码或认证令牌等敏感信息。这是安全漏洞的常见来源。作为替代，请使用专门的密钥管理服务：

- **AWS Secrets Manager** 或 **Parameter Store**
- **Google Secret Manager**
- **Azure Key Vault**
- **HashiCorp Vault** (一个流行的云无关选项)

您的应用代码可以被授予IAM角色，使其能够在运行时获取这些密钥。这种做法将密钥与您的代码库分离，便于凭证轮换，并提供清晰的审计追踪，记录谁在何时访问了哪个密钥。例如，一个提供模型服务的Flask应用程序可以在启动时从Secrets Manager获取其数据库密码，而不是从本地文件读取。

## 参考资料

- [AWS Shared Responsibility Model](https://aws.amazon.com/compliance/shared-responsibility-model/) — Amazon Web Services (AWS) (2023)
  阐明AWS与其客户之间的基本安全合作关系，明确云的安全性与云中数据的安全性责任划分。
- [AWS Well-Architected Framework - Security Pillar](https://docs.aws.amazon.com/wellarchitected/latest/security-pillar/security-pillar.html) — Amazon Web Services (AWS) (2023)
  提供云安全架构最佳实践，涵盖身份、网络、数据保护及与AI基础设施相关的其他关键领域。
- [Best practices for IAM in your AWS account](https://docs.aws.amazon.com/IAM/latest/UserGuide/best-practices.html) — Amazon Web Services (AWS) (2023)
  Publisher: Amazon Web Services
  提供实施强大身份和访问管理的实用指南，包括最小权限原则和使用IAM角色进行安全应用访问。
- [Guidelines on Security and Privacy in Public Cloud Computing](https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-144.pdf) — Peter Mell and Tim Grance (2011)
  Publisher: National Institute of Standards and Technology (NIST)
  提供关于云计算环境中安全和隐私的供应商中立指南，包括数据保护、身份管理和网络安全。
