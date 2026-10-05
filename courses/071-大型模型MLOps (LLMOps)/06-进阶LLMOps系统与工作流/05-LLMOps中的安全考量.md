# LLMOps中的安全考量

来源：[原文](https://apxml.com/zh/courses/mlops-for-large-models-llmops/chapter-6-advanced-llmops-systems-workflows/llmops-security-considerations)

[返回章节目录](README.md) · [返回课程目录](../README.md)

大语言模型 (LLM)的实际运用带来特有的安全问题，超出一般的MLOps做法。大语言模型常处理的数据的规模、复杂性、生成能力及敏感度，形成一个特别的受攻击面。在整个LLMOps生命周期中加入安全措施，不只是一个好做法；它是一个基本要求，用于构建可靠、值得信赖和有韧性的系统。忽视安全可能导致模型被破坏、数据泄露、服务中断和声誉严重受损。

### 了解LLMOps中的难题

保护LLM系统需要明白它们特有的弱点。虽然一般的基础设施安全仍然很重要，但在LLMOps环境中，有几种威胁尤其明显：

1. **提示注入（Prompt Injection）**：这涉及构造恶意输入（提示），旨在操纵LLM的行为。攻击者可能试图绕过安全规定，提取模型训练或可访问的敏感信息（例如，在RAG系统中），生成有害内容，或执行非预期操作。
   - *直接注入（Direct Injection）：* 恶意指令直接由用户输入提供。
   - *间接注入（Indirect Injection）：* 恶意指令隐藏在检索到的数据中（例如，LLM稍后处理的网页或文档，如在RAG系统中）。
2. **数据投毒（Data Poisoning）**：恶意行为者可能试图破坏训练或微调 (fine-tuning)数据，以在模型中引入偏见、后门或特定的弱点。在中使用外部数据源或持续学习周期中，这一点尤其相关。
3. **模型提取和窃取（Model Extraction and Theft）**：大型模型代表着重要的投入。攻击者可能通过多种方式窃取模型的权重 (weight)或架构，包括通过访问弱点或使用模型提取攻击（重复查询模型以推断其参数 (parameter)或复制其功能）。
4. **不安全的输出处理（Insecure Output Handling）**：LLM输出可能无意中包含敏感信息（PII）、专有数据或有害内容。未能充分清理或监控输出，使其到达最终用户或下游系统之前，会带来重大风险。
5. **拒绝服务（DoS）/资源耗尽（Resource Exhaustion）**：LLM推理 (inference)计算量大。攻击者可以借此发送大量复杂查询或资源密集型请求，导致服务不可用和运营成本上升。
6. **基础设施弱点（Infrastructure Vulnerabilities）**：支持LLMOps的复杂基础设施（分布式计算集群、向量 (vector)数据库、数据管道）如果未得到适当保护，会带来众多潜在的故障点或攻击点。配置错误的访问控制、不安全的网络设置或有漏洞的容器镜像都可能成为攻击目标。
7. **供应链攻击（Supply Chain Attacks）**：LLMOps管道常依赖许多第三方库、预训练 (pre-training)基础模型和数据集。供应链中任何一个环节的破坏都可能给整个系统带来弱点。
8. **向量数据库安全（RAG Specific）（Vector Database Security (RAG Specific)）**：使用检索增强生成（RAG）的系统，其向量数据库面临特有风险。这些风险包括未经授权访问敏感文档嵌入 (embedding)、通过相似性搜索导致数据泄露，或检索过程被篡改。

### 在LLMOps中实行安全控制

处理这些威胁需要一个多层安全策略，集成到LLMOps管道中。

#### 安全数据管理

保护用于训练、微调 (fine-tuning)和RAG的数据是根本性的。

- **访问控制：** 对数据集和数据存储系统（例如S3存储桶、数据库）实行严格的基于角色的访问控制（RBAC）。应用最小权限原则。
- **加密：** 对静态数据（存储中）和传输中的数据（网络传输）进行加密。
- **数据最小化与匿名化：** 在可行的情况下，减少敏感数据的使用。在数据准备阶段，特别是对于PII，采用匿名化或假名化技术。
- **安全管道：** 确保数据处理管道有适当的安全控制和输入验证。

#### 安全训练和微调

模型训练阶段需要仔细考虑安全问题，特别是在分布式环境中。

- **环境隔离：** 使用网络分段（例如VPC、子网、安全组）隔离训练集群。
- **依赖项扫描：** 定期使用Snyk、Trivy或云提供商的扫描工具检查训练代码、库和容器基础镜像是否存在已知弱点。
- **安全凭据：** 避免硬编码凭据。使用安全密钥管理方案（例如HashiCorp Vault、AWS Secrets Manager、Azure Key Vault）。
- **数据溯源：** 维护用于训练的数据来源的清晰记录，以帮助识别潜在的数据投毒事件。

#### 模型和推理 (inference)安全

保护模型本身和推理端点非常重要。

- **模型产物安全：** 使用访问控制和可能的加密方式安全存储模型检查点和最终产物。实行版本管理。
- **输入验证与清理：** 在模型周围设置“防护栏”。清理用户输入以检测并阻止潜在的提示注入尝试。这可能涉及模式匹配、训练用于检测恶意提示的分类器，或限制输入长度和复杂性。
- **输出监控与过滤：** 在返回LLM输出之前，检查其是否包含敏感数据模式（例如PII的正则表达式）、毒性、偏见，或是否符合安全策略。可以使用内容审查服务或定制过滤器。
- **API安全：** 使用标准API安全实践保护推理端点：认证（例如API密钥、OAuth）、授权、TLS加密、速率限制（以减轻DoS和模型提取），以及输入验证。使用API网关进行集中控制。
- **模型混淆/水印（进阶）：** 研究模型水印等技术以追踪泄露的模型，尽管实际实现可能复杂。

#### 基础设施和操作安全

保护底层基础设施和操作流程。

- **基础设施即代码（IaC）安全：** 在部署前检查IaC模板（Terraform、CloudFormation）是否存在安全配置错误。
- **容器安全：** 使用最小化、强化的基础镜像。检查容器镜像是否存在弱点。对容器实行运行时安全监控。
- **向量 (vector)数据库安全：** 应用严格的访问控制，加密向量存储中的数据，监控查询模式是否有异常，并保持数据库软件更新。
- **日志记录和审计：** 为所有组件实行全面的日志记录：API请求、模型响应、数据访问、基础设施变更、安全事件。使用集中式日志和监控平台（例如ELK stack、Datadog、Splunk）。定期审查日志以查找可疑活动。
- **事件响应：** 制定并定期测试专门针对潜在LLM安全事件的响应计划（例如，处理检测到的提示注入活动，响应模型泄露）。

### 将安全融入LLMOps工作流程

安全不应是事后补充，而应是自动化LLMOps工作流程的一个组成部分。

> 安全检查点集成在LLMOps管道中，从数据准备和训练到CI/CD和生产部署。

这包括：

- **自动化扫描：** 将代码的静态应用安全测试（SAST）、依赖项扫描、容器镜像扫描和IaC扫描直接集成到CI管道中。
- **安全门控：** 在部署管道中实行检查，如果检测到安全弱点或配置错误，则阻止推进。
- **持续监控：** 通过可观察性平台持续监控生产环境中的安全事件、模型行为异常或基础设施漂移。
- **定期审计：** 定期进行专门针对LLM应用及其基础设施的安全审计和渗透测试。
- **反馈循环：** 根据监控和事件中获得的信息来更新安全控制、数据过滤、模型再训练策略以及提示工程 (prompt engineering)实践。

保护LLMOps需要一种积极主动、纵深防御的方法。通过理解特有的威胁，并将安全控制措施嵌入 (embedding)到整个生命周期中，组织可以降低风险并构建更可靠的大语言模型 (LLM)系统。随着LLM能力和对抗技术的发展，这种持续的警惕性必不可少。

## 参考资料

- [OWASP Top 10 for Large Language Model Applications](https://llm.owasp.org/) — OWASP Foundation (2023)
  Publisher: OWASP Foundation
  本指南识别并解决了大型语言模型应用程序特有的最关键安全漏洞，涵盖了本节中提到的许多威胁。
- [Not what you've signed up for: Compromising real-world LLM-integrated applications with indirect prompt injections](https://arxiv.org/abs/2302.12173) — Kai Greshake, Sahar Abdelnabi, Shailesh Mishra, Christoph Endres, Thorsten Holz, Mario Fritz (2023)
  Journal: arXiv preprint arXiv:2302.12173; DOI: [10.48550/arXiv.2302.12173](https://doi.org/10.48550/arXiv.2302.12173)
  本研究论文分析了直接和间接提示注入攻击的机制和影响，这是本节内容中强调的关键威胁。
- [Artificial Intelligence Risk Management Framework (AI RMF 1.0)](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf) — National Institute of Standards and Technology (NIST) (2023)
  Journal: NIST AI 100-1; Publisher: National Institute of Standards and Technology
  提供了一个管理人工智能系统风险的全面框架，涵盖了适用于LLMOps系统和工作流的安全考虑。
- [Applied AI Security: Protecting AI Models from Adversarial Attacks](https://learning.oreilly.com/library/view/applied-ai-security/9781098158019/) — Alex Cheang, Alex Huang, and Kyla Singh (2024)
  Publisher: O'Reilly Media
  本书提供了针对多种AI安全威胁的实用防御策略，如数据中毒和模型窃取，与LLM运维安全密切相关。

---

[上一节](04-LLM%E5%86%8D%E8%AE%AD%E7%BB%83%E4%B8%8E%E5%BE%AE%E8%B0%83%E6%B5%81%E7%A8%8B%E8%87%AA%E5%8A%A8%E5%8C%96.md) · [下一节](06-LLM%E9%83%A8%E7%BD%B2%E4%B8%AD%E7%9A%84%E5%90%88%E8%A7%84%E6%80%A7%E4%B8%8E%E6%B2%BB%E7%90%86.md)
