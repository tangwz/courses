---
course: "mlops-for-large-models-llmops"
chapter: "advanced-llmops-systems-workflows"
lesson: "llm-compliance-governance"
sourceId: 3236
sourceUrl: "https://apxml.com/zh/courses/mlops-for-large-models-llmops/chapter-6-advanced-llmops-systems-workflows/llm-compliance-governance"
title: "LLM部署中的合规性与治理"
description: "为LLM使用实践合规要求并建立治理框架。"
order: 6
plots: []
sourceHash: "ae405e7d4cf104f3a1a6bef3c65f78beedb0326d7e9aa5c5c8083182cfb6ef43"
sourceCorrections: []
---

对于统一的LLMOps系统而言，建立合规性和治理框架不仅是良好实践，更是一种必需。大型语言模型凭借其生成能力以及复杂、有时不可预测的行为潜力，带来了超出传统软件乃至标准机器学习 (machine learning)治理范围的特殊挑战。未能应对这些挑战可能导致严重的监管处罚、声誉损害和用户信任流失。详细介绍了如何在先进LLMOps流程中，为LLM部署实施实践以满足合规要求并建立有效治理。

### LLM治理需求的增强

标准MLOps治理通常侧重于模型版本管理、数据血缘和性能监控。尽管这些仍然很重要，但LLM因以下原因需要更广泛的范围：

1. **生成性：** LLM生成新内容，增加了与错误信息、有害输出（毒性、偏见）和知识产权侵权相关的风险。治理必须关注输出监控和控制。
2. **数据规模和敏感性：** 训练和微调 (fine-tuning)涉及大量数据集，可能包含敏感或受版权保护的信息。提示和用户交互也代表着需要谨慎处理的敏感数据。
3. **复杂性和不透明性：** 大型模型的内部运作难以完全理解，这使得确保特定行为或诊断故障变得更困难，因此需要更强的验证和监控规程。
4. **动态输入：** 用户提示的变化性很大，这使得难以预测所有可能的滥用情景或问题交互。

有效的LLM治理是主动的，贯穿于整个生命周期，并考虑模型、数据、基础设施和应用环境。

### LLM的核心合规方面

您的治理策略必须应对几个重叠的合规方面：

- **数据隐私法规：** 遵守GDPR（通用数据保护条例）、CCPA（加州消费者隐私法案）、HIPAA（健康保险流通与责任法案）等法规是根本。这包括：
  - 确保处理训练和推理 (inference)数据的合法依据。
  - 在可行的情况下实施匿名化或假名化等技术（尽管对于非结构化文本有难度）。
  - 为提示和生成输出建立清晰的数据保留和删除策略。
  - 记录涉及个人数据的数据流和处理活动。
  - 管理跨境数据传输限制（如适用）。
- **知识产权（IP）和版权：**
  - 仔细检查训练数据源的版权限制。必要时实施过滤器以排除已知受版权保护的材料。
  - 制定关于LLM生成内容所有权和使用权的策略。
  - 监控输出是否存在潜在抄袭或侵权，尽管这在技术上有难度。
- **负责任的AI和使用策略：** 定义并执行可接受的使用策略对于减轻危害非常重要。这包括：
  - 禁止生成仇恨言论、骚扰或非法内容的使用场景。
  - 实施措施以防止模型输出中出现严重偏见。
  - 向用户提供系统能力和局限性的透明信息。
  - 使开发和部署符合已建立的AI伦理框架和原则。
- **行业特定法规：** 某些行业有特殊要求。金融服务可能要求严格的审计追踪和模型风险管理规程（例如美国SR 11-7）。医疗保健应用需要HIPAA规定的严格数据安全和患者隐私防护措施。
- **模型透明度和可审计性：** 维护全面的记录对于证明合规性和调试问题非常必要。这包括对数据集、代码、模型检查点、超参数 (parameter) (hyperparameter)、评估指标和部署配置进行版本控制。像`模型卡`和`数据集说明书`这样的工具为此类文档提供了结构化格式。

### 建立LLM治理框架

一个实际的治理框架涉及定义流程、角色和控制措施：

1. **定义角色和职责：** 明确指定LLM治理不同方面的归属。这通常需要MLOps工程师、数据科学家、法律顾问、风险管理人员和产品负责人之间的协作。
2. **制定清晰策略：** 创建包含以下内容的书面策略：
   - 数据获取、处理和隐私。
   - 模型开发、验证和测试标准（包括偏见和公平性评估）。
   - 可接受使用指南和内容限制。
   - 部署标准和发布管理流程。
   - 针对合规违规或有害输出的事件响应规程。
   - 定期审计计划。
3. **实施风险评估：** 系统地识别潜在风险（例如数据泄露、偏见输出、安全漏洞、提示注入攻击、监管不合规）。评估其可能性和影响，并定义缓解策略。这应该是一个持续的过程，随模型更新或使用模式变化而重新审视。
4. **采用文档标准：** 强制使用模型卡和数据集说明书，以确保模型目的、性能、局限性、伦理考量和数据来源的一致文档。

### 将治理整合到LLMOps流程中

治理不能是事后才考虑的；它必须融入您的LLMOps流水线的操作结构中。

> 一幅简化图，展示了治理检查点和流程如何整合到LLMOps流水线的不同阶段，从数据准备和评估到部署审批和运行时监控。

重要的集成点包括：

- **CI/CD中的自动化合规检查：**
  - 在数据准备阶段，集成自动化扫描，检查训练数据集中是否存在敏感数据（PII）或受版权保护的内容。
  - 包含对软件依赖项和潜在数据源的许可证合规性检查。
  - 在验证阶段，使用预定义的测试提示和评估指标，自动化测试偏见、毒性或内容策略的遵守情况。如果超出阈值，则触发告警或阻止部署。
- **部署时的策略执行：** 使用与Kubernetes或部署编排器集成的策略引擎（如Open Policy Agent - OPA）来执行规则，例如在部署到生产环境之前要求提供特定文档（例如关联的模型卡）或获得审批。
- **基于角色的访问控制（RBAC）：** 对数据集、模型制品、提示模板、监控仪表板和部署环境实施细粒度访问控制。确保只有授权人员才能触发训练任务、批准部署或访问敏感日志。
- **合规性感知监控：** 扩展监控基础设施（第5章）以跟踪与合规性和负责任AI直接相关的指标。这可能包括不同用户组间的公平性指标、输出监控器标记 (token)的内容策略违规率，或与敏感属性特别相关的数据漂移。
- **不可变日志和审计追踪：** 确保对所有重要事件进行全面、防篡改的日志记录：数据访问、训练运行、模型部署、API请求（可能匿名化或采样的提示和响应）、策略执行决策和用户访问。这些日志对于审计和事件调查非常必要。
- **治理反馈机制：** 实施渠道，供用户或审查人员报告有问题输出或合规顾虑。将此反馈循环不仅用于模型改进，也用于完善治理策略和自动化检查。

### 工具考量

尽管具体工具快速演变，但可以考虑支持LLM治理的工具类别：

- **数据治理平台：** 协助进行数据编目、血缘追踪，并对大型数据集应用隐私策略。
- **ML可观测性与监控平台：** 越来越多地提供针对LLM的功能，包括监控毒性、偏见、幻觉 (hallucination)、数据漂移，并与反馈循环集成。
- **模型风险管理（MRM）工具：** 提供用于模型文档（如模型卡）、跟踪验证结果、管理审批和进行风险评估的框架，这在受监管行业中通常是必需的。
- **策略执行引擎（例如OPA）：** 允许将策略定义为代码，并在Kubernetes或API网关等基础设施中执行它们。
- **安全和扫描工具：** 标准的代码扫描、漏洞扫描和PII检测工具仍然适用，应应用于LLMOps代码库和数据流水线。

### 持续挑战

有效治理LLM是一项持续的努力，伴随着持久的挑战：

- **不断演变的法规：** AI和LLM的法律和监管环境在全球范围内仍在发展（例如欧盟AI法案）。治理框架必须具有适应性。
- **监控的可扩展性：** 大规模监控广泛的潜在输出中是否存在不明显的偏见或知识产权侵权仍然困难。
- **“黑盒 (black box)”性质：** 尽管可解释性技术正在改进，但LLM固有的复杂性使得难以保证不存在不良行为。
- **平衡之道：** 在促成快速创新和确保严格治理之间找到恰当的平衡，需要仔细考虑和持续完善流程。

通过将合规检查、清晰策略和监控嵌入 (embedding)到您的自动化LLMOps流程中，您可以管理与大型语言模型相关的风险，并构建值得信赖、可用于生产的系统。这种集成不仅仅是为了避免处罚；更是为了构建可持续和负责任的AI应用。

## 参考资料

- [Ethics Guidelines for Trustworthy AI](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFP-T2UqIsF8LR2aqUFnkGyPwBIC1TfvDyM1EwxphxgriEOKPHXN4QXEjBJ4mnrAD76nZHOrccsOGx-sR717DV4TDy8lhMVPBVd10i6eosICbrhR_o1MiIyKK8JsC0xMpBcX0WD-UIg_f5Z2Xq-Vn2SooOAZBOAi8C0ZZtz2vPKWY2InGuSGnNGhgdVYQ==) — European Commission High-Level Expert Group on Artificial Intelligence (2019)
  Publisher: European Commission
  这份文件概述了人工智能的关键伦理原则，为制定负责任的人工智能政策和治理框架提供了基础。
- [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993) — Margaret Mitchell, Simone Wu, Andrew Zaldivar, Parker Barnes, Lucy Vasserman, Ben Hutchinson, Elena Spitzer, Inioluwa Deborah Raji, Timnit Gebru (2019)
  Journal: FAT* '19: Conference on Fairness, Accountability, and Transparency; Pages: 3287560–3287596; DOI: [10.1145/3287560.3287596](https://doi.org/10.1145/3287560.3287596)
  引入了模型卡作为机器学习模型、其目的和局限性透明报告的结构化格式。
- [NIST AI Risk Management Framework (AI RMF 1.0)](https://www.nist.gov/artificial-intelligence/ai-risk-management-framework) — National Institute of Standards and Technology (NIST) (2023)
  Publisher: NIST; DOI: [10.6028/NIST.AI.100-1](https://doi.org/10.6028/NIST.AI.100-1)
  为组织管理与人工智能相关的风险提供了全面的指导，为负责任的人工智能开发和部署提供了框架。
- [Open Policy Agent Documentation](https://www.openpolicyagent.org/docs/latest/) — Styra (on behalf of the OPA project) (Current)
  Open Policy Agent (OPA) 的官方文档，OPA 是一种通用策略引擎，用于在整个堆栈中实施策略，适用于LLMOps部署治理。
