---
course: "getting-started-with-llm-toolkit"
chapter: "foundational-text-generation"
lesson: "configuring-llm-providers"
sourceId: 7750
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-llm-toolkit/chapter-1-foundational-text-generation/configuring-llm-providers"
title: "配置LLM提供方"
description: "学习如何使用配置模块配置不同的LLM提供方，以实现统一的生成接口。"
order: 3
plots: []
sourceHash: "f0faa9b797bb4f18060726c4bc7e166d6be7506ced252c3339a356378de80f64"
sourceCorrections: []
---

与不同的大型语言模型（LLM）提供方协作可能比较复杂。从OpenAI到Anthropic再到Google，每项服务都有自己的API结构、认证方式和模型命名规则。统一的配置系统通过创建一致的接口来解决此问题，它允许你在一个地方设置多个提供方和模型的参数 (parameter)。这让你的应用更灵活，也更容易维护，因为切换模型或增加新模型只需简单的配置更改，而非对代码进行大范围修改。

该工具包的配置通过三个主要组件进行管理：`ConfigManager`、`ProviderConfig`和`ModelConfig`。

- **ProviderConfig**：定义特定LLM提供方的设置，例如其API端点和认证方式。
- **ModelConfig**：定义特定模型的设置，如`gpt-4o-mini`或`claude-3-5-haiku`，包括默认生成参数及其所属的提供方。
- **ConfigManager**：作为中心注册处，存放所有提供方和模型配置。

### 设置提供方和模型

我们先创建一个配置管理器。建议启用`auto_load_env=True`，这样管理器就能自动从你的系统环境变量中找到API密钥。

```python
from kerb.config import ConfigManager, ModelConfig, ProviderConfig
from kerb.config.enums import ProviderType

# 创建一个配置管理器
config = ConfigManager(
    app_name="my_llm_app",
    auto_load_env=True
)
```

接下来，我们将为要使用的提供方定义配置。这里最主要的设置是`api_key_env_var`，它指定了包含API密钥的环境变量名称。这是一种安全推荐做法，避免你将敏感凭据直接写入源代码。

```python
# 配置OpenAI提供方
openai_provider = ProviderConfig(
    provider=ProviderType.OPENAI,
    api_key_env_var="OPENAI_API_KEY"
)

# 配置Anthropic提供方
anthropic_provider = ProviderConfig(
    provider=ProviderType.ANTHROPIC,
    api_key_env_var="ANTHROPIC_API_KEY"
)

# 将提供方加入管理器
config.add_provider(openai_provider)
config.add_provider(anthropic_provider)
```

定义好提供方后，你现在可以配置具体的模型了。每个`ModelConfig`都关联一个`ProviderConfig`，并且可以保存默认参数 (parameter)，例如`max_tokens`或`temperature`，这些参数将用于该模型的生成调用。

```python
# 配置一个OpenAI模型
gpt4o_mini_config = ModelConfig(
    name="gpt-4o-mini",
    provider=ProviderType.OPENAI,
    max_tokens=4096,
    temperature=0.5
)

# 配置一个Anthropic模型
claude_haiku_config = ModelConfig(
    name="claude-3-5-haiku-20241022",
    provider=ProviderType.ANTHROPIC,
    max_tokens=1024,
    temperature=0.7
)

# 将模型加入管理器
config.add_model(gpt4o_mini_config)
config.add_model(claude_haiku_config)
```

下图显示了这些组件如何配合工作。`ConfigManager`包含所有内容，而每个`ModelConfig`都指向一个处理实际API通信的`ProviderConfig`。

> `ConfigManager`作为`ProviderConfig`和`ModelConfig`对象的中心注册处，建立了一个结构化且易于管理的配置系统。

### 使用环境变量管理API密钥

直接在代码中存储API密钥会带来很大的安全风险。推荐的方法是使用环境变量，它使你的凭据与应用逻辑分离。该工具包旨在配合此模式使用。

为使上述代码生效，你需要在运行应用前，在终端中设置相应的环境变量：

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

当你进行API调用时，`ConfigManager`会自动找到正确的`ProviderConfig`，从指定的环境变量中读取API密钥，并认证请求。这种设置不仅提高安全性，还方便管理开发、测试和生产环境的不同密钥。

### 获取和使用配置

设置完成后，你可以轻松获取已定义的任何模型的配置。

```python
# 获取特定模型的配置
retrieved_model = config.get_model("gpt-4o-mini")

if retrieved_model:
    print(f"模型: {retrieved_model.name}")
    print(f"提供方: {retrieved_model.provider.value}")
    print(f"默认温度: {retrieved_model.temperature}")
```

这个配置对象是工具包的`generate`函数在幕后使用的，它将你的请求以正确的设置导向正确的LLM提供方。你现在拥有了进行生成调用的灵活方案，我们将在下一节进行介绍。

## 参考资料

- [OpenAI API Reference](https://platform.openai.com/docs/api-reference) — OpenAI (2024)
  提供OpenAI模型的官方文档，详细说明API结构、认证方法和可用模型名称。
- [Anthropic API Reference](https://docs.anthropic.com/en/api) — Anthropic (2024)
  提供Anthropic Claude模型的官方文档，包括API使用和模型参数。
- [The Twelve-Factor App: III. Config](https://12factor.net/config) — Adam Wiggins (2011)
  阐述将应用程序配置（如API凭据）存储在环境变量中的架构原则。
