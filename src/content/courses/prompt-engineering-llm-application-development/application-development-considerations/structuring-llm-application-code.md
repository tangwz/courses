---
course: "prompt-engineering-llm-application-development"
chapter: "application-development-considerations"
lesson: "structuring-llm-application-code"
sourceId: 3088
sourceUrl: "https://apxml.com/zh/courses/prompt-engineering-llm-application-development/chapter-8-application-development-considerations/structuring-llm-application-code"
title: "大语言模型应用代码组织"
description: "组织基于大语言模型项目的代码的最佳实践。"
order: 1
plots: []
sourceHash: "1c77a7a8fc53aafff17cb7ea4d5588864fc47b588aa736b82418570d22457621"
sourceCorrections: []
---

随着您的大语言模型 (LLM)应用从简单的脚本演变为更复杂的系统，代码的组织方式变得越来越重要。就像在传统软件开发中一样，结构良好的代码库更易于理解、维护、测试和扩展。对于大语言模型应用而言尤其如此，它们通常涉及提示模板、与外部API的交互逻辑以及模型输入输出的特定数据处理等特有组件。及早采纳良好的结构实践将大大节省后期精力，尤其是在与他人协作或准备部署时。

### 关注点分离原则

软件设计中的一个基本原则是*关注点分离*。这意味着应用程序的不同部分应负责独立的功能。将此应用于大语言模型 (LLM)应用有助于管理复杂性。请考虑分离以下常见关注点：

- **大语言模型交互：** 专门用于向大语言模型API（例如OpenAI、Anthropic）发起请求、处理响应、管理温度或最大令牌数等参数 (parameter)，以及处理API特有错误或速率限制的代码。
- **提示管理：** 加载、格式化和可能版本化您的提示的逻辑。提示是应用程序行为的核心，应被视为重要资产，而不仅仅是散布在代码中的硬编码字符串。
- **业务逻辑：** 应用程序使用大语言模型的核心功能。这可能涉及协调多次大语言模型调用、处理中间结果或根据模型输出做决策。
- **数据处理：** 负责为大语言模型准备输入数据、解析大语言模型可能非结构化的输出、根据预期格式验证结果（如第7章所述），以及与数据源交互（例如，第6章讨论的RAG系统）的代码。
- **配置：** 管理API密钥、模型标识符、文件路径和行为参数等设置。这应从主代码库中外部化。
- **表现层/API层：** 如果您的应用程序具有用户界面或作为API公开（例如使用Flask或FastAPI），此层处理传入请求和传出响应，并与业务逻辑层交互。

### 项目组织：目录结构

合理的目录结构使项目浏览和理解变得容易得多。理想的结构取决于应用程序的复杂程度，但这里有几种常见模式：

**简单应用结构：**

对于小型项目，扁平结构可能足够，将关注点清晰分离到不同的Python模块中：

```python
my_llm_app/
├── app.py           # 主应用逻辑或Web服务器（例如Flask）
├── config.py        # 加载配置（通过环境变量获取API密钥，模型名称）
├── llm_client.py    # 用于LLM API交互的函数/类
├── prompt_utils.py  # 用于加载/格式化提示的辅助函数
├── prompts/         # 存储提示模板文件的目录
│   ├── summarize.txt
│   └── qa_cot.txt
├── utils.py         # 通用工具函数（例如，输出解析）
├── .env             # 环境变量（添加到.gitignore！）
└── requirements.txt # 项目依赖
```

**更复杂的应用结构：**

随着应用程序的增长，特别是在包含LangChain等框架或涉及多个独立功能（如问答、摘要、RAG）时，更分层或基于功能的结构更有益：

```python
advanced_llm_app/
├── main.py               # 主入口点（例如，启动Web服务器或CLI）
├── core/                 # 核心共享组件
│   ├── __init__.py
│   ├── config.py         # 配置加载（环境变量，文件）
│   ├── llm_interface.py  # 抽象的大语言模型交互逻辑
│   ├── prompt_manager.py # 集中式提示加载/模板处理
│   └── output_parser.py  # 共享输出解析工具
├── modules/              # 应用功能/模块
│   ├── __init__.py
│   ├── qa/               # 问答模块
│   │   ├── __init__.py
│   │   ├── chain.py      # 问答特有逻辑（例如，LangChain链）
│   │   └── prompts/      # 问答特有提示
│   │       └── retrieval_qa.yaml
│   ├── summarization/    # 摘要模块
│   │   ├── __init__.py
│   │   ├── service.py    # 摘要特有逻辑
│   │   └── prompts/      # 摘要特有提示
│   │       └── condense_document.txt
│   └── rag/              # RAG组件（如果使用）
│       ├── __init__.py
│       ├── retriever.py
│       └── vector_store.py
├── shared/               # 不属于LLM交互核心的共享工具
│   └── data_models.py    # 用于验证的Pydantic模型
├── tests/                # 单元测试和集成测试
│   ├── core/
│   └── modules/
├── .env                  # 环境变量
└── requirements.txt
```

> 结构化大语言模型 (LLM)应用的依赖关系流。功能模块使用核心服务，这些服务处理配置和直接的大语言模型交互。

### 设计模块化组件

思考如何设计可复用的代码组件（函数、类或模块）。

- **大语言模型 (LLM)客户端抽象：** 不要分散API调用，而是创建一个专用类或模块（例如`llm_client.py`或`core/llm_interface.py`）。此组件可以封装以下细节：
  - 选择正确的API端点。
  - 添加认证头。
  - 为临时网络错误或速率限制实现重试逻辑。
  - 规范模型、温度和最大令牌数等参数 (parameter)的传递方式。
  - 请求和响应的基本日志记录。
- **提示管理：** 实现一个`PromptManager`类或实用函数，其可以：
  - 从文件加载提示模板（例如，`.txt`、`.json`、`.yaml`）。
  - 使用模板引擎（如Jinja2或f-strings）处理占位符替换。
  - 可能缓存已加载的提示。
- **框架组件：** 如果使用LangChain等框架，使用其内置抽象。LangChain的`LLM`包装器、`PromptTemplate`、`OutputParser`、`Chain`、`Agent`和`Retriever`类本身就鼓励模块化设计。围绕这些组件组织您的代码。

### 有效管理配置

将API密钥、模型名称或文件路径直接硬编码到应用程序代码中是脆弱且不安全的。请使用以下方式外部化配置：

1. **环境变量：** 这是API密钥等敏感信息的标准做法。在本地开发期间，使用`python-dotenv`等库从`.env`文件加载变量（确保将`.env`添加到您的`.gitignore`中）。在部署环境中，这些变量通常通过托管平台设置。

   ```python
   # config.py
   import os
   from dotenv import load_dotenv

   load_dotenv() # 从.env文件加载变量

   API_KEY = os.getenv("OPENAI_API_KEY")
   MODEL_NAME = os.getenv("DEFAULT_MODEL", "gpt-3.5-turbo")
   ```
2. **配置文件：** 对于模型参数 (parameter)、提示路径或应用程序行为标志等非敏感设置，配置文件（例如`config.yaml`、`settings.toml`）通常更清晰。`PyYAML`、`tomli`或`hydra-core`等库可以帮助加载和管理这些文件。

   ```yaml
   # config.yaml
   llm:
     default_model: "claude-3-sonnet-20240229"
     temperature: 0.7
     max_tokens: 500
   paths:
     prompts_dir: "./prompts"
   features:
     enable_rag: true
   ```

   ```python
   # config.py（使用PyYAML）
   import yaml
   import os

   def load_config(path="config.yaml"):
       with open(path, 'r') as f:
           config = yaml.safe_load(f)
       # 如果需要，允许使用环境变量覆盖
       config["llm"]["api_key"] = os.getenv("ANTHROPIC_API_KEY")
       return config

   CONFIG = load_config()
   MODEL_NAME = CONFIG.get("llm", {}).get("default_model", "unknown-model")
   ```

结合这些方法（例如，从文件加载默认值并使用环境变量覆盖）可提供灵活性。

### 隔离和管理提示

提示定义您大语言模型 (LLM)的行为。在您的项目结构中，将它们视为首要元素：

- **外部存储：** 将提示从Python代码中分离出来。将它们存储在单独的文件中（例如，在专用`/prompts`目录中）。使用清晰的命名约定。
- **使用模板：** 对需要动态输入的提示采用模板引擎。这使结构清晰，并将静态指令部分与可变数据分离。Python的f-strings、Jinja2或框架特有模板类（如LangChain的`PromptTemplate`）都是不错的选择。

  ```python
  # prompts/summarize_template.j2
  Summarize the following text in {{ target_sentences }} sentences:

  {{ text_to_summarize }}

  Summary:
  ```

  ```python
  # prompt_utils.py
  from jinja2 import Environment, FileSystemLoader
  import os

  PROMPTS_DIR = os.path.join(os.path.dirname(__file__), 'prompts')
  env = Environment(loader=FileSystemLoader(PROMPTS_DIR))

  def get_prompt(template_name, **kwargs):
      template = env.get_template(template_name)
      return template.render(**kwargs)

  # 在其他地方使用
  # from prompt_utils import get_prompt
  # my_prompt = get_prompt("summarize_template.j2", target_sentences=3, text_to_summarize=user_input)
  ```
- **考虑版本控制：** 像代码一样，提示也会演进。使用单独的文件可以更容易地使用Git等版本控制系统追踪更改（如第3章所述）。

### 集成日志和错误处理

结构良好的应用程序更容易添加日志和错误处理：

- **集中式日志记录：** 及早配置Python的内置`logging`模块。记录重要事件、决策、输入/输出（可能已净化）和错误。结构化日志记录（例如JSON格式）对后续分析有益。
- **特定异常处理：** 策略性地使用`try...except`块，特别是在外部调用（LLM API、数据库查找）和数据解析/验证周围。如果需要，定义自定义异常类以区分应用程序特有错误。
- **隔离故障点：** 模块化设计有助于限制错误的影响。大语言模型 (LLM)交互模块中的错误，如果处理得当，不应必然导致整个应用程序崩溃。

从一开始就周全地组织您的大语言模型应用代码，可以奠定坚实基础。它提高了清晰度，并使本章后续考量点（例如保护API密钥、监控成本、有效测试和最终部署）的实施成为一个更易于管理的过程。

## 参考资料

- [LangChain Python Documentation](https://python.langchain.com/docs/) — LangChain Developers (2024)
  LangChain的官方文档，该框架为构建LLM应用程序提供抽象和工具，展示了链、提示和代理的模块化设计。
- [Architecting LLM-powered applications](https://cloud.google.com/solutions/llm/architecting-llm-powered-applications) — Google Cloud (2023)
  Publisher: Google Cloud
  一份实用指南，提供了使用大型语言模型构建可扩展和可维护应用程序的架构模式和最佳实践，涵盖系统设计的多个方面。
- [The Hitchhiker's Guide to Python! - Structuring Your Project](https://docs.python-guide.org/writing/structure/) — The Hitchhiker's Guide to Python! Authors (2024)
  该部分为组织Python项目提供了标准建议，包括目录结构、配置管理和依赖处理，直接支持项目中给出的项目组织示例。
