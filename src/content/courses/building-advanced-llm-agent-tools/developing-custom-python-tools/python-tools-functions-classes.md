---
course: "building-advanced-llm-agent-tools"
chapter: "developing-custom-python-tools"
lesson: "python-tools-functions-classes"
sourceId: 6554
sourceUrl: "https://apxml.com/zh/courses/building-advanced-llm-agent-tools/chapter-2-developing-custom-python-tools/python-tools-functions-classes"
title: "以Python函数和类的形式实现工具"
description: "了解如何将工具实现为Python函数或类，以及在LLM代理集成中何时选择不同方法。"
order: 1
plots: []
sourceHash: "209a549d749cc4b9505f318a55bd12d8bccfc2303d08dd245e557f0f1bc0a2d9"
sourceCorrections: []
---

使用Python实际构建LLM代理工具是代理开发的核心部分。构建自定义工具有两种主要方式：直接的Python函数或更有组织的Python类。你的选择将取决于工具需要执行的任务的复杂性，以及它是否需要在多次使用中记住信息。

Python的可读性和丰富的库使其成为开发LLM代理工具的极佳选择。无论你是执行简单的计算、查询数据库还是与网络服务交互，Python都能提供必要的构建块。

## 将工具实现为Python函数

对于许多任务来说，一个简单的Python函数足以创建一个有效的工具。函数非常适合以下类型的工具：

- **无状态**：它们不需要记住之前调用的任何信息。每次调用都是独立的。
- **单一职责**：它们执行单一、明确的操作。
- **简单的输入/输出**：它们接受清晰的输入并产生可预测的输出。

可以将基于函数的工具视为一个专家，它非常擅长做一件事，并且不需要在不同任务之间保留记录。

### 基于函数的工具的结构

一个结构良好的函数工具应包括：

1. **清晰的函数签名**：使用明确的参数 (parameter)名，关键在于包含Python类型提示来定义你的函数。类型提示可以提高代码清晰度，并可被LLM代理框架用来理解预期的数据类型。
2. **详细的文档字符串**：这非常重要。文档字符串通常是LLM（或管理LLM的代理框架）的主要信息来源，用于理解工具的功能、期望的参数以及返回的内容。一个好的工具文档字符串应清楚地描述：
   - 工具的整体用途。
   - 每个参数：其名称、类型以及它所代表的含义。
   - 返回值：其类型以及它所代表的含义。
3. **核心逻辑**：执行工具操作的实际Python代码。
4. **定义明确的返回值**：函数应以一致且可预测的格式返回数据，以便LLM能够轻松解析和使用。

我们来看一个例子。假设我们想要一个可以计算矩形面积的工具。

```python
def calculate_rectangle_area(length: float, width: float) -> float:
    """
    计算矩形的面积。

    参数：
        length (float): 矩形的长度。必须是正数。
        width (float): 矩形的宽度。必须是正数。

    返回：
        float: 计算得到的矩形面积。
    """
    if length <= 0 or width <= 0:
        # 处理无效输入是一种好的做法，
        # 尽管更详细的错误处理将在后面介绍。
        raise ValueError("长度和宽度必须是正数。")
    return length * width

# 示例用法（不属于工具本身，仅用于演示）
# area = calculate_rectangle_area(10.0, 5.0)
# print(f"面积为: {area}")
```

在这个`calculate_rectangle_area`工具中，文档字符串清楚地说明了其用途给LLM（以及任何人类开发者），包括预期的`length`和`width`参数（包括它们的类型和约束），以及它将返回的内容。类型提示（`length: float`、`width: float`、`-> float`）提供了额外的结构信息。

### 基于函数的工具的优点

- **简单性**：它们易于编写、理解和测试。
- **轻量**：只需要最少的样板代码。
- **直接性**：非常适合LLM需要快速、直接执行操作的任务。

### 局限性

- **无状态**：如果工具需要在多次调用中维持信息或上下文 (context)（例如，记住对话中的先前步骤或累积数据），简单的函数就不够了。
- **组织性**：对于包含多个相关操作或复杂内部逻辑的工具，单个函数可能会变得难以管理。

当你的工具需求超出这些局限时，是时候考虑使用Python类了。

## 将工具实现为Python类

当你需要构建有状态、组合相关功能或需要更复杂设置的工具时，Python类提供了一种有条理的方法。一个类可以封装数据（状态）和操作该数据的方法（行为）。

可以将基于类的工具视为一个更具多功能性的工作者，它能记住过去的交互、管理自己的资源，并提供一系列相关的服务。

### 何时为工具使用类

- **状态管理**：工具需要在调用之间记住信息。例如，连接到数据库的工具可以将连接对象作为其状态的一部分来维护。
- **组合相关操作**：一个类可以提供多个方法，每个方法都作为工具的独特（但相关）功能。例如，一个`FileManager`工具可能包含`read_file`、`write_file`和`list_directory_contents`等方法。
- **复杂初始化**：如果工具在使用前需要大量设置（例如，加载配置、初始化外部客户端），那么类的构造函数（`__init__`）是进行此操作的自然之处。
- **资源管理**：类可以使用`__enter__`和`__exit__`等方法进行上下文 (context)管理，更有效地管理资源（如网络连接或文件句柄），确保资源得到正确获取和释放。

### 基于类的工具的结构

1. **类定义**：以`class`关键字开始。
2. **`__init__`方法（构造函数）**：当创建类的实例时，此方法会被调用。可用于：
   - 初始化任何内部状态（实例属性）。
   - 执行一次性设置操作。
   - 接受工具可能需要的配置参数 (parameter)。
3. **作为工具功能的方法**：类的公有方法通常代表LLM可以调用的特定操作。每个此类方法都应：
   - 具有带有类型提示的清晰签名。
   - 包含详细的文档字符串，解释其特定用途、参数和返回值。这是LLM针对该特定操作将“看到”的内容。
4. **类文档字符串**：为类本身提供文档字符串，描述它所代表的工具集的整体用途和功能。
5. **内部状态**：将需要在方法调用之间持久存在的数据作为实例属性（例如，`self.my_data`）存储。

我们来看一个`UserProfileTool`，它可以存储和获取简单的用户偏好。

```python
class UserProfileTool:
    """
    管理简单的用户档案信息，如姓名和首选城市。
    该工具允许设置和获取这些偏好。
    """

    def __init__(self):
        """初始化一个空的用户档案。"""
        self._name: str | None = None
        self._preferred_city: str | None = None
        print("UserProfileTool 已初始化。") # 用于演示

    def set_preference(self, key: str, value: str) -> str:
        """
        设置用户偏好。
        当前支持“name”和“preferred_city”。

        参数：
            (str): 偏好（例如，“name”，“preferred_city”）。
            value (str): 偏好的值。

        返回：
            str: 一条确认消息。
        """
        if key == "name":
            self._name = value
            return f"用户名设置为“{value}”。"
        elif key == "preferred_city":
            self._preferred_city = value
            return f"首选城市设置为“{value}”。"
        else:
            return f"未知偏好：“{key}”。支持的键是“name”、“preferred_city”。"

    def get_preference(self, key: str) -> str | None:
        """
        获取用户偏好。

        参数：
            preference (str): 要获取的偏好（例如，“name”、“preferred_city”）。

        返回：
            str | None: 偏好的值，如果未设置或偏好未知则为 None。
        """
        if key == "name":
            return self._name
        elif key == "preferred_city":
            return self._preferred_city
        else:
            # 对于未知键，代理框架可能更喜欢报错或返回特定消息。
            # 这里我们返回 None，但你也可以抛出错误或返回消息。
            print(f"尝试获取未知偏好：{key}")
            return None

# 示例用法：
# profile_tool = UserProfileTool()
# print(profile_tool.set_preference("name", "Alex"))
# print(profile_tool.set_preference("preferred_city", "New York"))
# print(f"用户姓名: {profile_tool.get_preference('name')}")
# print(f"用户城市: {profile_tool.get_preference('preferred_city')}")
# print(f"用户最喜欢的颜色: {profile_tool.get_preference('favorite_color')}")
```

在这个`UserProfileTool`中，`__init__`方法初始化了`_name`和`_preferred_city`（内部状态，按照惯例以单下划线作为前缀，表示它们供内部使用，但仍然可以访问）。`set_preference`和`get_preference`方法操作此状态。每个方法都有自己的文档字符串，为LLM代理清楚地定义了其功能。LLM将被呈现`UserProfileTool`中`set_preference`和`get_preference`作为可用的操作。

### 基于类的工具的优点

- **状态管理**：类是构建需要维护状态的工具的自然方式。
- **封装**：它们将数据和操作该数据的方法捆绑在一起，从而产生更清晰、更易于维护的代码。
- **组织性**：非常适合将相关功能组合在一个工具的范畴之下。
- **可重用性**：类实例可以多次创建和使用，可能带有不同的初始配置。

### 考量

- **实例化**：你需要先创建类的实例，然后才能将其方法用作工具。代理框架通常会根据你的工具注册来处理这种实例化。
- **复杂性**：对于非常简单、无状态的任务，与独立的函数相比，使用类可能显得大材小用。

## 选择你的方法：函数对比类

将工具实现为Python函数还是类，其决定取决于工具的要求，特别是在状态和复杂性方面。

> 一个关于工具实现中Python函数和类选择的决策指南。如果工具需要状态或组合多个相关操作，通常更倾向于使用类。否则，一个更简单的函数可能就足够了。

以下是快速总结：

- **在以下情况下使用函数：**
  - 工具是无状态的。
  - 它执行单一、原子性操作。
  - 初始化非常简单。
- **在以下情况下使用类：**
  - 工具需要在调用之间维护状态。
  - 你想将几个相关操作组合在一起（作为方法）。
  - 工具需要非简单的设置或资源管理。

## 文档字符串和类型提示的关键作用

无论你选择函数还是类，有两个元素对于使你的Python代码可用作LLM工具始终非常重要：**文档字符串**和**类型提示**。

- **文档字符串**：如前所述，LLM代理框架通常会解析这些内容以理解你的工具功能。LLM本身依赖这些描述来：

  - **选择正确的工具**：当面对用户请求时，LLM需要将请求与工具文档字符串中描述的功能匹配。
  - **确定参数 (parameter)**：文档字符串应清楚地解释所需的参数、它们的用途和预期格式。
  - **理解输出**：知道工具返回什么有助于LLM处理结果。
  - 编写不佳或缺失的文档字符串可能导致LLM误用工具、未能在适当时候使用它，或提供不正确的参数。
- **类型提示**：Python的类型提示（例如，`name: str`、`count: int`、`-> list[str]`）有多种用途：

  - **开发者清晰度**：它们使你的代码更易于人类阅读和理解。
  - **静态分析**：MyPy等工具可以检查类型不一致性。
  - **LLM框架**：许多代理框架可以借助类型提示来：
    - 自动生成描述工具输入和输出结构的模式（如JSON Schema）。这通常比仅从文档字符串解析自然语言更可靠。
    - 对传入和传出工具的数据执行验证。

```python
# 类型提示和文档字符串的良好使用
def search_knowledge_base(query: str, filters: list[str] | None = None) -> list[dict]:
    """
    在知识库中搜索与查询匹配的文章。
    可以可选地应用过滤器。

    参数：
        query (str): 搜索词或问题。
        filters (list[str] | None, optional): 一个过滤器字符串列表
            要应用（例如，['category:technical', 'tag:python']）。
            默认为 None（无过滤器）。

    返回：
        list[dict]: 一个搜索结果列表，其中每个结果都是一个字典
                    包含“title”、“summary”和“url”。如果没有找到结果，则返回一个空
                    列表。
    """
    # ... 实现细节 ...
    print(f"正在搜索: {query} 使用过滤器: {filters}") # 占位符
    if query == "python tools":
        return [
            {"title": "Python Functions as Tools", "summary": "...", "url": "/ch2/functions"},
            {"title": "Python Classes for Tools", "summary": "...", "url": "/ch2/classes"}
        ]
    return []
```

在这个例子中，`query: str`及其文档字符串解释的结合为LLM提供了精确信息。同样，`-> list[dict]`连同其在文档字符串“返回”部分的描述，为LLM准备了输出的结构。

## 连接LLM代理框架

虽然我们专注于编写工具本身的Python代码，但记住这些函数和类如何成为LLM的“活跃”工具很有用。通常，你将使用LLM代理框架（例如LangChain、LlamaIndex，甚至是自定义构建的框架）。这些框架提供了以下机制来：

1. **注册你的工具**：你将“告知”框架你旨在作为工具的Python函数或类方法。
2. **公开工具描述**：框架通常会内省你的代码（读取文档字符串、类型提示），以创建LLM可以理解的每个工具的结构化描述。
3. **处理工具调用**：当LLM决定使用某个工具时，框架会管理你的Python代码执行，传入LLM提供的参数 (parameter)并返回结果。

本节我们不会详细介绍任何特定框架，但理解这一一般工作流程有助于理解为什么清晰的函数/方法签名、全面的文档字符串和准确的类型提示如此重要。它们是你的Python代码与LLM决策过程之间的桥梁。

通过掌握将工具实现为Python函数和类，你将获得为LLM代理创建广泛功能的灵活性。随着我们继续，我们将在这些基本结构上构建，以添加更高级的功能，如状态管理、外部服务交互和强大的错误处理。

## 参考资料

- [The Python Tutorial - Control Flow Tools and Classes](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQECQGK6pYmIcLU1phMnVEVwxSkp-tOecEg__SWxRhaXEIH_H3AR7SnQJnV6OLaNt5UlHNzPuQO5qdXucDH_Ix8Wqp6etme9V1-94dZqqjLmUOULo-lOIZ90reblASTw4JzB2YetWjI=) — Python Software Foundation (Current)
  官方文档，解释 Python 函数的定义、参数和返回值，以及 Python 类的定义、方法和属性。这些部分为工具实现奠定结构基础。
- [PEP 484 -- Type Hints](https://peps.python.org/pep-0484/) — Guido van Rossum, Jukka Lehtosalo, Łukasz Langa (2014)
  Python 类型提示的官方规范，对于提高代码清晰度并使 LLM 代理框架理解工具参数和返回值的预期数据类型至关重要。
- [Fluent Python: Clear, Concise, and Effective Programming](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHxPkhNGdEG0nsigQJQGY19qdijj-jPS0cYA5DsGz2krx9z4vgQ1QXSlbPcMu7giNGb72fIEB0cqRU66H8ZmwumKv4tp8-1QSOFIvshFJLVeP8OsyrAL8pLts6NFzIPo2izcWUxJpMXNhR6UANPXUtTr8E6XisbW6ExA4UkUIcJci6S02RujBEX6Wheg8_P) — Luciano Ramalho (2022)
  Publisher: O'Reilly Media
  本书深入讲解 Python 编程，强调结构良好的代码、现代语言特性和面向对象模式，适用于构建复杂和有状态的工具。
- [Function calling](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEoVP8optVWjMpNxoqheIG4f-nOj6rM7TgM-5vbmWvSTErW0TL4em-bP3aEYgxG4SgleTezZKwcn5BzELnr8U7yqwQpDbPDOa6EYu_VvgmExQKmhNwEpjJ1RLTJfNib0FQD5Xqx9trNDMk2o7qUR3EuS2G2AJnUOLxkd-QHUIBqDxUXP70mv7Q4) — OpenAI (Current)
  OpenAI 官方指南，演示 LLM 如何通过调用函数与外部工具交互。它展示了如何使用描述（源自文档字符串和类型提示）进行工具选择和参数生成。
