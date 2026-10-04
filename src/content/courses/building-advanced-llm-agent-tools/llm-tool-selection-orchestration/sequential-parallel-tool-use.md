---
course: "building-advanced-llm-agent-tools"
chapter: "llm-tool-selection-orchestration"
lesson: "sequential-parallel-tool-use"
sourceId: 6580
sourceUrl: "https://apxml.com/zh/courses/building-advanced-llm-agent-tools/chapter-3-llm-tool-selection-orchestration/sequential-parallel-tool-use"
title: "实现工具的顺序与并行使用"
description: "对比并实现针对不同智能体需求的顺序与并行工具执行策略。"
order: 6
plots: []
sourceHash: "837694e1393bfc3eae1d10cd827726eee8d2e9777678e63759177f092393637c"
sourceCorrections: []
---

随着你的LLM智能体变得更精巧，它常常需要使用多个工具来达成一个复杂目的。这些工具的执行方式——是依次进行还是同时运行——大大影响智能体的效率和功用。实现顺序与并行工具使用，能让你的智能体处理更复杂的任务。

### 顺序工具执行：有序方法

顺序工具执行是最直接的方法：工具每次运行一个，并按特定次序。当一个工具的输出作为下一个工具的输入，从而形成依赖链时，这种方法是必需的。可以把它想象成一个食谱，每一步都必须在前一步完成后才能开始。

举例来说，一个智能体若要执行“找到今天热门科技新闻，总结并邮件给我”的任务，很可能会按顺序执行这些步骤：

1. **工具1（网络搜索）**：找到关于“热门科技新闻”的文章。
2. **工具2（内容提取）**：从最前面几篇文章中提取文本内容。
3. **工具3（总结）**：总结提取出的文本。
4. **工具4（邮件发送）**：发送总结。

每个工具的成功完成和输出都是后续工具的前提。智能体，或者你为其定义的逻辑，需要管理这个流程，确保数据从一个工具正确传递给下一个。

> 典型的顺序工具执行流程，一个工具的输出作为下一个工具的输入。

**何时使用顺序执行：**

- **数据依赖**：主要原因。当工具B需要工具A产生的数据时。
- **逻辑顺序**：有些任务有固有的逻辑顺序，必须遵循。
- **简洁性**：相较于并行执行，更容易实现和调试。
- **资源限制**：如果工具是资源密集型的，逐个运行可以避免系统过载。

顺序流程中的错误处理相对直接。如果一个工具失败，智能体可以停止序列，若有备选工具则尝试使用，或报告失败。

### 并行工具执行：并发工作

并行工具执行指同时运行多个工具。当智能体需要执行多个独立任务，且这些任务的完成顺序不重要，或需要在继续前收集来自多个源的结果时，这种方法非常有效。

考虑一个智能体被要求“获取旧金山、伦敦和东京的当前天气”。这是三个独立查询。智能体可以：

1. **工具1（天气API - 旧金山）**：获取旧金山天气。
2. **工具2（天气API - 伦敦）**：获取伦敦天气。
3. **工具3（天气API - 东京）**：获取东京天气。

这三个调用可以并发进行。智能体无需等待旧金山天气报告，就可以请求伦敦的天气。一旦所有结果返回，智能体就可以汇总它们。

> 并行工具执行的示例，独立任务并发执行，其结果被组合。

**何时使用并行执行：**

- **独立任务**：当工具启动时不依赖彼此的输出时。
- **效率**：可以大大缩短总执行时间，尤其对于API调用等I/O密集型操作。
- **响应性**：智能体通过同时处理多个方面，可以显得更具响应性。

实现并行执行需要更精巧的管理。在Python中，`asyncio`非常适合智能体工具中常见的I/O密集型任务（如API调用）。你需要将工具函数定义为异步的（使用`async def`），并对可能阻塞的操作使用`await`。然后，可以使用`asyncio.gather()`来并发运行多个可等待任务。

```python
import asyncio

# 假设这些在其他地方已定义且是可等待的
# async def call_weather_api(city: str) -> dict: ...
# async def call_news_api(topic: str) -> list: ...

async def fetch_data_in_parallel():
    # 示例：获取两个城市的天气和某个主题的新闻
    weather_sf_task = call_weather_api("San Francisco")
    weather_london_task = call_weather_api("London")
    news_tech_task = call_news_api("technology")

    # asyncio.gather 并发运行所有任务
    results = await asyncio.gather(
        weather_sf_task,
        weather_london_task,
        news_tech_task,
        return_exceptions=True  # 对于处理单个任务失败很重要
    )

    # 处理结果：
    # results[0] 将是旧金山天气或异常
    # results[1] 将是伦敦天气或异常
    # results[2] 将是科技新闻或异常

    # 进一步的逻辑来处理和组合结果...
    # 例如，检查 results 中每个项是否为 isinstance(res, Exception)

    sf_weather = results[0] if not isinstance(results[0], Exception) else "Error fetching SF weather"
    london_weather = results[1] if not isinstance(results[1], Exception) else "Error fetching London weather"
    tech_news = results[2] if not isinstance(results[2], Exception) else "Error fetching tech news"

    print(f"旧金山天气: {sf_weather}")
    print(f"伦敦天气: {london_weather}")
    print(f"科技新闻: {tech_news}")

# 在脚本中运行此代码：
# if __name__ == "__main__":
#     asyncio.run(fetch_data_in_parallel())
```

处理并行执行中的错误需要小心。如果一个工具失败，智能体需要决定其他并行任务是否应继续，或者整体操作是否受到影响。使用`asyncio.gather`的`return_exceptions=True`参数 (parameter)，可以使智能体获取所有结果或异常，并单独处理它们。

### 混合方法：兼顾两者优点

许多复杂任务都受益于混合方法，即结合顺序和并行执行。智能体可能会并行执行多个数据收集步骤，然后顺序处理并整合收集到的信息。

例如，一个规划旅行的智能体可能：

1. **并行阶段**：
   - 工具A：获取航班选项。
   - 工具B：获取酒店空房情况。
   - 工具C：获取目的地的天气预报。
2. **顺序阶段（所有并行任务完成后）**：
   - 工具D：根据偏好和预算分析航班和酒店选项（输入：A和B的输出）。
   - 工具E：总结最佳方案，包括天气信息（输入：D和C的输出）。

这种结构允许智能体在可能的情况下优化速度（并行数据获取），同时在必要时保持逻辑顺序（分析和总结）。

> 结合并行数据收集与顺序处理的混合工作流。

### 智能体执行策略的决策

LLM智能体如何决定是顺序还是并行使用工具？这通常涉及多种方式的结合：

1. **工具描述**：精心编写的工具描述可以暗示依赖关系。如果一个工具的输入参数 (parameter)明确引用另一个特定工具的输出，则暗示了顺序执行。反之，具有独立输入的工具是并行使用的备选项。
2. **智能体的规划能力**：更高级的智能体可以生成多步骤计划。这个计划的结构（例如，线性动作序列与一组独立子目标）决定了执行策略。
3. **明确的编排逻辑**：作为开发者，你可以实现明确的逻辑来编排工具调用。这可能涉及为常见任务类型预定义模板，指定顺序或并行步骤。例如，一个“研究报告”模板可能总是在顺序总结步骤之前并行运行数据收集工具。
4. **资源意识**：精密的智能体可能会考虑资源限制。如果许多工具可用且可以并行运行，智能体（或其底层框架）可能会限制并行执行，以避免外部API或本地系统资源过载。

顺序、并行或混合执行的选择是工具编排的一个基础方面。通过理解这些模式及其实现影响，你可以设计出不仅有能力，而且在如何运用工具解决问题和完成任务上表现高效的智能体。

## 参考资料

- [\`asyncio\` - Asynchronous I/O](https://docs.python.org/3/library/asyncio.html) — Python Software Foundation (2024)
  Python `asyncio` 库的官方文档，对于理解和实现代理工具执行中的并行、I/O密集型任务很重要。
