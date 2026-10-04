---
course: "multi-agent-llm-systems-design-implementation"
chapter: "agent-communication-coordination"
lesson: "sync-async-agent-communication"
sourceId: 6673
sourceUrl: "https://apxml.com/zh/courses/multi-agent-llm-systems-design-implementation/chapter-3-agent-communication-coordination/sync-async-agent-communication"
title: "建立同步与异步通信连接"
description: "大型语言模型智能体同步与异步通信模式的实际实施。"
order: 7
plots: []
sourceHash: "b49e899891454f4dbe5b3763d899d923aa1904ef098e596748478b31bb050679"
sourceCorrections: []
---

智能体交换消息的方式极大影响着您系统的响应速度、复杂度和可扩展性。在设计多智能体LLM系统的通信时，一个主要决策是确定消息内容和交互协议，然后选择这些交换是同步发生，即智能体等待即时回复；还是异步发生，即智能体发送消息后继续其操作，稍后处理回复。选择同步或异步通信连接，对于构建高效的多智能体LLM系统而言，是根本性的。

### 同步通信：直接对话

同步通信类似于一次直接的电话通话。当智能体A向智能体B发送消息时，智能体A会暂停当前活动，并等待智能体B处理消息并发送响应。只有在收到此响应（或超时）后，智能体A才会恢复其操作。

**特点：**

- **阻塞操作：** 发送智能体被阻塞，等待接收方回复。这简化了发送智能体的逻辑，因为控制流程是顺序的：发送、等待、接收、继续。
- **即时反馈：** 当智能体需要来自另一智能体的信息或结果以完成其当前子任务时，这会很有用。
- **紧密关联的交互：** 通常意味着交互智能体之间有更紧密的依赖。
- **潜在瓶颈：** 如果智能体B响应缓慢，智能体A将保持空闲，可能占用系统资源或后续任务。基于LLM的智能体，其响应时间可能因推理 (inference)延迟而异，这会使情况更糟。

**LLM智能体的实施考量：**
在多智能体LLM系统中，同步通信可能表现为：

- 如果智能体是同一进程内的对象，则为直接方法调用。
- HTTP请求，其中客户端（发送智能体）等待来自服务器（接收智能体）的HTTP响应。
- 基于更简单的消息传递机制实现的“请求-回复”模式。

考虑一个“分析师”智能体，它在生成报告之前需要从“数据获取器”智能体那里获得特定数据。分析师智能体将同步请求数据：

1. 分析师智能体：“数据获取器，请获取第四季度的销售数据。”
2. 分析师智能体：*等待中...*
3. 数据获取器智能体：*处理请求，查询数据库，获取数据。*
4. 数据获取器智能体：“分析师，这是第四季度销售数据：{...}。”
5. 分析师智能体：*接收数据，继续报告生成。*

实施超时机制非常重要。如果数据获取器智能体耗时过长或失败，分析师智能体必须能够妥善应对，例如通过重试、使用默认值或报告错误，而不是无限期等待。

> 一次同步消息交换。智能体A启动通信并暂停执行，直到智能体B处理请求并返回响应。

Python的`requests`库用于HTTP通信，是同步行为的常见例子：

```python
# 智能体A的简化示例
import requests
import time

AGENT_B_URL = "http://agent_b_endpoint/process"

def get_data_synchronously(payload):
    print("智能体A：正在向智能体B发送请求...")
    try:
        response = requests.post(AGENT_B_URL, json=payload, timeout=10) # 在此处阻塞
        response.raise_for_status() # 对HTTP错误引发异常
        print("智能体A：已收到智能体B的响应。")
        return response.json()
    except requests.exceptions.Timeout:
        print("智能体A：向智能体B的请求超时。")
        return None
    except requests.exceptions.RequestException as e:
        print(f"智能体A：向智能体B的请求失败：{e}")
        return None

# result = get_data_synchronously({"query": "Q4 sales"})
# if result:
#     # 处理结果
#     pass
```

在此代码片段中，`requests.post`是一个阻塞调用。智能体A的代码执行在该行暂停，直到智能体B响应或达到超时。

### 异步通信：非阻塞交换

异步通信类似于发送电子邮件或短信。智能体A将其消息发送给智能体B，并立即继续执行其其他任务。智能体B在自己的时间处理消息。当智能体B有响应时，它会将其发回，而智能体A可以在准备好时处理此响应，通常通过回调机制或定期检查消息队列。

**特点：**

- **非阻塞操作：** 发送智能体在发送消息后不等待回复。这提高了发送方的响应速度和系统总体吞吐量 (throughput)。
- **解耦：** 智能体之间耦合更松散。发送方不需要知道接收方是否立即可用或处理请求需要多长时间。
- **复杂性：** 管理异步流程可能更复杂。您需要将响应与请求关联（如果需要）、处理回调或管理消息队列的机制。
- **可扩展性：** 异步系统通常可扩展性更好，因为智能体不会直接等待彼此，并且可以同时处理多个交互。

**LLM智能体的实施考量：**
异步模式在以下情况特别有益：

- LLM智能体需要执行长时间运行的任务（例如，广泛研究、复杂内容生成）。
- 智能体需要向其他多个智能体广播信息，而无需等待单个确认。
- 您希望构建一个更具弹性的系统，其中一个智能体的临时不可用不会停止其他智能体。

实现异步通信的常见方式包含：

- **消息队列：** RabbitMQ、Apache Kafka或Redis Streams等系统充当中间件。智能体A向队列发布消息，智能体B订阅该队列以接收和处理消息。响应可以通过另一个队列发送回去。
- **回调：** 智能体A发送请求并提供一个函数（回调），智能体B应使用结果调用该函数。
- **Future/Promise（未来/承诺）对象：** 这些是表示异步操作最终结果的对象。智能体A立即获得一个Future对象，并可以检查其状态或附加一个在完成时执行的回调。
- **事件驱动架构：** 智能体响应事件。发送消息可以是一个事件，接收响应可以是另一个事件。

设想一个“内容生成器”智能体，其任务是撰写一篇长篇文章。“编排器”智能体可能会异步分配此任务：

1. 编排器智能体：“内容生成器，请撰写一篇关于‘AI的未来’的文章。”（将请求放入队列或通过异步调用发送）
2. 编排器智能体：*继续处理其他任务，例如为其他智能体分配工作或监控系统状态。*
3. 内容生成器智能体：*在可用时从队列中获取任务，进行研究（可能包含其他智能体），起草文章（这个过程可能需要几分钟或几小时）。*
4. 内容生成器智能体：“编排器，‘AI的未来’文章已准备就绪。请在[链接/ID]处查看。”（将通知/结果放入响应队列或调用回调）
5. 编排器智能体：*在检查队列或其回调被触发时，接收通知并处理已完成的文章。*

> 使用消息队列进行异步消息交换。智能体A发送消息到队列并继续其操作。智能体B独立地从队列中获取并处理消息。响应也可以异步方式应对。

Python的`asyncio`库是使用事件循环编写并发代码的标准方式：

```python
# 使用asyncio和消息队列客户端的智能体A简化示例
import asyncio

# 假设some_message_queue_client提供异步发送和接收方法
# async def send_to_queue(queue_name, message): ...
# async def listen_to_queue(queue_name, callback): ...

async def process_response_callback(message):
    print(f"智能体A：收到异步响应：{message}")
    # 进一步处理响应

async def submit_task_asynchronously(payload):
    print("智能体A：正在异步向智能体B提交任务...")
    # 如果智能体A本身是异步的，此调用通常不会阻塞整个智能体A
    # 它可能包含将消息放入队列或进行异步HTTP调用。
    # 为了演示，我们模拟发送到队列并启动一个响应监听器。
    # await some_message_queue_client.send("agent_b_task_queue", payload)
    print("智能体A：任务已提交。继续其他操作。")
    # 智能体A现在可以做其他事情。
    # 一个单独的任务可能会监听响应。

# 在一个真正的asyncio应用中，您会更稳固地负责任务和监听器。
# async def main():
#     # 启动响应监听器（简化）
#     # asyncio.create_task(some_message_queue_client.listen_to_queue("agent_a_response_queue", process_response_callback))
#
#     await submit_task_asynchronously({"task": "generate_report"})
#     print("智能体A：提交任务后返回主流程。")
#     await asyncio.sleep(1) # 继续运行以允许其他任务（如监听器）操作
#
# if __name__ == "__main__":
#   asyncio.run(main())
```

此`asyncio`示例阐明了非阻塞特性。`submit_task_asynchronously`通常会将任务放置给智能体B并迅速返回，允许智能体A执行其他操作。与智能体B的实际通信将在后台进行，由事件循环和潜在的消息队列客户端负责。

### 选择合适的连接：同步、异步，还是两者兼有？

同步和异步通信之间的决定并非总是相互排斥的；许多复杂的系统采用混合方法。选择很大程度上取决于具体的交互和期望的系统特点：

- **任务依赖：**

  - **同步：** 如果智能体A在没有智能体B的即时结果的情况下绝对无法继续。例如，智能体在允许访问前验证用户凭据。
  - **异步：** 如果智能体A在智能体B处理请求时可以执行其他有用工作，或者任务是“即发即忘”的通知。示例：智能体记录事件。
- **响应时间预期：**

  - **同步：** 适用于快速、可预测的交互。如果LLM推理 (inference)时间对于特定类型的请求持续很低，同步可能可以接受。
  - **异步：** 优先用于完成时间可变或可能较长的任务，这类情况在复杂的LLM生成或工具使用中并不少见，以防止占用调用智能体。
- **系统响应速度和吞吐量 (throughput)：**

  - **同步：** 如果许多智能体被阻塞等待同步调用，可能会降低系统整体响应速度。
  - **异步：** 通常带来更高的系统整体吞吐量和更好的响应速度，因为智能体空闲时间更少。
- **实施复杂性：**

  - **同步：** 对于单个请求-响应对来说，更容易理解，也更容易在本地调试。
  - **异步：** 引入更多活动组件（消息代理、回调、事件循环），这会增加开发和调试的复杂性。负责分布式状态和跨异步调用的错误应对需要周密设计。
- **可扩展性和弹性：**

  - **同步：** 如果同步链中的一个重要智能体变得无响应，可能造成级联故障。扩展可能需要复制整个同步链。
  - **异步：** 通常更具可扩展性和弹性。消息队列可以缓冲请求，允许消费者智能体按照自己的速度处理请求，并在智能体暂时失败时提供持久性。
- **资源利用：**

  - **同步：** 如果线程或进程被占用而等待，可能导致资源利用效率低下。
  - **异步：** 可以更有效利用资源，特别是在I/O密集型任务中，通过允许单个线程通过事件循环负责许多并发操作。

**混合场景：**
一种常见模式是，对智能体自身处理步骤内的重要、快速内部查找或验证使用同步通信，而对较长时间的任务或与外部系统或其他可能具有不可预测延迟的智能体的交互使用异步通信。例如，“编排器”智能体可能会同步询问“规划器”智能体工作流程的下一步（一个快速的内部决定），但随后异步地将该步骤分派给“工作者”智能体，该智能体包含LLM调用和工具执行。

**实际难题：**

- **错误应对：** 在同步调用中，错误通常通过异常或错误代码立即传播。在异步系统中，您需要机制来报告和应对可能发生在解耦组件中的错误，例如死信队列或专用的错误应对服务。
- **调试和追踪：** 在具有多个队列和事件驱动组件的异步系统中追踪请求可能具有挑战性。分布式追踪工具和全面的日志记录是必不可少的。
- **状态维护：** 当智能体发送异步请求时，它需要维护其状态，以便在响应最终到达时能够正确处理。这可能包含存储上下文 (context)或使用关联ID。

建立通信连接，无论是同步还是异步，都是高效多智能体LLM系统的根本。这种选择不仅影响单个智能体的交互，还影响系统的整体架构、性能和可维护性。当您进入实践练习时，请思考哪种模式或模式组合最适合解决使两个LLM智能体能够通信并达成共同目标的问题。

## 参考资料

- [An Introduction to MultiAgent Systems](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQG2qRukPDZbvyCQyHKrnodNZgwSjX520ZjS96pKdhuOIuzoKy47Gd5P7eaiql6KlF69Flc8vQ_mGjbCcJ0wDih9IpbPCJ0Uw9nrSJ7jNnlkC72EM6dfLDkiiP7_5gON9317mTb39ZCM9nVZ76CipHvCGmXbrS6rEpP6FP822Yst05e9fsdQmpZvvtK0_kWHcBZl4W0f7CMB3wk=) — Michael Wooldridge (2009)
  Publisher: John Wiley & Sons
  一本标准教科书，讨论了代理通信语言、交互协议和协调机制，为多代理架构中的通信提供了背景。
- [\`asyncio\` - Asynchronous I/O, event loop, coroutines and tasks](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQF_USPCUxQZSpDC1nkh1dMBcvEbEAGdSciOKaC1GJysERpZOIM89xTwi7-92VidZkQjxIJGLcZWABFpw10ICXCt5HSY5KzBgh_0g8_St3p3vg3ch_0SpZcOh5RaejewNAC4E9KgohEMRw==) — Python Software Foundation (2024)
  Python `asyncio` 库的官方文档，对于在基于 Python 的多代理系统中实现异步和并发通信模式至关重要。
- [Enterprise Integration Patterns: Designing, Building, and Deploying Messaging Solutions](https://enterpriseintegrationpatterns.com/) — Gregor Hohpe, Bobby Woolf (2003)
  Publisher: Addison-Wesley
  一本关于消息传递模式和架构风格的开创性著作，这些模式和风格是构建可扩展、异步分布式系统（包括使用消息队列的系统）的基础。
