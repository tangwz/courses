---
course: "intro-llm-agents"
chapter: "your-first-llm-agent"
lesson: "hands-on-creating-to-do-list-agent"
sourceId: 6578
sourceUrl: "https://apxml.com/zh/courses/intro-llm-agents/chapter-3-your-first-llm-agent/hands-on-creating-to-do-list-agent"
title: "动手实践：创建一个待办事项代理"
description: "一项动手活动，构建一个管理待办事项清单的基本LLM代理。"
order: 8
plots: []
sourceHash: "743466e679c4a19e2051b788bd56def0cd0b49f8dc11971b9c7ec9c14eea6434"
sourceCorrections: []
---

构建你的第一个LLM代理。这是一个帮助管理待办事项清单的简单代理。它将展示一个代理的主要组成部分是如何运作的，从接收指令到执行任务。我们将着重于结构和流程，对LLM的决策部分采用简化方法，以便你能专注于代理的运行机制。

### 定义我们的待办事项代理的任务

我们的代理将是一个用于管理任务列表的个人助理。在基本层面上，它应该能够：

1. 添加新任务到列表。
2. 查看所有当前任务。
3. 从列表中移除一个任务。
4. 让用户知道它执行了什么操作。

在这个例子中，我们将把待办事项清单保存在内存中，这意味着如果你停止并重新启动程序，它就会重置。在后面的章节中，我们将了解代理如何更持久地记住信息。

### 设置代理的“工具”（任务管理函数）

首先，让我们创建一些基本函数，我们的代理将用它们来管理待办事项清单。把这些看作是代理的特定技能或工具。我们将使用一个简单的Python列表来存储任务。

```python
# 此列表将存储我们的任务
todo_list = []

def add_task(task_description):
    """添加一个任务到待办事项清单。"""
    todo_list.append({"task": task_description, "done": False})
    return f"Okay, I've added '{task_description}' to your list."

def view_tasks():
    """显示待办事项清单中的所有任务。"""
    if not todo_list:
        return "Your to-do list is empty!"

    response = "Here are your tasks:\n"
    for index, item in enumerate(todo_list):
        # 我们会显示带编号的任务，以便后续更容易移除
        response += f"{index + 1}. {item['task']}\n"
    return response.strip() # 移除末尾的换行符

def remove_task(task_number_str):
    """根据任务编号从列表中移除任务。"""
    try:
        task_number = int(task_number_str)
        if 1 <= task_number <= len(todo_list):
            removed_task = todo_list.pop(task_number - 1)
            return f"Okay, I've removed '{removed_task['task']}'."
        else:
            return "Sorry, I couldn't find a task with that number. Try 'view tasks' to see the numbers."
    except ValueError:
        # 这处理输入不是数字的情况
        return "Please tell me the number of the task you want to remove."
    except IndexError:
        # 这是一个备用处理，尽管上面的检查应该能捕获到
        return "That task number is out of range."
```

在这段代码中：

- 我们初始化了一个名为 `todo_list` 的空列表。列表中的每个项目将是一个字典，包含任务描述和状态（我们目前还没有完全使用状态，但这是一种好的习惯）。
- `add_task` 函数接收一个描述并将其添加到我们的列表中。
- `view_tasks` 函数格式化列表以供显示。如果列表为空，它会提示。
- `remove_task` 函数接收一个字符串（我们期望它是一个数字），将其转换为整数，并尝试移除该位置的任务（对0基索引进行调整）。如果数字无效或未找到，它包含一些基本的错误处理。

### 代理的“大脑”：解释用户命令（简化版）

这是大型语言模型（LLM）通常发挥其核心作用的地方。LLM会分析用户的自然语言输入，并决定使用哪个动作（或工具）以及向该动作传递什么信息。

对于我们的第一个代理，我们将用一个非常简单的函数来模拟这个“大脑”。它会在用户的输入中查找关键词来决定做什么。这是一个很大的简化，但它有助于我们专注于代理的整体结构。稍后，你将了解到如何在这里集成一个真实的LLM。

```python
def process_user_command(user_input):
    """
    模拟LLM理解用户输入并决定采取哪个动作。
    返回要调用的函数和任何必要的参数。
    """
    user_input_lower = user_input.lower()

    if user_input_lower.startswith("add "):
        # 提取 "add " 后面的任务描述
        task_description = user_input[len("add "):].strip()
        if task_description:
            return add_task, (task_description,)
        else:
            return lambda: "Please tell me what task to add.", () # 返回一个返回字符串的函数

    elif user_input_lower == "view tasks" or user_input_lower == "show tasks" or user_input_lower == "list tasks":
        return view_tasks, () # view_tasks 不需要参数

    elif user_input_lower.startswith("remove "):
        task_identifier = user_input[len("remove "):].strip()
        if task_identifier:
            return remove_task, (task_identifier,)
        else:
            return lambda: "Please tell me which task to remove (e.g., 'remove 1').", ()

    elif user_input_lower in ["exit", "quit", "bye"]:
        return "exit_command", None

    else:
        # 如果命令无法识别
        unrecognized_response = (
            "Sorry, I didn't understand that. You can:\n"
            "- Add a task: 'add buy milk'\n"
            "- View tasks: 'view tasks'\n"
            "- Remove a task: 'remove 1' (use the number from 'view tasks')\n"
            "- Exit: 'exit' or 'quit'"
        )
        return lambda: unrecognized_response, ()
```

在 `process_user_command` 函数中：

- 我们将输入转换为小写，使关键词匹配不区分大小写。
- 我们使用 `startswith()` 和直接比较来识别意图动作。
- 如果一个动作需要参数 (parameter)（例如 `add_task` 需要 `task_description`），我们会从输入字符串中提取它。
- 该函数返回两项内容：要调用的实际函数（例如 `add_task`）以及该函数的参数元组（例如 `(task_description,)`）。
- 如果输入无法识别或不完整，它会返回一个lambda函数，该函数在被调用时提供有用的反馈或错误消息。
- 我们还添加了一个“退出命令”，以便用户停止代理。

这个 `process_user_command` 函数是一个占位符，用于替代LLM会提供的更复杂的推理 (inference)能力。一个真实的LLM，在精心设计的提示引导下，在理解这些命令的各种表达方式方面会更加灵活。

### 代理的运行循环

现在，让我们为代理创建主循环。这个循环将：

1. 问候用户。
2. 获取用户输入。
3. 将输入传递给我们的 `process_user_command` 函数。
4. 执行所选动作。
5. 显示结果。
6. 重复直到用户希望退出。

这是我们在第二章中讨论的“观察、思考、行动”循环的简化版本。

```python
def run_todo_agent():
    """运行待办事项代理的主循环。"""
    print("Hi! I'm your To-Do List Agent. How can I help you today?")
    print("You can 'add [task]', 'view tasks', 'remove [task number]', or 'exit'.")

    while True:
        try:
            user_input = input("> ")

            action_function, action_args = process_user_command(user_input)

            if action_function == "exit_command":
                print("Goodbye!")
                break

            if action_function:
                if action_args:
                    response = action_function(*action_args)
                else:
                    response = action_function()
                print(response)
            else:
                # 这种情况理想情况下应在 process_user_command 内部处理
                # 通过为无法识别的命令返回一个lambda函数。
                print("I'm not sure how to handle that. Please try again.")

        except Exception as e:
            # 循环过程中出现意外问题的基本错误捕获
            print(f"An unexpected error occurred: {e}")
            print("Let's try that again.")

# 启动代理：
if __name__ == "__main__":
    run_todo_agent()
```

当你运行这个Python脚本时：

- `run_todo_agent` 函数启动。
- 它进入一个无限的 `while True` 循环，用 `>`  提示用户输入。
- 输入由 `process_user_command` 处理。
- 如果检测到“退出命令”，循环中断。
- 否则，返回的 `action_function` 会与它的 `action_args`（如果有的话）一起被调用。
- 动作函数返回的 `response`（例如“任务已添加”或任务列表）会被打印出来。

### 运行你的代理：一个交互示例

让我们想象一下运行这个脚本。下面是交互可能的样子：

```text
Hi! I'm your To-Do List Agent. How can I help you today?
You can 'add [task]', 'view tasks', 'remove [task number]', or 'exit'.
> add Buy groceries
Okay, I've added 'Buy groceries' to your list.
> add Call the plumber
Okay, I've added 'Call the plumber' to your list.
> view tasks
Here are your tasks:
1. Buy groceries
2. Call the plumber
> remove 1
Okay, I've removed 'Buy groceries'.
> view tasks
Here are your tasks:
1. Call the plumber
> remove Call the plumber
Please tell me the number of the task you want to remove.
> remove 1
Okay, I've removed 'Call the plumber'.
> view tasks
Your to-do list is empty!
> something random
Sorry, I didn't understand that. You can:
- Add a task: 'add buy milk'
- View tasks: 'view tasks'
- Remove a task: 'remove 1' (use the number from 'view tasks')
- Exit: 'exit' or 'quit'
> exit
Goodbye!
```

这个交互展示了代理如何添加任务、查看任务以及根据编号移除任务。它还展示了我们的简单命令处理器如何处理无法识别的输入并提供帮助。你会注意到，尝试通过名称（“remove Call the plumber”）移除任务对我们当前的 `remove_task` 函数不起作用，因为它需要一个数字。这是一个你可能会遇到并需要改进的“初步问题”的小例子。一个使用LLM的更高级的代理可能能够更好地推断用户意图或请求澄清。

### 真正的LLM如何融入

在我们当前的 `process_user_command` 函数中，我们使用了基于关键词的简单 `if/elif` 语句。一个真正的LLM驱动的代理会以不同的方式处理这个问题：

1. **系统提示**：你会有一个“系统提示”或一套给LLM的初始指令。对于我们的待办事项代理，它可能是这样的：
   “你是一个有用的待办事项列表助手。你可用的动作是：`add_task(description)`、`view_tasks()`、`remove_task(task_number)`。当用户要求做某事时，识别正确的动作和任何参数 (parameter)。如果用户要求按名称移除任务，请首先使用 `view_tasks` 查找其编号，然后使用 `remove_task(task_number)`。”
2. **LLM调用**：`process_user_command` 函数会将用户的输入（例如“add pick up dry cleaning”）和这个系统提示发送给LLM API。
3. **LLM响应**：LLM理想情况下会以结构化输出响应，可能采用JSON格式，例如：
   `{"action": "add_task", "parameters": {"description": "pick up dry cleaning"}}`
   或者对于查看任务：
   `{"action": "view_tasks", "parameters": {}}`
4. **动作分发**：你的Python代码随后会解析这个JSON，并使用提取的参数调用相应的Python函数（`add_task` 或 `view_tasks`）。

这种方法使代理在处理自然语言方面更加灵活，并且能够进行更复杂的推理 (inference)，我们将在后续内容中进行讲解。

### 总结与后续步骤

恭喜！你刚刚完成了创建一个非常基础的代理。尽管我们简化了LLM的角色，但你已经了解了如何：

- 为代理定义特定动作（工具）（我们的Python函数）。
- 创建一个机制（即使是简单的）来解释用户意图。
- 实现一个运行循环（观察、“思考”、行动）。
- 执行并查看代理的行为。

这个待办事项代理构成了一个坚实的基础。随着你学习本课程的深入，你将了解到如何用实际的LLM调用替换简化的 `process_user_command`，为你的代理提供更精密的工具，使它们能够在不同会话之间记住信息，并帮助它们规划更复杂的动作序列。目前，尝试修改这个代理：你能添加一个功能，将任务标记 (token)为“已完成”而不是仅仅移除它们吗？或者允许编辑任务？实践是一个很好的学习方式！

## 参考资料

- [Agents](https://python.langchain.com/docs/modules/agents/) — LangChain (2024)
  提供了使用LangChain框架构建LLM代理的实用概述和实现细节，包含各种工具使用模式。
