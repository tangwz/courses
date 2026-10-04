---
course: "python-for-beginners"
chapter: "putting-it-all-together"
lesson: "writing-project-code"
sourceId: 2395
sourceUrl: "https://apxml.com/zh/courses/python-for-beginners/chapter-10-putting-it-all-together/writing-project-code"
title: "编写代码：一步一步"
description: "逐步完成命令行工具示例项目的实现过程。"
order: 5
plots: []
sourceHash: "7863955cf3a4bec2923bd0c3060d0d5e9d2a7468d642d72d8dea4bfd370cfedf"
sourceCorrections: []
---

编写一个简单的命令行应用程序的 Python 代码。应用程序将逐步构建，并应用编程知识。例如，将创建一个非常基础的“待办事项列表”应用程序。

记住我们概述的计划吗？我们需要添加任务、查看任务、将任务标记 (token)为已完成以及退出等功能。我们还决定了数据的存储结构。

### 1. 设置文件和初始数据结构

首先，创建一个新的 Python 文件。让我们将其命名为 `todo_app.py`。

在这个文件中，我们需要一种方式来存储任务。列表是个不错的选择，其中列表中的每个项代表一个任务。我们应该如何表示一个任务呢？字典看起来很合适，它可以让我们存储任务描述及其状态（例如，“待处理”或“已完成”）。

```python
# todo_app.py

# 从一个空列表开始，用于存放任务。
# 每个任务都将是一个字典，例如：{'description': '做作业', 'status': '待处理'}
tasks = []
```

这奠定了基础。`tasks` 目前是一个空列表，准备好存放我们的任务字典。

### 2. 构建核心函数

现在，让我们为应用程序需要执行的每个操作创建函数。这种方法使我们的代码模块化且更易于理解，体现了第 5 章（函数）中的原则。

#### 添加任务

我们需要一个函数，它会要求用户输入任务描述，并将其以“待处理”状态添加到我们的 `tasks` 列表中。

```python
def add_task(task_list):
    """要求用户输入任务描述并将其添加到列表中。"""
    description = input("输入任务描述：")
    new_task = {'description': description, 'status': 'pending'}
    task_list.append(new_task)
    print(f"任务 '{description}' 已添加。")

# 使用示例（我们稍后会将其集成到主循环中）
# add_task(tasks)
# print(tasks)
```

此函数将主 `task_list` 作为参数 (parameter)，获取用户输入，创建字典，并使用 `append` 方法（来自第 4 章“列表”）将其添加。

#### 查看任务

我们需要一种方式来显示所有任务，或许可以加上编号，以便用户可以方便地引用它们。我们还应该处理尚未有任务的情况。

```python
def view_tasks(task_list):
    """显示所有任务及其索引和状态。"""
    print("\n--- 您的任务 ---")
    if not task_list:
        print("暂无任务！")
        return

    # enumerate 提供索引和项，在此处很有用。
    for index, task in enumerate(task_list):
        # 为方便用户，索引从 1 开始显示
        print(f"{index + 1}. {task['description']} [{task['status']}]")
    print("------------------")

# 使用示例：
# view_tasks(tasks) # 最初将显示“暂无任务！”
# add_task(tasks)
# view_tasks(tasks)
```

此函数遍历 `task_list`。它使用 `enumerate` 来同时获取索引（从 0 开始）和任务字典。我们为显示目的将索引加 1，使其对用户来说更自然（1、2、3……）。在尝试循环之前，它会检查列表是否为空。

#### 将任务标记 (token)为已完成

此函数需要显示任务，询问用户要将哪个任务标记为已完成，验证输入，并更新所选任务字典中的状态。这涉及到使用第 3 章（控制流）和可能第 9 章（错误处理）中的知识。

```python
def mark_complete(task_list):
    """将特定任务标记为已完成。"""
    view_tasks(task_list) # 先显示任务，以便用户选择

    if not task_list: # 如果列表为空，则无法标记为已完成
        return

    while True: # 循环直到接收到有效输入
        try:
            task_number_str = input("输入要标记为已完成的任务编号：")
            # 将输入字符串转换为整数（第 2 章）
            task_index = int(task_number_str) - 1 # 调整为基于 0 的索引

            # 检查索引是否有效（第 3 章，条件语句）
            if 0 <= task_index < len(task_list):
                # 直接修改列表中的字典（第 4 章）
                task_list[task_index]['status'] = 'complete'
                print(f"任务 {task_index + 1} 已标记为已完成。")
                break # 退出输入循环
            else:
                print("任务编号无效。请重试。")
        except ValueError:
            # 处理输入不是数字的情况（第 9 章）
            print("输入无效。请输入一个数字。")

# 使用示例：
# add_task(tasks) # 先添加一个任务
# mark_complete(tasks)
# view_tasks(tasks)
```

在这里，我们首先显示任务。然后，我们要求用户输入编号。我们使用 `while True` 循环结合 `try-except`（第 9 章）来处理用户输入非数字时可能出现的 `ValueError`。我们还会检查该数字是否对应列表中的有效索引。如果输入有效，我们通过其索引 (`task_list[task_index]`) 访问特定的任务字典，并更改与 `'status'` 键关联的值。然后我们 `break`（跳出）输入循环。

### 3. 创建主应用程序循环

现在我们需要应用程序的主体部分，它会持续运行，向用户显示选项菜单，获取他们的选择，并调用相应的函数。这使用了 `while` 循环和 `if/elif/else` 语句（第 3 章）。

```python
def display_menu():
    """打印菜单选项。"""
    print("\n--- 待办事项列表菜单 ---")
    print("1. 添加任务")
    print("2. 查看任务")
    print("3. 将任务标记为已完成")
    print("4. 退出")
    print("-----------------------")

# --- 主应用程序执行 ---
if __name__ == "__main__": # 标准做法，稍后简要解释
    print("欢迎使用简单待办事项列表！")

    while True: # 主应用程序循环
        display_menu()
        choice = input("输入您的选择 (1-4)：")

        if choice == '1':
            add_task(tasks)
        elif choice == '2':
            view_tasks(tasks)
        elif choice == '3':
            mark_complete(tasks)
        elif choice == '4':
            print("正在退出待办事项列表应用程序。再见！")
            break # 退出 while 循环
        else:
            print("无效选择。请输入 1 到 4 之间的一个数字。")

    # 可选：在此处添加退出前将任务保存到文件的功能
    # print("任务已保存。") # 占位符
```

`display_menu` 函数只是打印出选项。主体部分在一个 `while True` 循环中运行。它显示菜单，获取用户的 `choice`（选择），并使用 `if/elif/else` 来决定调用哪个函数。如果选择是“4”，它会打印一条告别消息并使用 `break` 来终止循环，从而结束程序。一个 `else` 子句处理无效输入。

`if __name__ == "__main__":` 这一行是 Python 中常见的结构。这意味着此代码块中的代码仅在脚本直接执行时运行（而不是作为模块导入到另一个脚本时运行，这一点我们在第 7 章中讨论过）。对于我们这个简单的应用程序，它标志着执行的起始点。

### 4. 整合所有代码（完整代码）

让我们将所有部分组合到最终的 `todo_app.py` 文件中：

```python
# todo_app.py

# 从一个空列表开始，用于存放任务。
tasks = []

def add_task(task_list):
    """要求用户输入任务描述并将其添加到列表中。"""
    description = input("输入任务描述：")
    # 基本验证：确保描述不为空
    if description.strip(): # strip() 移除开头/结尾的空白字符
        new_task = {'description': description, 'status': 'pending'}
        task_list.append(new_task)
        print(f"任务 '{description}' 已添加。")
    else:
        print("任务描述不能为空。")

def view_tasks(task_list):
    """显示所有任务及其索引和状态。"""
    print("\n--- 您的任务 ---")
    if not task_list:
        print("暂无任务！")
        return

    for index, task in enumerate(task_list):
        print(f"{index + 1}. {task['description']} [{task['status']}]")
    print("------------------")

def mark_complete(task_list):
    """将特定任务标记为已完成。"""
    view_tasks(task_list)

    if not task_list:
        return

    while True:
        try:
            task_number_str = input("输入要标记为已完成的任务编号（输入 0 取消）：")
            task_number = int(task_number_str)

            if task_number == 0: # 允许用户取消
                break

            task_index = task_number - 1

            if 0 <= task_index < len(task_list):
                # 检查是否已完成以避免不必要的更新
                if task_list[task_index]['status'] == 'complete':
                    print(f"任务 {task_number} 已经标记为已完成。")
                else:
                    task_list[task_index]['status'] = 'complete'
                    print(f"任务 {task_number} 已标记为已完成。")
                break
            else:
                print("任务编号无效。请重试。")
        except ValueError:
            print("输入无效。请输入一个数字。")

def display_menu():
    """打印菜单选项。"""
    print("\n--- 待办事项列表菜单 ---")
    print("1. 添加任务")
    print("2. 查看任务")
    print("3. 将任务标记为已完成")
    print("4. 退出")
    print("-----------------------")

# --- 主应用程序执行 ---
if __name__ == "__main__":
    print("欢迎使用简单待办事项列表！")
    # 未来增强：在此处从文件加载任务

    while True:
        display_menu()
        choice = input("输入您的选择 (1-4)：")

        if choice == '1':
            add_task(tasks)
        elif choice == '2':
            view_tasks(tasks)
        elif choice == '3':
            mark_complete(tasks)
        elif choice == '4':
            # 未来增强：在此处将任务保存到文件
            print("正在退出待办事项列表应用程序。再见！")
            break
        else:
            print("无效选择。请输入 1 到 4 之间的一个数字。")
```

### 运行代码

将此代码保存为 `todo_app.py`。打开您的终端或命令提示符，导航到您保存文件的目录，然后使用以下命令运行它：

```bash
python todo_app.py
```

您应该会看到消息和菜单。尝试添加任务、查看任务和将任务标记 (token)为已完成。

这个循序渐进的过程展示了如何通过以下方式构建一个简单的应用程序：

1. 定义数据结构（字典列表）。
2. 将功能分解为单独的函数（`add_task`、`view_tasks`、`mark_complete`）。
3. 使用控制流（`while`、`if/elif/else`）来管理应用程序的执行和用户交互。
4. 实施基本的错误处理（`try-except`）以提高代码的健壮性。

尽管基础，但这种结构构成了许多命令行工具的基础，并整合了本课程中涵盖的许多知识点。下一节将讨论测试这个应用程序。

## 参考资料

- [The Python Tutorial](https://docs.python.org/3/tutorial/) — Guido van Rossum and the Python development team (2024)
  提供了Python核心语言功能的介绍，包括数据结构（列表、字典）、控制流、函数和错误处理，对于理解应用程序的构建模块至关重要。
- [Python Crash Course, 3rd Edition: A Hands-On, Project-Based Introduction to Programming](https://nostarch.com/python-crash-course-3rd-edition/) — Eric Matthes (2022)
  Publisher: No Starch Press
  一本针对初学者的实践指南，展示了如何应用Python基础概念构建实际项目，与本节中循序渐进的应用程序开发方法一致。
- [Google's Python Class](https://developers.google.com/edu/python/) — Nick Parlante (2010)
  Publisher: Google Developers
  提供免费且全面的Python基础课程，内容涵盖基本语法、数据类型、控制结构和函数定义，这些都在示例应用程序中得到了应用。
