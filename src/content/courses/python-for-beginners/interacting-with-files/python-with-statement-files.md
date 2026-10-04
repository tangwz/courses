---
course: "python-for-beginners"
chapter: "interacting-with-files"
lesson: "python-with-statement-files"
sourceId: 2348
sourceUrl: "https://apxml.com/zh/courses/python-for-beginners/chapter-6-interacting-with-files/python-with-statement-files"
title: "使用 `with` 自动关闭文件"
description: "了解处理文件的推荐方式：使用 `with` 语句，这种方式确保文件自动关闭。"
order: 6
plots: []
sourceHash: "e6d37c4179b6203fc5bbb9d8650b1e6ce738dd5c5d89f1f7df1b62094df6bcab"
sourceCorrections: []
---

在Python中，通常使用 `open()` 函数打开文件，然后使用 `file.close()` 方法手动关闭它们。尽管这种方法可以实现基本的文件操作，但它会带来一个潜在问题：如果程序在文件打开之后但在调用 `close()` 方法之前遇到错误，会发生什么？在这种情况下，文件可能会保持打开状态，可能导致数据损坏或不必要地占用系统资源。

忘记关闭文件，或者因错误阻止 `close()` 方法被调用，是一个常见问题。Python 提供了一种更可靠、更简洁的方式来使用 `with` 语句处理文件。

`with` 语句确保资源（如文件）得到妥善管理。当你使用 `with` 打开文件时，Python 保证无论 `with` 语句下的代码块是正常结束还是因错误而结束，文件都会在代码块执行完毕后自动关闭。

### `with` 语句的语法

使用 `with` 打开文件的基本结构如下：

```python
with open('path/to/your/file.txt', 'mode') as file_variable:
    # 在这里编写文件操作代码
    # 使用 file_variable 进行读写
    # 例如：content = file_variable.read()
    pass # 将 pass 替换为你的文件操作

# with 代码块结束后，程序会从这里继续执行
# 至此，文件已自动关闭
```

让我们分解一下这个结构：

1. `with`：这个关键字表示上下文 (context)管理块的开始。
2. `open('path/to/your/file.txt', 'mode')`：这是大家熟悉的 `open()` 函数调用，指定文件路径和模式（例如 `'r'` 表示读取，`'w'` 表示写入）。
3. `as file_variable`：`open()` 函数返回一个文件对象。`as` 关键字将此对象赋值给一个变量（这里是 `file_variable`），你可以在 `with` 块内部使用它。你可以选择任何有效的变量名。
4. 缩进的代码块：`with` 语句下所有缩进的代码都在文件打开期间执行。你使用 `file_variable` 来执行读写等操作。
5. 自动关闭：一旦代码执行离开缩进块（无论是正常到达末尾还是因为错误），Python 会自动调用文件对象上的必要清理操作，有效地关闭文件。你*不需要*自己调用 `file_variable.close()`。

### 使用 `with` 读取文件

这是一个使用 `with` 语句读取名为 `example.txt` 文件内容的例子：

```python
# 假设 'example.txt' 包含这一行: "Hello from the file!"

try:
    with open('example.txt', 'r') as f:
        content = f.read()
        print("文件内容读取成功:")
        print(content)
    # 文件现在已自动关闭
    print("文件已关闭。")

except FileNotFoundError:
    print("错误：未找到文件 'example.txt'。")
except Exception as e:
    print(f"发生意外错误: {e}")
    # 如果文件在错误发生前已打开，它也会自动关闭
```

在这个例子中，`example.txt` 以读取模式 (`'r'`) 打开。文件对象被赋值给变量 `f`。我们在 `with` 块内部读取其内容。无论 `read()` 操作是否成功，或者块内是否发生其他错误，Python 保证在执行完此块之前，会隐式调用 `f.close()`。

### 使用 `with` 写入文件

`with` 语句对于写入操作也同样有效：

```python
lines_to_write = ["First line.\n", "Second line.\n", "Third line.\n"]

try:
    with open('output.txt', 'w') as outfile:
        outfile.writelines(lines_to_write)
        print("数据已成功写入 output.txt。")
    # 文件现在已自动关闭

    # 让我们通过再次读取来验证（再次使用 'with'！）
    with open('output.txt', 'r') as infile:
        print("\n验证文件内容:")
        print(infile.read())

except IOError as e:
    print(f"文件读写时发生错误: {e}")
except Exception as e:
    print(f"发生意外错误: {e}")
```

这里，`output.txt` 以写入模式 (`'w'`) 打开。我们使用 `writelines()` 写入多行。一旦代码退出 `with` 块，`output.txt` 就会关闭，确保所有缓冲数据写入磁盘。然后我们使用另一个 `with` 语句安全地重新打开它以读取和验证内容。

### 为什么要使用 `with`？

在 Python 中，使用 `with` 语句被认为是处理文件的标准且推荐的方式，因为：

1. **安全性**：它保证文件能够被妥善关闭，即使发生错误，也能避免资源泄露和潜在的数据损坏。
2. **可读性**：它使代码更简洁、更易懂。块结构清楚地定义了文件打开的范围。
3. **便捷性**：它消除了仅为关闭文件而编写显式 `try...finally` 块的需要，减少了样板代码。

尽管你可能最初是分别学习 `open()` 和 `close()`，但请养成在所有文件操作中使用 `with` 语句的习惯。这是一种更简单、更安全、更符合 Python 风格的方法。

## 参考资料

- [\`open()\` function](https://docs.python.org/3/library/functions.html#open) — Python Software Foundation (2023)
  此官方 Python 文档页面描述了内置的 `open()` 函数，并明确推荐将其与 `with` 语句结合使用，以进行正确的资源管理。
- [The \`with\` statement](https://docs.python.org/3/reference/compound_stmts.html#the-with-statement) — Python Software Foundation (2023)
  此官方 Python 文档参考资料解释了 `with` 语句的语法和目的，强调了其在文件等资源上下文管理中的作用。
- [Learning Python](https://www.oreilly.com/library/view/learning-python-5th/9781449355739/) — Mark Lutz (2013)
  Publisher: O'Reilly Media
  这本权威教材全面解释了 Python 文件 I/O，包括 `with` 语句及其在可靠资源管理中的优势。第五版。
- [Fluent Python: Clear, Concise, and Effective Programming](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHa7p2iAj1EmgNwWE9DqHMZI_5kbaSDZQrYNSqZBakFy1f_cAzdL_LHX92TiQm7ykFlDAF44ewKwdAOcSUnGSoxGbqkFZrZrSJRflB6zJHE3Ai5jnu_Ixm1swGUiPlJw3TbUjbZRasA_M3RX5wwiAWYt_3AuGi4oQ87YL35bOC9w7zXgs9a3wpR4dDtYASq) — Luciano Ramalho (2022)
  Publisher: O'Reilly Media
  本书讨论了 Python 式的方法，解释了 `with` 语句如何利用上下文管理器协议进行自动资源处理。第二版。
