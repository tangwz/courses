---
course: "python-for-beginners"
chapter: "interacting-with-files"
lesson: "python-opening-closing-files"
sourceId: 2342
sourceUrl: "https://apxml.com/zh/courses/python-for-beginners/chapter-6-interacting-with-files/python-opening-closing-files"
title: "文件的打开与关闭"
description: "了解如何使用内置的 `open()` 函数打开文件以及关闭它们的重要性。"
order: 2
plots: []
sourceHash: "3d6c6b65b08f0951c73382c37b643c868d5cfe86ebeadf35fdc2ab8836cecda0"
sourceCorrections: []
---

要从计算机磁盘上的文件读取或向其写入数据，你首先需要从 Python 脚本建立与该文件的连接。此过程称为“打开”文件。Python 提供了一个内置函数 `open()`，专门用于此目的。

### 使用 `open()` 函数

`open()` 函数使用简单。最基本的使用方式是，它需要你希望操作的文件名（可能还有路径）以及一个模式，用于指定你打算对文件做什么（例如从文件读取或向文件写入）。

基本语法如下所示：

```python
file_object = open('filename', 'mode')
```

我们来分解一下这些参数 (parameter)：

1. **`filename`**：这是一个字符串，表示你希望打开的文件名。如果文件与你的 Python 脚本在同一目录中，你只需提供文件名（例如 `'notes.txt'`）。如果文件位于其他地方，你将需要提供其路径。路径的指定方式取决于你的操作系统（如在“理解文件路径”部分所讨论的）。例如，你可以在 Windows 上使用 `'data/config.txt'` 作为相对路径，或者使用完整的路径，如 `'C:/Users/username/Documents/report.txt'`；在 Linux/macOS 上则可以使用 `'/home/user/documents/report.txt'`。
2. **`mode`**：这是一个字符串，表明文件将如何使用。模式决定了你被允许执行哪些操作（例如读取或写入），以及文件在已经存在或不存在时如何处理。以下是文本文件最常用的三种模式：

   - `'r'` (读取模式)：如果你不指定模式，这是默认模式。它允许你从现有文件读取数据。如果文件不存在，Python 将会引发 `FileNotFoundError`。
   - `'w'` (写入模式)：此模式允许你向文件写入数据。如果文件已经存在，其当前内容在写入开始前将被完全清除。如果文件不存在，将创建一个新的空文件。使用此模式时请谨慎，因为它可能覆盖现有数据。
   - `'a'` (追加模式)：此模式允许你将新数据添加到现有文件的*末尾*。如果文件存在，新数据会简单地附加到末尾。如果文件不存在，则会创建一个新文件（类似于写入模式，但如果文件*确实*存在，则不会清除现有内容）。

例如，要以读取模式打开名为 `config.txt` 的文件：

```python
config_file = open('config.txt', 'r')
# 或者简单地，因为 'r' 是默认值：
# config_file = open('config.txt')
```

要以追加新条目的模式打开（或创建）名为 `log.txt` 的文件：

```python
log_file = open('log.txt', 'a')
```

要以写入模式打开（或创建并覆盖）名为 `output.txt` 的文件：

```python
output_file = open('output.txt', 'w')
```

### 文件对象

成功调用 `open()` 后，它会返回一个*文件对象*（有时也称为文件句柄）。此对象充当 Python 与磁盘上实际文件的连接。你将使用与此文件对象相关联的方法来执行操作，例如从文件读取数据或向其中写入数据。在上面的例子中，`config_file`、`log_file` 和 `output_file` 都是持有这些文件对象的变量。

### 关闭文件的必要性

完成对文件的操作（读取、写入或追加）后，*关闭*它非常重要。关闭文件会告诉操作系统你已完成文件使用，这有几个作用：

1. **释放系统资源：** 操作系统对用户或程序可以同时打开的文件数量有限制。关闭不再使用的文件会释放这些资源。
2. **确保数据写入：** 当你向文件写入数据时，为了提高效率，数据最初可能会保存在一个名为缓冲区的临时存储区域。关闭文件可确保所有缓冲数据被实际写入磁盘。未能关闭文件，尤其是在写入后，可能导致文件中的数据不完整或丢失。
3. **防止数据损坏：** 不必要地让文件保持打开状态，特别是当你的程序意外终止时，有时可能导致数据损坏问题。

### 使用 `close()` 方法

Python 文件对象有一个 `close()` 方法，在你使用完文件后必须调用它。

这是一个简单的模式：

```python
# 以写入模式打开文件
output_file = open('output.txt', 'w')

# 执行写入操作（将在后续章节介绍）
output_file.write('This is the first line.\n')
output_file.write('This is the second line.\n')

# 显式关闭文件
output_file.close()
```

务必记住调用 `close()`。但是，存在一个潜在问题：如果在你打开文件*之后*但在到达 `close()` 调用*之前*，你的代码中发生错误怎么办？在这种情况下，文件可能会保持打开状态。Python 提供了一种使用 `with` 语句处理文件打开和关闭的方式，它保证即使发生错误，文件也会自动关闭。我们将在“使用 `with` 语句自动关闭文件”部分介绍这种更推荐的方法。目前，请理解在使用这种基本方法时，打开文件需要对应的 `close()` 调用。

## 参考资料

- [\`open()\` built-in function (Python 3.12.3 Documentation)](https://docs.python.org/3/library/functions.html#open) — Python Software Foundation (2024)
  描述了Python中文件操作的 `open()` 函数、其参数及各种文件模式。
- [Reading and Writing Files (Python 3.12.3 Documentation)](https://docs.python.org/3/tutorial/inputoutput.html#reading-and-writing-files) — Python Software Foundation (2024)
  Publisher: Python Software Foundation
  提供了关于Python文件I/O的官方教程，详细说明了 `open()` 函数、各种文件模式以及显式关闭文件的重要性。
- [Learning Python, 5th Edition](https://www.oreilly.com/library/view/learning-python-5th/9781449355739/) — Mark Lutz (2013)
  Publisher: O'Reilly Media; Pages: 247-280 (Chapter 9: Files and Streams)
  提供了Python文件I/O机制的完整说明，包括 `open()` 函数、文件模式、文件对象以及关闭文件的必要性。
