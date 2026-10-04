---
course: "python-for-beginners"
chapter: "interacting-with-files"
lesson: "python-appending-files"
sourceId: 2347
sourceUrl: "https://apxml.com/zh/courses/python-for-beginners/chapter-6-interacting-with-files/python-appending-files"
title: "追加文件内容"
description: "理解如何向现有文件末尾添加新数据，而无需擦除其当前内容。"
order: 5
plots: []
sourceHash: "9a9c4c049bed1c1a05059cf32cf61584663099f0304c551634dd695305450a24"
sourceCorrections: []
---

有时，你不想替换文件的全部内容；你只想在文件末尾添加新的信息。这在记录事件、更新记录或只是随着时间推移添加更多数据而不想丢失之前保存的内容等任务中很常见。写入用的 `'w'` 模式会先擦除文件现有的内容再写入任何新内容，因此不适用于这些情况。

要向现有文件的末尾添加数据，你需要以**追加模式**打开它。这通过在 `open()` 函数中将 `'a'` 指定为模式参数 (parameter)来完成。

当你以追加模式打开文件时：

1. **如果文件存在：** 文件指针（指示下一次写入操作发生的位置）会定位在文件的末尾。你写入的任何数据都将添加到现有内容之后。
2. **如果文件不存在：** Python 会为你创建一个新的空文件，就像它在 `'w'` 模式下所做的那样。

让我们看看它是如何工作的。假设我们有一个名为 `log.txt` 的文件，内容如下：

```text
Event: System Start
```

现在，我们想向此日志添加另一个事件。我们可以使用追加模式：

```python
# log.txt 已存在，且包含 "Event: System Start\n"

# 以追加模式 ('a') 打开 log.txt
with open('log.txt', 'a') as f:
    # 在文件末尾写入新行
    # 注意：我们添加 '\n' 以确保新条目从新行开始
    f.write("Event: User Login\n")
    f.write("Event: Data Processed\n")

# 'with' 代码块会自动关闭文件

# 让我们验证内容
with open('log.txt', 'r') as f:
    content = f.read()
    print(content)
```

运行此代码后，`log.txt` 文件现在将包含：

```text
Event: System Start
Event: User Login
Event: Data Processed
```

请注意新行是如何在原始内容之后添加的。另外，请观察 `write()` 方法，它与使用 `'w'` 模式时一样，不会自动添加换行符。如果你想让每段追加的数据都显示在单独的行上，你必须在传递给 `write()` 的字符串中明确包含换行符 (`\n`)。

将追加模式 (`'a'`) 与 `with` 语句结合使用是向文件末尾添加信息的标准方式，可确保数据完整性和适当的资源管理。它提供了一种简单的方法来扩展文件，而不会有意外覆盖重要现有数据的风险。

## 参考资料

- [Built-in Functions - open()](https://docs.python.org/3/library/functions.html#open) — Python Software Foundation (2023)
  Python `open()` 内置函数的官方文档，详细说明了其各种模式（包括用于追加的 'a'）以及如何在 Python 中与文件交互。
- [Python Crash Course, 3rd Edition](https://nostarch.com/products/python-crash-course-3rd-edition) — Eric Matthes (2023)
  Publisher: No Starch Press; Pages: 185-204
  Python 文件处理的入门读物，为初学者提供了文件读取、写入和追加数据的实用示例。
- [Learning Python, 5th Edition](https://www.oreilly.com/library/view/learning-python-5th/9781449355739/) — Mark Lutz (2013)
  Publisher: O'Reilly Media; Pages: 277-310
  Python 核心功能的综合指南，详细解释了文件输入/输出操作、不同的文件模式和正确的文件管理技术。
