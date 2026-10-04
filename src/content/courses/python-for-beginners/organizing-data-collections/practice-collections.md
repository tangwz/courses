---
course: "python-for-beginners"
chapter: "organizing-data-collections"
lesson: "practice-collections"
sourceId: 2326
sourceUrl: "https://apxml.com/zh/courses/python-for-beginners/chapter-4-organizing-data-collections/practice-collections"
title: "动手实践：使用集合"
description: "练习使用列表、元组、字典和集合来解决常见的数据组织问题。"
order: 9
plots: []
sourceHash: "bc7d11e2ac4289c9b45a308445d2fe445a07b0b41f7024a5fa8e63baa811799f"
sourceCorrections: []
---

提供实践示例以巩固你对创建、操作和使用基本Python数据结构（列表、元组、字典和集合）的理解。请跟着做，并在你的Python解释器或脚本中尝试运行这些示例。

### 列表的使用

列表是有序的、可变的集合。它们可能是你遇到的最常见的集合类型。让我们管理一个任务列表。

1. **创建列表：**
   从创建一个表示任务的字符串列表开始。

   ```python
   tasks = ["Review Chapter 3", "Start Chapter 4", "Practice list operations"]
   print(tasks)
   # Output: ['Review Chapter 3', 'Start Chapter 4', 'Practice list operations']
   ```
2. **访问元素：**
   使用索引访问元素（请记住索引从0开始）。

   ```python
   first_task = tasks[0]
   print(f"First task: {first_task}")
   # Output: First task: Review Chapter 3

   last_task = tasks[-1] # 负索引从末尾开始计数
   print(f"Last task: {last_task}")
   # Output: Last task: Practice list operations
   ```
3. **添加元素：**
   使用 `append()` 在末尾添加项目，或使用 `insert()` 在指定位置添加。

   ```python
   tasks.append("Take a break")
   print(tasks)
   # Output: ['Review Chapter 3', 'Start Chapter 4', 'Practice list operations', 'Take a break']

   tasks.insert(1, "Read list documentation") # 在索引1处插入
   print(tasks)
   # Output: ['Review Chapter 3', 'Read list documentation', 'Start Chapter 4', 'Practice list operations', 'Take a break']
   ```
4. **删除元素：**
   使用 `remove()` 删除值的首次出现，或使用 `del` 按索引删除。

   ```python
   tasks.remove("Take a break") # 按值删除
   print(tasks)
   # Output: ['Review Chapter 3', 'Read list documentation', 'Start Chapter 4', 'Practice list operations']

   del tasks[0] # 删除索引0处的项目
   print(tasks)
   # Output: ['Read list documentation', 'Start Chapter 4', 'Practice list operations']
   ```
5. **遍历列表：**
   使用 `for` 循环处理列表中的每个项目。

   ```python
   print("\n剩余任务：")
   for task in tasks:
       print(f"- {task}")
   # Output:
   # 剩余任务：
   # - Read list documentation
   # - Start Chapter 4
   # - Practice list operations
   ```

### 元组的使用

元组是有序的、不可变的集合。一旦创建，其内容不能更改。它们常用于固定集合的项目，例如坐标或RGB颜色值。

1. **创建元组：**
   元组使用括号 `()` 定义。

   ```python
   point = (10, 20) # 表示一个 (x, y) 坐标
   rgb_color = (255, 0, 0) # 红色

   print(f"点坐标： {point}")
   # Output: Point coordinates: (10, 20)
   print(f"RGB 颜色： {rgb_color}")
   # Output: RGB Color: (255, 0, 0)
   ```
2. **访问元素：**
   像列表一样访问元素，使用索引。

   ```python
   x_coordinate = point[0]
   red_value = rgb_color[0]

   print(f"X 坐标： {x_coordinate}")
   # Output: X coordinate: 10
   print(f"红色值： {red_value}")
   # Output: Red value: 255
   ```
3. **不可变性：**
   尝试更改元组中的元素会导致错误。这是其一个主要特点。

   ```python
   # 这一行会引起错误：
   # point[0] = 15
   # TypeError: 'tuple' object does not support item assignment
   ```

   不可变性使得元组在表示不应意外更改的数据时更加可靠。它们也可以用作字典的键（与列表不同）。

### 字典的使用

字典以键值对的形式存储数据。它们是无序的（在旧版Python中）或按插入顺序排列的（在新版Python中），并且是可变的。键必须是唯一的且不可变的（例如字符串、数字或元组）。字典提供基于键的快速查找。

1. **创建字典：**
   使用花括号 `{}` 和 `键: 值` 对。

   ```python
   student_scores = {
       "Alice": 85,
       "Bob": 92,
       "Charlie": 78
   }
   print(student_scores)
   # Output: {'Alice': 85, 'Bob': 92, 'Charlie': 78}
   ```
2. **访问值：**
   使用方括号 `[]` 中的键来获取对应的值。

   ```python
   bob_score = student_scores["Bob"]
   print(f"Bob 的分数： {bob_score}")
   # Output: Bob's score: 92
   ```

   尝试访问不存在的键将引发 `KeyError`。你可以使用 `.get()` 方法进行更安全的访问，如果键未找到，它会返回 `None`（或指定的默认值）。

   ```python
   david_score = student_scores.get("David") # 键不存在
   print(f"David 的分数： {david_score}")
   # Output: David's score: None

   david_score_default = student_scores.get("David", "Not found") # 带默认值
   print(f"David 的分数： {david_score_default}")
   # Output: David's score: Not found
   ```
3. **添加或修改条目：**
   为新键赋值以添加条目，或为现有键赋值以修改条目。

   ```python
   # 添加新学生
   student_scores["David"] = 88
   print(student_scores)
   # Output: {'Alice': 85, 'Bob': 92, 'Charlie': 78, 'David': 88}

   # 更新 Alice 的分数
   student_scores["Alice"] = 87
   print(student_scores)
   # Output: {'Alice': 87, 'Bob': 92, 'Charlie': 78, 'David': 88}
   ```
4. **删除条目：**
   使用 `del` 按键删除条目。

   ```python
   del student_scores["Charlie"]
   print(student_scores)
   # Output: {'Alice': 87, 'Bob': 92, 'David': 88}
   ```
5. **遍历字典：**
   你可以遍历键、值或键值对（项目）。

   ```python
   print("\n学生分数：")
   # 遍历键（默认遍历方式）
   for student in student_scores:
       print(f"- {student}: {student_scores[student]}")

   print("\n使用 .items()：")
   # 遍历键值对
   for student, score in student_scores.items():
       print(f"- {student} scored {score}")

   # Output:
   # 学生分数：
   # - Alice: 87
   # - Bob: 92
   # - David: 88
   #
   # 使用 .items()：
   # - Alice 得分 87
   # - Bob 得分 92
   # - David 得分 88
   ```

### 集合的使用

集合是无序的、包含唯一且不可变项目的集合。它们可用于成员资格测试、去除重复项以及执行数学集合操作。

1. **创建集合：**
   使用花括号 `{}` 或 `set()` 函数。请注意，`{}` 会创建一个空字典，因此对于空集合，请使用 `set()`。

   ```python
   # 从包含重复项的列表创建
   tags_list = ["python", "data", "web", "python", "data"]
   unique_tags = set(tags_list)
   print(unique_tags)
   # Output: {'data', 'python', 'web'} (顺序可能不同)

   # 直接创建
   allowed_users = {"alice", "bob", "charlie"}
   print(allowed_users)
   # Output: {'charlie', 'alice', 'bob'} (顺序可能不同)
   ```
2. **添加元素：**
   使用 `add()` 方法。重复项会自动被忽略。

   ```python
   allowed_users.add("david")
   print(allowed_users)
   # Output: {'charlie', 'david', 'alice', 'bob'}

   allowed_users.add("alice") # 添加现有元素没有效果
   print(allowed_users)
   # Output: {'charlie', 'david', 'alice', 'bob'}
   ```
3. **成员资格测试：**
   使用 `in` 运算符检查元素是否存在于集合中。这效率很高。

   ```python
   if "bob" in allowed_users:
       print("Bob 是允许的用户。")
   else:
       print("Bob 不被允许。")
   # Output: Bob is an allowed user.

   if "eve" in allowed_users:
       print("Eve 是允许的用户。")
   else:
       print("Eve 不被允许。")
   # Output: Eve is not allowed.
   ```
4. **集合操作：**
   集合支持并集 (`|`)、交集 (`&`)、差集 (`-`) 和对称差集 (`^`) 等操作。

   ```python
   admin_users = {"alice", "eve"}

   # 并集：存在于任一集合中的所有用户
   all_users = allowed_users | admin_users
   print(f"所有用户： {all_users}")
   # Output: All users: {'eve', 'charlie', 'david', 'alice', 'bob'}

   # 交集：同时存在于两个集合中的用户
   common_users = allowed_users & admin_users
   print(f"共同用户（被允许的管理员）： {common_users}")
   # Output: Common users (admins who are allowed): {'alice'}

   # 差集：存在于 allowed_users 但不在 admin_users 中的用户
   non_admin_allowed = allowed_users - admin_users
   print(f"被允许但不是管理员的用户： {non_admin_allowed}")
   # Output: Allowed users who are not admins: {'charlie', 'david', 'bob'}
   ```

### 选择合适的结构：一个小例子

设想你需要在一个图书馆目录中存储图书信息。

- 为了存储每本书的ISBN（唯一标识符）和标题，**字典**是合适的：`books_by_isbn = {"978-0134": "Python Crash Course", ...}`。这允许按ISBN快速查找。
- 为了存储特定用户借阅图书的序列，并按借阅顺序排列，**列表**是合适的：`borrow_history = ["Python Crash Course", "Fluent Python", ...]`。顺序很重要，并且列表可以随着借阅更多图书而增长。
- 为了存储一组固定的类别，例如（'小说', '科学', '历史'），如果你不期望在程序执行期间类别经常更改，则可以使用**元组**。
- 为了快速检查一本书是否属于某个特殊收藏（例如，“珍本”），该收藏的ISBN**集合**将是高效的：`rare_book_isbns = {"978-...", "978-..."}`。检查 `isbn in rare_book_isbns` 速度很快。

本次实践涵盖了列表、元组、字典和集合的创建、访问、修改和遍历。试验这些示例是熟练有效使用Python集合类型的最佳方式。记住在决定使用哪种类型来满足你的特定数据组织需求时，要考虑每种类型的特点：顺序、可变性、唯一性、键值映射。

## 参考资料

- [Built-in Types](https://docs.python.org/3/library/stdtypes.html) — Python Software Foundation (2024)
  作为官方Python文档的一部分，涵盖所有内置类型的基本特性和操作，包括列表、元组、字典和集合。
- [Python Crash Course, 2nd Edition: A Hands-On, Project-Based Introduction to Programming](https://www.oreilly.com/library/view/python-crash-course/9781492071266/) — Eric Matthes (2019)
  Publisher: No Starch Press; Pages: 544
  一本实用的Python编程入门书，通过清晰的解释和示例介绍了列表、元组、字典和集合等基本数据结构。
