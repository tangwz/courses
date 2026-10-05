# Julia中数据结构的介绍

来源：[原文](https://apxml.com/zh/courses/getting-started-julia-programming/chapter-4-working-with-collections/introduction-data-structures-julia)

[返回章节目录](README.md) · [返回课程目录](../README.md)

Julia的主要集合类型包括数组（Arrays）、元组（Tuples）、字典（Dictionaries）和集合（Sets）。在查看每种类型的具体细节之前，了解**数据结构**在编程中的普遍作用以及它们为何对编程如此重要，是很有帮助的。

在许多编程场景中，数据项彼此相关。尽管变量非常适合存储单个信息，例如一个数字、一个人的姓名或一个布尔状态，但我们通常需要一种方法将多条数据组合并进行管理。例如，考虑维护一个班级的学生分数列表、表示三维空间中一个点的 (x, y, z) 坐标，或者管理一个产品目录（其中每个产品都由唯一代码标识）。对于这些情况，使用单独的变量会很繁琐且效率低下。这正是数据结构派上用场的地方。

**数据结构**本质上是一种用于组织、处理、获取和存储数据的专门格式。你可以将其视为一个专门设计用于容纳多个数据项的容器，以特定方式排列它们，从而方便执行常见操作。在日常生活中，你使用各种系统来组织信息：待办事项列表让任务保持有序，日历按日期组织日程，地址簿将姓名与联系方式关联起来。同样，像Julia这样的编程语言也提供了一系列数据结构，每种都针对不同类型的数据和不同的操作需求进行设计。

理解并有效使用数据结构很重要，原因如下：

- **管理复杂性**：它们允许你将相关数据打包成一个单一的逻辑单元。这通过减少你需要跟踪的单个变量数量来简化你的代码，使程序的整体设计更清晰。
- **效率**：数据结构的选择直接影响程序的性能。数据的组织方式会影响你执行查找项、添加新数据或删除现有条目等操作的速度。选择合适的结构可以带来显著的速度提升。
- **代码可读性和可维护性**：使用选择得当的数据结构的程序通常更容易理解、调试和修改。当数据组织得直观时，代码的逻辑会变得更清晰。

在Julia中，这些有组织的数据组通常被称为集合。正如本章介绍中概述的，我们将重点介绍四种主要的集合类型：

- **数组（Arrays）**：可以将它们视为灵活、有序的列表，就像你可以添加或更改的购物清单。数组中的项按特定顺序存储，你可以通过它们的位置访问它们。
- **元组（Tuples）**：它们也是有序列表，但有一个主要区别：一旦创建元组，其内容和大小就不能更改。它们非常适合表示固定项组，例如特定日期（年、月、日）或一对地理坐标。
- **字典（Dictionaries）**：这些集合将数据存储为键值对。它们类似于查找表，就像地址簿将姓名（键）链接到联系方式（值）一样。如果你知道键，字典可以非常快速地获取数据。
- **集合（Sets）**：集合是一种每个项都必须是唯一的集合，并且项的顺序通常不是主要考虑因素。它们可用于执行以下任务：检查项是否属于某个组、从调查中查找唯一响应，或执行数学集合操作，例如并集和交集。

下图提供了一个简单的示例，说明单个数据项如何在数据结构中进行分组。

> 数据结构使我们能够将不同的信息片段（例如`"葡萄"`、`17`和`true`）分组到一个单一、易于管理的集合中，例如一个数组。

在本章的后续章节中，我们将详细查看Julia的每种集合类型。你将了解它们的具体特点、如何创建和初始化它们、如何添加、访问和修改其内容，以及它们解决编程问题的常见方式。熟练掌握这些集合是编写功能更强、效率更高的Julia程序的重要进步。

## 参考资料

- [Introduction to Algorithms](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition) — Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, and Clifford Stein (2022)
  Publisher: MIT Press
  提供了数据结构和算法的全面理论基础，解释了它们的设计和性能特征。
- [Collections and Data Structures](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQF_7LdBRzXztUPY-I0Dfr4dbZrTkUdFbSA50ARYwMXTBC9hxmXn_aO_up-OLOwWGxgjegwi2T8u_SKoMzW8hHjMC5u7iR6mbfGKTZOT9lIbbuP5Hpvq_rzMP4cb13TfdnW-FrX2KHNXBnGsVh_aeLU1ZVZkgtk0q_7w-sqr) — JuliaLang (2024)
  Publisher: The Julia Language
  详细介绍了 Julia 内置集合类型（数组、元组、字典和集合）的官方文档，包括它们的用法和具体行为。
- [Data Structures and Algorithms in Java](https://www.wiley.com/en-us/Data+Structures+and+Algorithms+in+Java%2C+6th+Edition-p-9781118771334) — Michael T. Goodrich, Roberto Tamassia, Michael H. Goldwasser (2014)
  Publisher: John Wiley & Sons
  展示了基础数据结构，并清晰地解释了它们的实现和应用，为解决编程问题提供了实用见解。

---

[上一节](../03-%E8%A1%A8%E8%BE%BE%E5%BC%8F%E4%B8%8E%E6%8E%A7%E5%88%B6%E6%B5%81%E7%BB%93%E6%9E%84/09-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%AE%9E%E7%8E%B0%E5%86%B3%E7%AD%96%E9%80%BB%E8%BE%91.md) · [下一节](02-%E6%95%B0%E7%BB%84%EF%BC%9A%E6%9C%89%E5%BA%8F%E3%80%81%E5%8F%AF%E5%8F%98%E9%9B%86%E5%90%88.md)
