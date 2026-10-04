# 半结构化数据（例如JSON、XML）简介

来源：[原文](https://apxml.com/zh/courses/intro-etl-pipelines/chapter-2-the-extraction-stage/extracting-semi-structured-data)

[返回章节目录](README.md) · [返回课程目录](../README.md)

尽管结构化数据（例如数据库或CSV文件中的整齐行和列）很常见，但当今生成的大部分数据无法整齐地放入表中。这介绍了半结构化数据，它是一种介于结构化数据严格格式和非结构化数据（如纯文本文档或图像）完全无组织形式之间的数据类别。

半结构化数据不遵循关系数据库表那样的固定正式模式，但它确实包含标签或其他标记 (token)来分隔语义元素，并强制记录和字段的层次结构。可以将其视为有组织线索，但没有严格蓝图。这种灵活性使其在结构可能演变或变化的系统中非常有用。

提取半结构化数据时，您会经常遇到两种普遍使用的格式是JSON和XML。

### JSON：JavaScript对象表示法

JSON是一种轻量级的基于文本的格式，便于人类读写，也便于机器解析和生成。随着Web API（应用程序编程接口）的兴起，它变得非常流行，因为它与许多编程语言中的数据结构非常自然地对应。

JSON数据建立在两种主要结构之上：

1. **对象：** 键值对的集合，用花括号`{}`括起来。键是字符串，值可以是字符串、数字、布尔值（`true`/`false`）、数组，甚至是其他嵌套对象。
2. **数组：** 值的有序列表，用方括号`[]`括起来。数组中的值可以是任何有效的JSON类型。

这是一个表示个人信息的简单例子：

```json
{
  "firstName": "Jane",
  "lastName": "Doe",
  "age": 30,
  "isStudent": false,
  "address": {
    "streetAddress": "123 Main St",
    "city": "Anytown",
    "postalCode": "12345"
  },
  "phoneNumbers": [
    {
      "type": "home",
      "number": "555-1234"
    },
    {
      "type": "work",
      "number": "555-5678"
    }
  ]
}
```

注意`address`是一个嵌套对象，而`phoneNumbers`是一个包含多个电话号码对象的数组，每个对象都有自己的`type`和`number`。

### XML：可扩展标记 (token)语言

XML是另一种基于文本的格式，用于存储和传输数据。它使用标签来定义元素，并使用属性来提供有关这些元素的额外信息。XML是分层的，这意味着元素可以嵌套在其他元素中，形成树状结构。

尽管与JSON相比，XML现在在新Web API中的使用可能较少，但它仍然广泛应用于配置文件、文档格式（如Microsoft Office XML格式）以及企业系统中的数据交换。

这是用XML表示的相同个人信息：

```xml
<person>
  <firstName>Jane</firstName>
  <lastName>Doe</lastName>
  <age>30</age>
  <isStudent>false</isStudent>
  <address>
    <streetAddress>123 Main St</streetAddress>
    <city>Anytown</city>
    <postalCode>12345</postalCode>
  </address>
  <phoneNumbers>
    <phone type="home">
      <number>555-1234</number>
    </phone>
    <phone type="work">
      <number>555-5678</number>
    </phone>
  </phoneNumbers>
</person>
```

在这个XML例子中，`<person>`、`<firstName>`、`<address>`等是定义元素的标签。注意`<phone>`标签内的`type="home"`；这是一个属性，提供了关于电话元素的额外细节。

### 提取半结构化数据

与从关系数据库表中提取数据相比，从JSON或XML源提取数据面临不同的挑战。

1. **解析：** 原始JSON或XML只是文本。提取的第一步通常是将这些文本解析成ETL工具或脚本能够理解和操作的内部数据结构。大多数编程语言和ETL工具都内置了专门用于解析JSON和XML的库或组件。
2. **遍历层次结构：** 因为这些格式是分层的，您通常需要遍历嵌套对象或元素以找到所需的特定数据片段。例如，要从JSON示例中获取Jane Doe的工作电话号码，您需要访问`phoneNumbers`数组，找到`type`为“work”的对象，然后获取与`number`键关联的值。
3. **模式多变性：** 尽管存在*一些*结构，但它可能并非在所有记录或文件中都完全一致。一些记录可能包含其他记录没有的可选字段。您的提取逻辑需要足够灵活，以处理这种潜在的多变性而不会失败。

例如，设想从前面所示的JSON结构中提取街道地址。这个过程包括访问顶层对象，然后是`address`键，最后是嵌套地址对象中的`streetAddress`键。

> 此图显示了JSON和XML数据中常见的嵌套结构，显示对象包含其他对象或数组。

理解如何处理这些半结构化格式在现代ETL中非常重要，因为它们是Web服务、物联网设备和各种应用程序日志的常见输出。提取阶段需要具备解析这些格式并提取相关信息的能力，然后才能开始转换阶段。

## 参考资料

- [The JSON Data Interchange Format](https://www.rfc-editor.org/rfc/rfc8259.html) — T. Bray, Ed. (2017)
  Journal: RFC 8259; Publisher: Internet Engineering Task Force (IETF)
  这是定义JSON、其语法和数据类型的官方IETF标准。它是该格式的权威规范。
- [Designing Data-Intensive Applications: The Big Ideas Behind Reliable, Scalable, and Maintainable Systems](https://www.oreilly.com/library/view/designing-data-intensive-applications/9781449373320/) — Martin Kleppmann (2017)
  Publisher: O'Reilly Media; Pages: 611
  本书提供了数据处理概念的全面概述，涵盖了JSON和XML等数据模型和格式，以及它们对系统设计的影响。第四章讨论了这些格式。
- [Fundamentals of Data Engineering](https://www.oreilly.com/library/view/fundamentals-of-data/9781098108298/) — Joe Reis, Matt Housley (2022)
  Publisher: O'Reilly Media
  这本新书指导数据工程实践，涵盖了现代数据管道中的数据摄取和各种数据格式，包括半结构化数据。与提取阶段相关。

---

[上一节](03-%E5%A4%84%E7%90%86%E7%BB%93%E6%9E%84%E5%8C%96%E6%95%B0%E6%8D%AE%EF%BC%88%E4%BE%8B%E5%A6%82%EF%BC%9A%E6%95%B0%E6%8D%AE%E5%BA%93%E3%80%81CSV%EF%BC%89.md) · [下一节](05-%E6%95%B0%E6%8D%AE%E5%8F%98%E5%8C%96%E6%8D%95%E8%8E%B7%EF%BC%88CDC%EF%BC%89%E5%8E%9F%E7%90%86.md)
