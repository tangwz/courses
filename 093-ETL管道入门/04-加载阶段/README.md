# 第 4 章：加载阶段

来源：[原章节](https://apxml.com/zh/courses/intro-etl-pipelines/chapter-4-the-loading-stage)

[返回课程目录](../README.md)

数据提取并转换为可用格式后，ETL 工作流程的最后一步是将其加载到目标系统。本章讨论“加载”阶段，已准备好的数据会传输到其目的地，例如数据仓库或数据库。

您将学习到：
*   选择合适的目标系统。
*   常见的加载策略，包括完全加载 ($L_{full}$) 和增量更新 ($L_{incremental}$)。
*   目标模式的重要性以及转换后数据字段的映射。
*   处理加载错误的方法。
*   加载后数据验证的方法。

成功完成此阶段，使得处理过的数据可用于分析和下游应用。

## 小节

- 1. [选择目标系统](01-%E9%80%89%E6%8B%A9%E7%9B%AE%E6%A0%87%E7%B3%BB%E7%BB%9F.md)
- 2. [加载策略：完整加载](02-%E5%8A%A0%E8%BD%BD%E7%AD%96%E7%95%A5%EF%BC%9A%E5%AE%8C%E6%95%B4%E5%8A%A0%E8%BD%BD.md)
- 3. [加载策略：增量加载（追加/更新）](03-%E5%8A%A0%E8%BD%BD%E7%AD%96%E7%95%A5%EF%BC%9A%E5%A2%9E%E9%87%8F%E5%8A%A0%E8%BD%BD%EF%BC%88%E8%BF%BD%E5%8A%A0-%E6%9B%B4%E6%96%B0%EF%BC%89.md)
- 4. [理解目标模式](04-%E7%90%86%E8%A7%A3%E7%9B%AE%E6%A0%87%E6%A8%A1%E5%BC%8F.md)
- 5. [模式映射：从源到目标](05-%E6%A8%A1%E5%BC%8F%E6%98%A0%E5%B0%84%EF%BC%9A%E4%BB%8E%E6%BA%90%E5%88%B0%E7%9B%AE%E6%A0%87.md)
- 6. [处理加载失败](06-%E5%A4%84%E7%90%86%E5%8A%A0%E8%BD%BD%E5%A4%B1%E8%B4%A5.md)
- 7. [加载后数据验证](07-%E5%8A%A0%E8%BD%BD%E5%90%8E%E6%95%B0%E6%8D%AE%E9%AA%8C%E8%AF%81.md)
- 8. [练习：数据加载](08-%E7%BB%83%E4%B9%A0%EF%BC%9A%E6%95%B0%E6%8D%AE%E5%8A%A0%E8%BD%BD.md)

章节测验：[在线测验](https://apxml.com/zh/courses/intro-etl-pipelines/chapter-4-the-loading-stage/quiz)
