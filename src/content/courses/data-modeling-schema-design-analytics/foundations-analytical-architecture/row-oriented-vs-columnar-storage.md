---
course: "data-modeling-schema-design-analytics"
chapter: "foundations-analytical-architecture"
lesson: "row-oriented-vs-columnar-storage"
sourceId: 7970
sourceUrl: "https://apxml.com/zh/courses/data-modeling-schema-design-analytics/chapter-1-foundations-analytical-architecture/row-oriented-vs-columnar-storage"
title: "行式存储与列式存储"
description: "列式存储为何在分析方面优于行式存储。"
order: 2
plots: ["plots/7970-0.json"]
sourceHash: "bb450242d09faa639f56847a8d90f9d52598df85fc5325fee39d9150d8263e1b"
sourceCorrections: []
---

数据库的性能从根本上受数据在存储介质上物理排列方式的制约。尽管SQL抽象了数据获取过程，数据库引擎在处理请求时最终都必须从磁盘或内存读取数据块。这些数据块的排列方式决定了一个系统是擅长处理单个事务还是汇总数百万条记录。

### 数据读取的物理原理

要明白分析系统为何偏爱列式存储，我们必须考察I/O传输单位。数据库并非读取单个行或单元格；它们读取的是数据块（或页）。一个典型数据块大小，根据系统配置，通常介于4KB到32KB之间。

当一个查询请求单个数据点时，引擎会将包含该点的整个数据块加载到内存中。数据库架构的效率主要由在此操作中加载的有用数据与总加载数据之比来衡量。

### 行式存储布局

在行式数据库中（如PostgreSQL、MySQL或SQL Server），数据按行顺序存放。如果有一个`Sales`表，包含`Date`、`Product_ID`、`Store_ID`和`Revenue`列，存储引擎会先写入第一个事务的数据，紧接着就是第二个事务的数据。

考量三个事务的内存排列：

> 内存布局图示了行式系统如何交错存储属性。行1的所有属性都彼此紧邻存放。

这种布局非常适合操作型（OLTP）工作负载。当用户登录或查看其具体的订单历史时，查询通常是这样的：

```sql
SELECT * FROM Sales WHERE Transaction_ID = 105;
```

数据库寻找包含事务105的特定数据块。由于数据是行式存储，引擎可以在一次I/O操作中获取该事务的日期、产品、商店和收入信息。这对于“点查询”非常有效，其目的是获取特定实体的所有相关信息。

### 分析瓶颈

当目标转向分析时，性能表现会大幅改变。分析查询很少关注单个事务。它们通常是扫描数据区域以计算聚合结果。

考量一个计算总收入的查询：

```sql
SELECT SUM(Revenue) FROM Sales;
```

在上面所示的行式布局中，数据库必须读取包含销售数据的每个数据块。然而，对于加载的每个数据块，只有`Revenue`（收入）这个整数是有用的。`Date`（日期）、`Product_ID`（产品ID）和`Store_ID`（商店ID）都被加载到内存中，占用了带宽和缓存空间，但最终却被废弃。

如果`Revenue`列只占总行宽度的5%，那么95%的I/O吞吐量 (throughput)就被浪费了。这种低效是操作型数据库在数据仓库工作负载下表现不佳的主要原因。

### 列式存储架构

列式存储通过改变物理布局来解决这种低效问题。它不是按行存储数据，而是按列存储。`Date`列的所有值都顺序存放，随后是所有`Product_ID`值的独立存储区域，以此类推。

这种结构变化意味着，一个计算收入总和的查询只需访问包含`Revenue`数据的特定数据块。引擎可以完全不理会包含日期、产品或客户备注的数据块。

> 列式布局将属性分离到不同的存储块中。一个仅请求收入的查询，只会访问紫色块。

### 量化 (quantization)比较

我们可以用数学方式定义性能差异。设$N$为表中的行数，$W$为行的平均字节宽度。

在行式存储中，全表扫描需要读取的总数据量$V_{row}$为：

$V_{row} = N \times W$

在列式存储中，如果我们只用到列的一个子集$C$，其组合宽度为$w_{c}$，那么扫描的数据量$V_{col}$是：

$V_{col} = N \times w_{c}$

由于分析查询通常只选择列的一个小部分（通常是50多列中的2到5列），所以$w_{c}$明显小于$W$。

比如，如果一个表有50列，总宽度为500字节，而我们只需要1列，其宽度为4字节：

- 行式存储每条记录读取500字节。
- 列式存储每条记录读取4字节。
- **性能提高：** I/O减少了$500 / 4 = 125$倍。

下图显示了随着选择列数增加，扫描时间所受的影响。



![查询执行时间与列选择的关系](plots/7970-0.json)



> 扫描时间对比。行式存储（红色）由于读取整行数据，不管选择多少列，都会产生固定的高成本。列式存储（蓝色）则根据所需列数线性增长。

### 压缩效能

除了简单的I/O减少，列式存储还能实现更优异的数据压缩。在行式数据块中，数据类型差别很大，一个字符串后可能跟着一个整数，再跟着一个时间戳。这种无序性使得压缩算法的效果不佳。

在列式数据块中，每个值都具有相同的数据类型，并且通常属于类似的范围。存储“国家”的列可能会一个接一个地出现“USA”这个值数千次。

列式数据库采用\*\*行程长度编码（RLE）\*\*等编码方案，以便更好地处理这种多次出现的情况。如果一列中一个接一个地出现500次“USA”这个值，数据库会将其存储为简化的元组`('USA', 500)`，而不是写入500次“USA”。这可以将存储空间减少10到50倍，此外还能加快扫描速度，因为CPU可以直接在L1/L2缓存中处理压缩数据。

### 写入开销

数据库架构中没有免费的午餐。对读取速度的优化会给写入操作带来很大的开销。

在行式存储中，插入一条记录是对活动文件末尾进行一次追加操作。在列式存储中，插入一条记录需要数据库进行以下步骤：

1. 将行分解成其各个属性。
2. 打开*每个*列的特定文件或分区。
3. 将值追加到每个文件，同时管理压缩元数据。

这个过程，被称为“元组拆分”，开销较大。因此，分析系统（OLAP）通常避免使用单条`INSERT`语句。它们更依赖于批量加载过程，在其中数千或数百万行数据被缓存并同时写入，从而分摊了重构成本。这种基本的架构不同之处决定了我们为何将分析型数据仓库与实时事务数据库分开。

## 参考资料

- [C-Store: A Column-Oriented DBMS](https://doi.org/10.14778/1070617.1070670) — Michael Stonebraker, Daniel J. Abadi, Adam Batkin, Xuedong Chen, Mitch Cherniack, Miguel Ferreira, Edmond Lau, Amerson Lin, Samuel Madden, Elizabeth J. O'Neil, Patrick E. O'Neil, Alex Rasin, Nga Tran, Stanley B. Zdonik (2005)
  Journal: Proceedings of the 31st International Conference on Very Large Data Bases (VLDB); Publisher: VLDB Endowment; Pages: 553-564; DOI: [10.14778/1070617.1070670](https://doi.org/10.14778/1070617.1070670)
  这篇开创性论文介绍了C-Store架构，详细阐述了列式存储在分析工作负载中的优势，并为许多现代分析型数据库提供了蓝图。
- [Designing Data-Intensive Applications: The Big Ideas Behind Reliable, Scalable, and Maintainable Systems](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGSr0kw2JjzLzcqw12A69FF4V0m20NAy5FzKtSVaKgxhXFWXuQUNCd23BO5Fo_a_GiUI0fIm4AfHOF6V3X_90to1SFYh52saTymRQZ4CFN6Zylw2_VKJBqhNYoxxZ7xlpJ4o7AbtpGmFhhD_NCXOLm0drYqNna1GJ6MTBvMAQ7e5zqYuz-lka29VFL-GwdXOY1xddaTSxVe-2sLVRR8yA==) — Martin Kleppmann (2017)
  Publisher: O'Reilly Media, Inc.
  这本备受推崇的书解释了数据库存储和检索中的基本权衡，包括对行式和列式布局、压缩及其对OLTP和OLAP系统影响的深入比较。
- [Database System Concepts](https://www.db-book.com/) — Avi Silberschatz, Henry F. Korth, S. Sudarshan (2019)
  Publisher: McGraw-Hill
  这是一本涵盖数据库基础概念的综合教科书，包括物理存储结构、文件组织、索引和I/O效率，为理解行式和列式存储提供了理论背景。(第7版)
