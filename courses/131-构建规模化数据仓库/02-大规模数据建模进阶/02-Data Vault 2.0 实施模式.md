# Data Vault 2.0 实施模式

来源：[原文](https://apxml.com/zh/courses/building-scalable-data-warehouses/chapter-2-advanced-data-modeling-scale/data-vault-implementation)

[返回章节目录](README.md) · [返回课程目录](../README.md)

标准维度建模非常侧重终端用户体验，通过非规范化来优先简化查询逻辑。然而，在处理PB级数据的分布式MPP环境中，2型缓慢变化维度（SCD）的维护开销可能成为一个性能限制。更新数百万行以关闭不同记录版本会引发计算节点间大量的数据重排。Data Vault 2.0 通过将业务主键、关系和描述性属性分离到不同结构中来应对此问题。这种分离使得100%并行加载成为可能，消除了摄取依赖，并支持灵活的模式演变，无需重构现有表。

### 组成架构

Data Vault 2.0 依靠三种主要结构：中心表（Hubs）、链接表（Links）和卫星表（Satellites）。与以事实表为中心的星型模型不同，Data Vault 以业务主键为核心。

#### 中心表（Hubs）

中心表只包含核心业务实体（例如客户、产品、发票）的业务主键的唯一列表。它不含描述性数据或关系。中心表的主要用途是建立一个独特的标识符列表，供其他结构引用。

在传统数据仓库中，代理键通常是由数据库生成的顺序整数。在MPP系统中，生成顺序ID会阻碍并行加载，因为加载器必须等待集中的计数器。Data Vault 2.0 用**哈希键**替代顺序ID。

哈希键是业务主键的确定性哈希值（通常是MD5或SHA-256）。

$K_{哈希} = f_{哈希}(\text{修剪}(\text{大写}(K_{业务})))$

这使得多个计算节点可以独立计算键值而无需协调，从而在摄取过程中实现线性扩展能力。

#### 链接表（Links）

链接表表示中心表之间的关联。它们在物理上处理多对多关系，而与逻辑基数无关。链接表包含其自身的哈希键、它所连接的中心表的哈希键以及加载日期。通过将关系外部化到链接表中，你可以在不修改中心表或现有卫星表的情况下添加新关系。

#### 卫星表（Satellites）

卫星表存储描述性属性（上下文 (context)）和历史数据。卫星表连接到中心表或链接表。它包含父级的哈希键、加载日期、哈希差异值（用于变更检测）和实际数据列。

当源记录发生变化时，你不会更新现有行。相反，你会以新的加载日期向卫星表插入新行。这种“只插入”架构在日志结构合并树和列式存储引擎（在Snowflake和BigQuery中常见）上表现出高效率。

> 基本Data Vault模型的结构，呈现中心表（蓝色）、链接表（红色）和卫星表（绿色）之间的关系。

### 并行摄取和哈希差异值

Data Vault 的可扩展性来源于其加载模式。由于哈希键是确定性的，你可以并行加载中心表、链接表和卫星表。在高容量环境中的一个特定难题是变更数据捕获（CDC）。

为避免查找前一条记录值以检测变更（这需要耗费资源的自连接或窗口函数），Data Vault 采用**哈希差异值**。哈希差异值是通过连接源数据流中所有描述性列并对结果进行哈希计算得出的。

在暂存过程中，你会为传入记录计算哈希差异值。加载逻辑变成一个简单的集合比较：

1. 计算 `HashDiff_Incoming`。
2. 将暂存表左连接到当前卫星表，基于 `Hub_Hash_Key`。
3. 过滤出 `Satellite.Hash_Key` 为 NULL (新父级) 或 `Staging.HashDiff != Satellite.HashDiff` (数据已变更) 的记录。
4. 插入所得记录。

此比较运用了MPP引擎高效执行大规模哈希连接的能力。

### 卫星表拆分以优化性能

Data Vault 早期实现中一个普遍的反模式是将业务实体的所有属性分组到一个单一的卫星表中。在PB级数据仓库中，这会造成过度的I/O扫描。

属性的变更速度通常不同。举例来说，客户的 `LoyaltyTier` 可能每年只变更一次，而他们的 `LastLoginDate` 每天都在变更。如果存储在同一个卫星表中，每次登录事件都需要在新行中重复存储静态 `LoyaltyTier` 数据，这会增加存储量并减慢历史查询速度。

根据以下几点拆分卫星表：

1. **变更频率：** 将高频变化的列和低频变化的列分别分组。
2. **源系统：** 按源系统隔离数据（例如 `Sat_Cust_Salesforce`, `Sat_Cust_SAP`）。这能避免一个源系统的延迟阻塞另一个源系统的加载。
3. **数据分类：** 将PII（个人身份信息）分离到一个受保护的卫星表，并使用受限访问策略（行级安全）加以保护。

### 查询性能和即时（PIT）表

尽管Data Vault在写入方面表现出色，但高度规范化要求通过多次连接才能重构数据的“扁平”视图。将一个中心表连接到五个不同的卫星表和两个链接表可能会降低查询延迟。

为解决消费层的此问题，我们实施\*\*即时（PIT）\*\*表。PIT表是一个桥接表，它会预计算特定时间快照的有效连接。

PIT表的结构通常包含：

- 中心表哈希键。
- 快照日期。
- 指向该日期每个相关卫星表中有效记录的外键。

通过在批处理窗口期间（或持续地）计算这些时间对齐 (alignment)，下游查询变为基于主键的简单等值连接，使得MPP优化器能够高效运用分区修剪。



[交互图表：查询延迟：原始Data Vault与PIT表优化对比](https://apxml.com/zh/courses/building-scalable-data-warehouses/chapter-2-advanced-data-modeling-scale/data-vault-implementation#plot-7mh7wm)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": {
      "text": "查询延迟：原始Data Vault与PIT表优化对比",
      "font": {
        "size": 14
      }
    },
    "xaxis": {
      "title": "连接的卫星表数量"
    },
    "yaxis": {
      "title": "查询执行时间 (秒)"
    },
    "width": 600,
    "height": 400,
    "margin": {
      "l": 50,
      "r": 20,
      "t": 50,
      "b": 50
    }
  },
  "data": [
    {
      "type": "scatter",
      "mode": "lines+markers",
      "name": "原始Vault连接",
      "x": [
        1,
        2,
        3,
        4,
        5,
        6
      ],
      "y": [
        1.2,
        2.5,
        4.8,
        8.1,
        14.5,
        25.2
      ],
      "line": {
        "color": "#fa5252"
      }
    },
    {
      "type": "scatter",
      "mode": "lines+markers",
      "name": "PIT表连接",
      "x": [
        1,
        2,
        3,
        4,
        5,
        6
      ],
      "y": [
        0.8,
        1.1,
        1.3,
        1.5,
        1.8,
        2.1
      ],
      "line": {
        "color": "#228be6"
      }
    }
  ]
}
```

</details>



> 对比直接连接多个卫星表与运用预计算的即时（PIT）结构时查询延迟的扩展情况。

### 模式演变的处理

模式漂移是不可避免的。在星型模型中，添加列通常需要对可能非常庞大的事实表执行 `ALTER TABLE` 命令，这会锁定资源。在Data Vault中，模式演变通过添加新的卫星表来处理。

如果源系统为客户增加了五个新字段：

1. **方案A：** 将列添加到现有卫星表（需要 `ALTER TABLE`，但在Snowflake/BigQuery中通常是安全的，因为它是一个元数据操作）。
2. **方案B：** 创建一个新的垂直卫星表（例如 `Sat_Customer_v2`），链接到同一个中心表。

当新数据来自不同的微服务或具有不同的加载节奏时，方案B更受青睐。中心表保持不变。链接表保持不变。下游视图会更新以包含新列，但现有数据的物理摄取管道不会中断。这种增量特性使得Data Vault 对管理持续集成流水线的敏捷工程团队尤其适用。

## 参考资料

- [Designing Data-Intensive Applications: The Big Ideas Behind Reliable, Scalable, and Maintainable Systems](https://www.oreilly.com/library/view/designing-data-intensive-applications/9781491903063/) — Martin Kleppmann (2017)
  Publisher: O'Reilly Media
  一本关于构建可伸缩、可靠和可维护数据系统原理和挑战的权威书籍，为现代数据仓库中使用的分布式处理和存储概念提供了背景。
- [Why Data Vault 2.0?](https://scalefree.com/data-vault-2-0-definition/) — Daniel Linstedt (2014)
  一篇重要的白皮书，阐述了 Data Vault 2.0 的核心原则、优势和区别，用于构建敏捷、可审计和可伸缩的数据仓库。

---

[上一节](01-%E5%A4%A7%E6%95%B0%E6%8D%AE%E4%B8%AD%E7%9A%84%E7%BB%B4%E5%BA%A6%E5%BB%BA%E6%A8%A1%E9%99%90%E5%88%B6.md) · [下一节](03-%E5%A4%84%E7%90%86%E5%8D%8A%E7%BB%93%E6%9E%84%E5%8C%96%E6%95%B0%E6%8D%AE.md)
