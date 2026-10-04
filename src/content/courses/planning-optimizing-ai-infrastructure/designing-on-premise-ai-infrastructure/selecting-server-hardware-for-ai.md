---
course: "planning-optimizing-ai-infrastructure"
chapter: "designing-on-premise-ai-infrastructure"
lesson: "selecting-server-hardware-for-ai"
sourceId: 6980
sourceUrl: "https://apxml.com/zh/courses/planning-optimizing-ai-infrastructure/chapter-2-designing-on-premise-ai-infrastructure/selecting-server-hardware-for-ai"
title: "AI服务器硬件的选择"
description: "一份关于如何选择合适的服务器机箱、主板和电源来构建专用AI设备的指南。"
order: 2
plots: []
sourceHash: "f25ed89a4df6a42ee5398149cf7f5296124ba347e57c38a2d77ac30cc7003b1a"
sourceCorrections: []
---

选择AI工作负载的物理硬件流程，始于确定其性能和容量目标。这不仅仅是从货架上挑选功能最强的组件。这关乎构建一个平衡且整合的系统，其中服务器机箱、主板和CPU需协调运作，以支持您昂贵且高功耗的GPU，确保它们永远不会等待数据或指令。

### 服务器机箱：您系统的根本

服务器机箱是容纳所有组件的物理外壳。对于专用AI服务器而言，它的作用远不止一个简单的外壳。它是您系统散热和结构设计的重要组成部分。

- **尺寸规格：** 大多数高性能多GPU服务器采用机架式机箱，其中**4U**尺寸规格是常见的选择。这种高度提供了充足的垂直空间，可容纳多个双槽甚至三槽GPU，以及有效散热所需的大型散热器和风扇。
- **GPU兼容性：** 查看机箱规格，了解其可容纳的最大GPU数量以及对全长卡的兼容性。高端GPU通常较长较重，需要坚固的安装支架来防止下垂和主板PCIe插槽损坏。
- **气流与散热：** AI工作负载会产生大量热量。一个好的服务器机箱应设计为最佳气流，具有从前到后的进气和排气风扇的清晰通道。寻找能够有力地推动空气穿过密集的GPU散热器和其他组件的高静压风扇。
- **电源单元（PSU）仓位：** 机箱必须容纳一个或多个强大的电源单元。一台搭载四个高端GPU的服务器很容易需要3000瓦或更高功率。许多服务器机箱支持冗余电源，允许其中一个故障而不会导致系统下线。

### 主板：系统的骨干

主板是连接所有组件的中心枢纽。对于AI服务器而言，最主要的功能是其为每个GPU提供最大数据带宽的能力。这几乎完全由其\*\*外围组件互连标准高速（PCIe）\*\*架构决定。

现代GPU需要PCIe x16插槽才能以其全部带宽运行。主板上可用的PCIe通道数量决定了您可以运行多少个GPU而不产生数据瓶颈。这些通道来源于两个方面：CPU和主板芯片组。为获得最高性能，您会希望GPU直接连接到CPU提供的PCIe通道。

考虑一下典型的消费级平台与服务器级或高端桌面（HEDT）平台之间的差异。

> 消费级CPU的PCIe通道有限，迫使第二个GPU通过芯片组共享带宽。而服务器级CPU则提供了足够的直接通道，可供四个GPU同时以完整的x16带宽运行。

选择主板时，请注意以下规格：

- **PCIe插槽配置：** 验证主板是否提供了足够的物理x16插槽。更重要的是，查看主板手册，了解通道是如何分布的。一块主板可能拥有四个物理x16插槽，但当所有插槽都插满时，它们可能只能以`x16/x8/x8/x4`的配置运行。为获得最佳性能，您需要一块能够为您打算使用的每个GPU插槽提供`x16`通道的主板。
- **插槽间距：** 高性能GPU由于其大型散热组件，通常是“双槽”或“三槽”宽度。确保主板的PCIe插槽间隔足够远，以便容纳所有您计划使用的GPU。
- **内存支持：** 验证最大支持内存和DIMM插槽的数量。虽然GPU有自己的显存 (VRAM)，但系统内存对于数据暂存和预处理非常重要，特别是处理超大数据集时。

### CPU：交响乐的指挥家

虽然GPU备受关注，但CPU仍然是运作的大脑。它负责处理操作系统、数据加载和预处理，并协调发送给GPU的任务。对于多GPU训练服务器而言，CPU最主要的功能不是其原始时钟速度，而是其**PCIe通道数量**。

如上图所示，消费级CPU（如英特尔酷睿或AMD锐龙）通常提供约20-24条PCIe通道。这足以满足一个GPU以全速x16运行，以及一个高速NVMe SSD以x4运行。然而，如果您添加第二个GPU，系统将被迫拆分通道，通常会使两个GPU都以较慢的x8配置运行，从而有效减半它们潜在的主板带宽。

这就是为什么HEDT和服务器级CPU，例如**AMD的线程撕裂者/EPYC**或**英特尔的至强**系列，是多GPU构建的标配。这些处理器可以直接从CPU提供64、128甚至更多的PCIe通道。这使您能够运行四个、八个或更多GPU，每个都使用专用x16插槽并提供完整带宽。

次要考虑因素是核心数量。一个拥有更高核心数量的CPU（例如16、32或64个核心）可以运行更多并行数据预处理线程。这对于构建能够不间断地为GPU提供数据的有效数据管道非常重要。如果您的数据加载和增强代码无法跟上GPU的处理速度，您昂贵的加速器将会闲置，浪费时间和电力。目标是选择一个拥有足够核心和PCIe通道的CPU，以有效地服务所有GPU。一个性能过弱的CPU会造成瓶颈，而CPU性能对于GPU数量而言过于强大则代表着资本支出的浪费。

## 参考资料

- [Thermal Guidelines for Data Processing Environments, 5th Edition](https://www.ashrae.org/technical-resources/bookstore/thermal-guidelines-for-data-processing-environments-5th-edition) — ASHRAE Technical Committee 9.9 (2021)
  Publisher: ASHRAE
  数据中心热设计方面的权威指南，涵盖高密度服务器环境，对AI服务器冷却至关重要。
- [A Study of the Impact of PCI Express Bandwidth on Deep Neural Network Training](https://ieeexplore.ieee.org/document/8939762) — Jongyoon Kim, Jinsuk Kim, Minsoo Kim, Jinwoo Kim (2019)
  Journal: 2019 International Conference on Information and Communication Technology Convergence (ICTC); Publisher: IEEE; Pages: 572-574; DOI: [10.1109/ICTC46691.2019.8939762](https://doi.org/10.1109/ICTC46691.2019.8939762)
  研究了PCIe带宽如何影响多GPU系统上深度神经网络的训练性能，并提供了数据驱动的见解。
