---
course: "llm-model-sizes-hardware"
chapter: "essential-hardware-components"
lesson: "ram-role"
sourceId: 4206
sourceUrl: "https://apxml.com/zh/courses/llm-model-sizes-hardware/chapter-2-essential-hardware-components/ram-role"
title: "随机存取存储器 (RAM)"
description: "了解系统内存（RAM）及其对加载数据和模型的重要性。"
order: 2
plots: []
sourceHash: "4d331e421818f0c951b675d43c150a5068114a13a5c2743b6d1bf6239ce8d315"
sourceCorrections: []
---

可以将您电脑的内存想象成一个工作区域。中央处理器（CPU）是进行计算的工作者，它需要一个地方来临时存放正在处理的指令和数据。这个主要的工作区域被称为**随机存取存储器（RAM）**，通常简称为系统内存。

RAM是一种电子存储，比固态硬盘（SSD）或机械硬盘（HDD）等长期存储设备要快得多。当您启动程序或打开文件时，所需数据会从较慢的存储驱动器加载到速度快得多的RAM中，以便CPU能迅速读取。这种速度对于流畅的计算体验不可或缺。

### RAM在系统中的作用

RAM有以下几项重要功能：

1. **操作系统：** 它存放部分操作系统（如Windows、macOS或Linux），这些部分需要随时可用。
2. **应用程序：** 当您运行软件时，例如网页浏览器、代码编辑器或与AI模型交互的应用程序，这些软件代码会被加载到RAM中。
3. **活跃数据：** 应用程序当前正在使用的数据，例如您正在编辑文档中的文本或计算的中间结果，都存放在RAM中。

RAM的特点是它是**易失性**的。这意味着它只在电脑通电时保存数据。当您关闭电脑时，RAM中存储的所有内容都会消失。因此，您需要将工作保存到固态硬盘或机械硬盘等持久性存储设备上。

> CPU、RAM、GPU、VRAM和存储设备之间的基本关系。数据从持久存储移动到RAM进行通用处理，并通常进入VRAM进行像运行LLM这样需要大量GPU的任务。

### RAM容量与AI模型

RAM的计量单位是千兆字节（GB）。现代电脑中常见的容量包括8GB到16GB用于日常操作，32GB用于要求较高的用途，以及64GB或更多用于高性能工作站。

处理大型语言模型需要多少RAM？虽然LLM的*参数 (parameter)*本身通常加载到GPU的专用VRAM中（我们稍后会介绍），但系统RAM仍然扮演着辅助角色。

- **系统开销：** 您需要足够的RAM来运行操作系统和任何后台进程。
- **应用软件：** 用于运行或与LLM交互的应用程序（如Python环境、框架或用户界面）需要RAM。
- **数据处理：** 准备发送给模型或从模型接收的数据，可能会临时使用系统RAM。
- **模型溢出（不太理想）：** 在没有专用GPU或VRAM不足的系统上，LLM的部分内容有时可能会加载到系统RAM中。这比使用VRAM慢得多，并显著影响性能，但这表明在某些限制下，系统RAM*可以*是模型加载的一个因素。

对于典型的LLM*推理 (inference)*（使用模型，而非训练模型），系统RAM的容量通常不如VRAM的容量那么有局限性。然而，RAM不足（例如，尝试在只有4GB或8GB内存的系统上运行要求较高的软件）仍然会导致运行缓慢或阻止应用程序正常运行，无论是否涉及AI任务。

### RAM速度与延迟

除了容量（能容纳多少数据）之外，RAM还有速度（数据传输的快慢）和延迟（传输开始前的延时）。这些参数 (parameter)以兆赫（MHz）或每秒兆次传输（MT/s）以及时序（如CL）来衡量。虽然更快的RAM可以提高整体系统响应速度，但对于运行预训练 (pre-training)的LLM而言，RAM，特别是VRAM的*容量*通常比RAM速度上的差异影响更大。

总而言之，系统RAM是电脑的主要工作区域，对于运行操作系统、应用程序以及处理活跃数据来说不可或缺。虽然它通常不是容纳LLM参数本身的主要瓶颈（那通常是VRAM），但足够的RAM对于整体系统稳定性和高效处理AI模型所需的性能而言是必需的。接下来，我们将介绍在LLM所需的并行计算方面表现出色的专用硬件：GPU及其专用内存VRAM。

## 参考资料

- [Computer Organization and Design RISC-V Edition: The Hardware/Software Interface](https://www.elsevier.com/books/computer-organization-and-design-risc-v-edition/patterson/978-0-12-820331-6) — David A. Patterson, John L. Hennessy (2020)
  Publisher: Morgan Kaufmann; Pages: 736
  一本广泛认可的教材，详细介绍了计算机体系结构，包括内存层次结构、RAM的功能以及其与CPU和其他组件的交互。
- [Operating System Concepts](https://www.wiley.com/en-us/Operating+System+Concepts,+10th+Edition-p-9781119320913) — Abraham Silberschatz, Peter Baer Galvin, Greg Gagne (2023)
  Publisher: John Wiley & Sons
  这本权威教材阐明了操作系统如何管理和使用RAM，包括进程内存、虚拟内存和系统资源分配等方面。
- [Lecture 10: Memory Hierarchy](https://ocw.mit.edu/courses/6-004-computation-structures-spring-2017/resources/lecture-10-memory-hierarchy/) — MIT OpenCourseWare (2017)
  Publisher: MIT
  作为计算结构基础课程的一部分，本讲座清晰地解释了计算机内存层次结构，侧重于RAM的特性和系统位置。
