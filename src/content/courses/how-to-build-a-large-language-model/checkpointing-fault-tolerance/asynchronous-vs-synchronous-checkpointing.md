---
course: "how-to-build-a-large-language-model"
chapter: "checkpointing-fault-tolerance"
lesson: "asynchronous-vs-synchronous-checkpointing"
sourceId: 6046
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-19-checkpointing-fault-tolerance/asynchronous-vs-synchronous-checkpointing"
title: "异步检查点与同步检查点"
description: "比较同步与异步检查点方法及其开销。"
order: 4
plots: []
sourceHash: "416715047f52984638cc522e7a2c43b811103477808f83c7dd4507dde50bbe25"
sourceCorrections: []
---

在为长时间运行的训练任务实现检查点时，首要的考量是保存过程如何与正在进行的训练计算相互影响。训练在写入检查点时是完全暂停，还是可以同时进行？这引出了两种主要方式：同步检查点和异步检查点。选择它们时需要权衡简单性、一致性和性能开销。

### 同步检查点

同步检查点是最直接的方式。当检查点触发时（例如，在达到一定的训练步数或经过一段时间后），训练过程会明确地暂停所有计算。然后，它收集所需的状态组成部分：模型参数 (parameter)、优化器状态、学习率调度器状态、当前的轮次或步数，以及可能的数据加载器迭代器状态。一旦所有状态收集完毕，它们就会被序列化并写入持久化存储（如分布式文件系统或云存储）。只有在写入操作成功完成后，训练过程才会恢复计算。

在分布式训练环境下，同步检查点需要所有参与工作节点之间的协调。通常，在保存之前会使用屏障同步，以确保所有工作节点都到达同一点。一个工作节点（通常是0号节点）可能会被指定从其他节点收集状态，或者每个工作节点保存自己部分的状态。保存之后可能会使用另一个屏障，以确保所有工作节点都等到检查点完全写入后再继续。

**优点：**

- **简单性：** 更易于实现和理解。保存的状态在训练中某个特定、明确的时间点，保证所有组件和工作节点间的一致性。
- **一致性保证：** 检查点反映了其被触发时训练任务的精确状态。如果恢复，训练会精确地从该点开始。

**缺点：**

- **性能开销：** 训练在保存操作期间完全停止。将大型检查点（对于大型语言模型可能达到数百吉字节或太字节）写入存储可能需要相当长的时间，为GPU等昂贵的计算资源带来大量空闲时间。这种开销随模型大小和分布式工作节点数量的增加而增加。
- **阻塞性：** 整个训练过程被阻塞，影响整体训练吞吐量 (throughput)。

以下是一个在分布式环境中，使用类似PyTorch语法的训练循环中同步检查点的表示：

```python
# 假设已初始化 torch.distributed

def save_synchronous_checkpoint(
    rank, world_size, model, optimizer, scheduler, step, path
):
    # 确保所有进程在保存前都到达这一点
    if world_size > 1:
        torch.distributed.barrier()

    if rank == 0: # 0号节点处理合并状态的保存
        print(
            f"Rank {rank}: Starting synchronous checkpoint save at step {step}..."
        )
        # 在实际场景中，状态可能会从其他节点收集
        # 或者每个节点保存自己部分（例如，使用DeepSpeed/FSDP辅助函数）
        state = {
            'step': step,
            'model_state_dict': model.state_dict(),
            # 或者 model.module.state_dict()
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
            # 可能添加数据加载器状态、随机数生成器状态等
        }
        torch.save(state, path)
        print(f"Rank {rank}: Finished synchronous checkpoint save to {path}.")
    else:
        # 其他节点等待0号节点完成保存
        pass

    # 确保0号节点上的保存完成，然后所有节点才能继续
    if world_size > 1:
        torch.distributed.barrier()

# --- 训练循环内部 ---
model.train()
for step, batch in enumerate(data_loader):
    # 前向传播、反向传播、优化器步进...
    outputs = model(batch['input_ids'])
    loss = calculate_loss(outputs, batch['labels'])
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

    # 定期检查点
    if step % checkpoint_interval == 0 and step > 0:
        checkpoint_path = f"/path/to/checkpoints/step_{step}.pt"
        # --- 阻塞保存操作 ---
        save_synchronous_checkpoint(
            rank,
            world_size,
            model,
            optimizer,
            scheduler,
            step,
            checkpoint_path
        )
        # --- 训练仅在保存完成后恢复 ---

    # ... 循环的其余部分（日志记录、评估等）
```

下图说明了同步检查点的阻塞特性。

> 训练在所有节点同步保存检查点时完全停止。

### 异步检查点

异步检查点旨在减轻同步保存的性能开销。其核心思路是将写入检查点这一计算开销大的I/O操作与主训练循环分离。

当检查点触发时，主训练进程会启动保存操作，但不会等待其完成。这通常通过以下方式实现：

1. **复制状态：** 在内存中快速复制所需状态（模型参数 (parameter)、优化器状态等）。这种复制操作相对于磁盘I/O应该较快。
2. **卸载I/O：** 将复制的状态移交给独立的线程、进程，甚至专门的I/O节点，以便在后台执行实际的序列化和写入持久化存储。
3. **继续训练：** 主训练循环在启动复制或移交后几乎立即恢复计算，使检查点I/O与有用计算重叠进行。

**优点：**

- **减少开销：** 最大限度地减少主训练循环停滞的时间，显著提升训练吞吐量 (throughput)和硬件利用率。阻塞时间缩短为初始状态复制所需的时间，而非完整的I/O持续时间。
- **吞吐量提升：** 通过将I/O与计算重叠，可以缩短总体训练时间。

**缺点：**

- **复杂性增加：** 正确实现异步检查点更为复杂。它需要细致地管理后台线程或进程，处理后台保存期间可能出现的错误，并确保数据一致性。
- **潜在状态偏差：** 异步保存的检查点将表示模型在保存操作完成到磁盘*之前*的短暂时间内的状态。如果在异步保存启动后不久发生故障，最新完成的检查点可能比同步情况稍旧。这种“滞后性”通常可以接受，仅代表少量可能丢失的进度（几次迭代）。
- **资源管理：** 需要管理后台保存进程所使用的资源（CPU、内存、网络带宽），使其不会过度干扰主训练计算。

实现异步检查点通常涉及使用线程或多进程库。

```python
import threading
import torch
import time
import os
# 假设 torch.distributed 已初始化 (rank, world_size)

# 实际模型、优化器等的占位符
class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
    def forward(self, x): return self.linear(x)
    def state_dict(self): return {'param': torch.randn(10, 10)}

model = DummyModel()
optimizer = torch.optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=10, gamma=0.1
)

# 用于跟踪后台保存线程的全局变量
checkpoint_thread = None

def background_save_task(state, path):
    """由后台线程执行的函数。"""
    print(f"Background Saver: Starting async save to {path}...")
    try:
        # 模拟慢速I/O
        time.sleep(5) # 模拟保存时间
        # 如果目录不存在则创建
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(state, path)
        print(f"Background Saver: Finished async save to {path}.")
    except Exception as e:
        print(f"Background Saver: Error during checkpointing: {e}")

def save_asynchronous_checkpoint(
    rank, world_size, model, optimizer, scheduler, step, path
):
    global checkpoint_thread

    # 确保前一个后台保存完成，然后才能开始新的保存
    if checkpoint_thread is not None and checkpoint_thread.is_alive():
        print(f"Rank {rank}: Waiting for previous async checkpoint to finish...")
        checkpoint_thread.join() # 等待前一个线程完成

    if rank == 0: # 0号节点启动并管理保存线程
        print(
            f"Rank {rank}: Initiating asynchronous checkpoint save at step {step}..."
        )
        # --- 快速复制状态 ---
        # 如有必要，请使用deepcopy，state_dict()通常返回副本/视图
        state = {
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }
        # --- 启动后台线程 ---
        checkpoint_thread = threading.Thread(
            target=background_save_task, args=(state, path)
        )
        checkpoint_thread.start()
        print(f"Rank {rank}: Background save launched. Training continues.")
    else:
        # 其他节点可能需要最少的协调，例如，确保它们
        # 如果一致性非常重要，不要进展太远，
        # 但通常它们会继续训练。
        # 如果需要复制状态前的严格一致性，
        # 则在复制状态*之前*可能需要一个屏障。
        pass

    # --- 训练在所有节点上立即继续 ---
    # 注意：此处没有屏障，允许重叠

# --- 训练循环内部 ---
step = 0
checkpoint_interval = 5 # 示例：每5步检查点
max_steps = 20

print("启动模拟训练循环...")
while step < max_steps:
    step += 1
    print(f"主循环：训练步 {step}")
    # 模拟训练工作
    time.sleep(0.5)
    # model(...), loss.backward(), optimizer.step()...

    if step % checkpoint_interval == 0:
        # --- 非阻塞保存启动 ---
        checkpoint_path = f"/tmp/async_checkpoints/step_{step}.pt"
        save_asynchronous_checkpoint(
            0, 1, model, optimizer, scheduler, step, checkpoint_path
        )
        # 为简单起见，假设节点0，world_size为1

# 等待循环退出后最后一个检查点线程完成
if checkpoint_thread is not None and checkpoint_thread.is_alive():
    print("主循环：等待最终检查点完成...")
    checkpoint_thread.join()
print("模拟训练循环完成。")
```

下图说明了异步检查点如何将I/O与计算重叠。

> 主训练线程仅短暂暂停以复制状态，然后继续计算，而实际保存则在后台线程中进行。

### 同步与异步检查点的选择

最佳方式取决于具体的训练设置和优先级：

- **简单性与性能：** 同步方式更简单，但开销更高。异步方式更复杂，但显著减少空闲时间。
- **存储速度：** 如果检查点存储极快（例如，高性能并行文件系统），同步保存的开销可能可以接受。如果存储I/O是瓶颈，异步保存则更具吸引力。
- **故障敏感度：** 如果故障时恢复绝对最新状态非常重要，同步检查点所保证的一致性可能更受青睐。如果损失少量迭代进度可以接受以换取更高的吞吐量 (throughput)，异步方式是一个不错的选择。
- **框架支持：** 现代分布式训练框架（如DeepSpeed或PyTorch FSDP）通常提供内置的优化检查点支持，有时包括异步选项或高度优化的同步方法，以最大限度地减少阻塞时间。通常建议利用这些框架功能。

实践中，对于检查点时间可能很长（数分钟或更久）的超大型模型，异步检查点通常更受青睐，以最大限度地利用昂贵的GPU资源，尽管增加了实现复杂性。细致的实现和测试是确保异步保存过程可靠性所必需的。

## 参考资料

- [Saving and Loading Models](https://pytorch.org/docs/stable/notes/serialization.html) — PyTorch Contributors (2017)
  提供了在PyTorch中保存和加载模型状态、优化器状态及其他组件的基础方法，是理解检查点实现的基础。
- [DeepSpeed: Advanced Checkpointing](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGPZ9X3reh0uxmmu0Rv3ihs5fUcMRQ6OTky7jWcfRVIDIHLzdvTjwFBlnUMyOBtQ3xbuL_O_KD7dj18l0_XQSAo2btNaGyZy4E0b9Ilxiup8SOoFLKBNuswsISL-fGhnaae8SneNhjfYGpeoH3Mt1-D6FSJSgcr) — DeepSpeed Team (2024)
  讨论了DeepSpeed框架中优化的检查点策略，DeepSpeed广泛用于大型语言模型训练，通常包含高效的同步或异步机制。
- [Hydra: Understanding and Improving Distributed Checkpointing in Deep Learning](https://dl.acm.org/doi/10.1145/3550209.3552097) — Karki, Hritik and Narayanasamy, Sanjeev and Shah, Nirmit and Chen, Bo and Wang, Yuandong and D'Sa, Renju and Agarwal, Sachin and Chen, Chien-Chung and Chintapalli, Srinivas (2022)
  Journal: SC'22: International Conference for High Performance Computing, Networking, Storage and Analysis; Publisher: ACM; Pages: 1-13; DOI: [10.1145/3550209.3552097](https://doi.org/10.1145/3550209.3552097)
  该论文分析并提出了分布式检查点方法的改进方案，包括管理一致性和性能权衡的策略，直接解决了同步与异步选择的难题。
- [Tesseract: A Two-Level Checkpointing Protocol for Large-Scale Deep Learning](https://dl.acm.org/doi/10.1145/3575693.3575702) — Kang, Yu and Zhang, Peifeng and Yang, Hong and Wang, Jiaqi and Zhu, Yanyuan and Wu, You and Zhang, Wei and Liu, Yong and Tian, Jin (2023)
  Journal: Proceedings of the 28th ACM International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS); Publisher: Association for Computing Machinery; Pages: 1064-1077; DOI: [10.1145/3575693.3575702](https://doi.org/10.1145/3575693.3575702)
  介绍了一种新颖的两级检查点协议，旨在提高大规模深度学习的效率，通过结合同步和异步方法的优势来应对它们的挑战。
