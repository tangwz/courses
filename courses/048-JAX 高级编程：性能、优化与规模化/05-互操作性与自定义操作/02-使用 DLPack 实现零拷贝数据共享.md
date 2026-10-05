# 使用 DLPack 实现零拷贝数据共享

来源：[原文](https://apxml.com/zh/courses/advanced-jax/chapter-5-jax-interoperability-custom-operations/zero-copy-dlpack)

[返回章节目录](README.md) · [返回课程目录](../README.md)

大型科学计算环境中，经常需要 JAX 与 NumPy、PyTorch、TensorFlow 或 CuPy 等其他数组或张量库之间传递数据。一种简单的方法是复制数据：将 JAX 数组转换为 CPU 上的 NumPy 数组，然后将该 NumPy 数组转换为例如 PyTorch 张量，并可能将其移回 GPU。这种 CPU 往返和内存重复会带来很大的性能开销，特别是对于驻留在 GPU 等加速器上的大型数组。

设想一个场景，JAX 和 PyTorch 都在同一个 GPU 上运行。如果你有一个 PyTorch 计算出的大张量，想要在 JAX 函数中使用（反之亦然），在 *同一个设备* 上，在这些框架的内存分配之间物理复制千兆字节的数据是非常低效的。这会消耗时间和宝贵的内存带宽。

这就是 DLPack 标准发挥作用的地方。DLPack 定义了一种通用的、语言无关的内存中张量数据结构规范。支持 DLPack 的库可以在数据位于同一设备上的情况下，不执行任何内存复制就交换张量数据。它们本质上是交换指向底层内存缓冲区的指针，以及描述张量（形状、数据类型、步幅、设备）的元数据。

### DLPack 机制

可以将 DLPack 看作是库之间关于如何在内存中描述张量的一项约定。当你希望从一个 DLPack 兼容的库（如 JAX）导出张量时，你会请求一个 DLPack “胶囊”（capsule）。这个胶囊通常是一个轻量级对象（在 Python 中，通常是 `PyCapsule`），它包含：

1. 指向张量数据实际所在内存缓冲区（例如，GPU 内存地址）的指针。
2. 描述张量的元数据：
   - 设备类型（CPU、CUDA GPU、ROCm GPU 等）和设备 ID。
   - 数据类型（float32、int64 等）。
   - 维度数量 (ndim)。
   - 形状（维度大小的元组）。
   - 步幅（沿每个维度移动所需的字节步长）。
   - 缓冲区起始位置的字节偏移。

当另一个 DLPack 兼容的库接收到这个胶囊时，它会读取指针和元数据。然后，它可以包装这个现有的内存缓冲区，创建自己的张量对象（例如 `jax.Array` 或 `torch.Tensor`），直接使用数据，*而无需复制*。

这个过程通常被称为“零拷贝”共享，因为主要张量数据没有发生任何重复。

### JAX 使用 DLPack

JAX 在 `jax.dlpack` 模块中提供函数，以便利这种交换。

#### 从 JAX 导出：`to_dlpack`

要与另一个库共享 JAX 数组，请使用 `jax.dlpack.to_dlpack()`：

```python
import jax
import jax.numpy as jnp
import torch # 示例：使用 PyTorch
import cupy # 示例：使用 CuPy

# 确保 JAX 正在使用 GPU（如果可用）
try:
    _ = jax.devices('gpu')[0]
    print("JAX 正在使用 GPU。")
except RuntimeError:
    print("JAX 正在使用 CPU。DLPack GPU 共享需要 GPU。")
    # 针对此示例，我们将继续使用 CPU，但请注意其限制。

# 在默认设备上创建 JAX 数组（理想情况下是 GPU）
key = jax.random.PRNGKey(0)
jax_array_gpu = jax.random.normal(key, (1024, 1024), device=jax.devices()[0])
print(f"原始 JAX 数组设备：{jax_array_gpu.device()}")

# 将 JAX 数组导出为 DLPack 胶囊
# 该胶囊需要被明确使用或删除
dlpack_capsule = jax.dlpack.to_dlpack(jax_array_gpu)

print(f"DLPack 胶囊已创建：{type(dlpack_capsule)}")

# --- 现在，导入到另一个库 ---

# 示例：导入到 PyTorch
# 需要支持 from_dlpack 的 PyTorch 版本
try:
    torch_tensor_shared = torch.from_dlpack(dlpack_capsule)
    print(f"通过 DLPack 创建的 PyTorch 张量。设备：{torch_tensor_shared.device}")
    # 验证数据共享（可选，检查内存指针或修改）
    # 注意：JAX 数组是不可变的，修改 torch_tensor_shared 可能会出错
    # 或者如果 PyTorch 创建了可变视图，则可能会修改缓冲区。
    # 如果可能进行修改，请注意潜在的别名问题。
    print(f"PyTorch 张量共享内存：{torch_tensor_shared.data_ptr() == jax_array_gpu.device_buffer.unsafe_buffer_pointer()}")

    # 重要：使用胶囊会使其失效。
    # 再次尝试使用 dlpack_capsule 可能会失败。
    try:
        another_tensor = torch.from_dlpack(dlpack_capsule)
    except Exception as e:
        print(f"\n按预期，尝试重用胶囊失败：{e}")

except ImportError:
    print("\nPyTorch 未安装或版本过旧，不支持 DLPack。")
except TypeError as e:
     print(f"\n将 DLPack 导入 PyTorch 时出错：{e}。通常表示胶囊已被使用。")
except RuntimeError as e:
     print(f"\n将 DLPack 导入 PyTorch 时运行时错误：{e}。通常表示设备不匹配或胶囊已被使用。")

# 示例：再次导出并导入到 CuPy（如果可用）
# 需要安装 CuPy
try:
    # 重新导出，因为之前的胶囊已被使用
    dlpack_capsule_for_cupy = jax.dlpack.to_dlpack(jax_array_gpu)
    cupy_array_shared = cupy.from_dlpack(dlpack_capsule_for_cupy)
    print(f"\n通过 DLPack 创建的 CuPy 数组。设备：{cupy_array_shared.device}")
    print(f"CuPy 数组共享内存：{cupy_array_shared.data.ptr == jax_array_gpu.device_buffer.unsafe_buffer_pointer()}")
    # 胶囊在此处也被使用。
except ImportError:
    print("\nCuPy 未安装。")
except Exception as e:
    print(f"\nCuPy DLPack 导入期间出错：{e}")

# 清理原始数组（可选）
del jax_array_gpu
# 注意：内存可能仍由 torch_tensor_shared 或 cupy_array_shared 持有
```

关于 `to_dlpack` 的重要说明：

- 它以 JAX 数组作为输入。
- 它返回一个 `PyCapsule`。
- 该胶囊只能被 *使用一次*。使用它的库（如 `torch.from_dlpack`）通常在成功导入后使胶囊失效。

#### 导入到 JAX：`from_dlpack`

要通过 DLPack 从其他库拥有的数据创建 JAX 数组，请使用 `jax.dlpack.from_dlpack()`：

```python
import jax
import jax.numpy as jnp
import torch
import numpy as np # 针对 CPU 示例

# 确保 PyTorch 使用 JAX 打算使用的同一设备
if torch.cuda.is_available():
    pytorch_device = torch.device('cuda')
    jax_device = jax.devices('gpu')[0]
    print(f"PyTorch 正在使用设备: {pytorch_device}")
    print(f"JAX 目标设备: {jax_device}")
    # 简单检查，假设两者都使用设备 0。在生产环境中请更严谨。
    assert str(pytorch_device) == f"cuda:{jax_device.id}", "PyTorch 和 JAX 必须使用相同的 GPU 设备 ID。"
else:
    pytorch_device = torch.device('cpu')
    jax_device = jax.devices('cpu')[0]
    print("PyTorch 和 JAX 正在使用 CPU。")

# 创建 PyTorch 张量
torch_tensor = torch.randn(512, 512, device=pytorch_device) * 10
print(f"\n原始 PyTorch 张量设备: {torch_tensor.device}")

# 将 PyTorch 张量导出为 DLPack
# 新版本请使用 torch.to_dlpack
try:
    # PyTorch >= 1.7
    pt_dlpack_capsule = torch.to_dlpack(torch_tensor)
    print(f"PyTorch DLPack 胶囊已创建: {type(pt_dlpack_capsule)}")
except AttributeError:
    # 较旧的 PyTorch 版本可能需要不同的语法或支持不佳。
    print("未找到 torch.to_dlpack。请更新 PyTorch 或查阅旧版本文档。")
    pt_dlpack_capsule = None

if pt_dlpack_capsule:
    # 导入到 JAX
    try:
        jax_array_shared = jax.dlpack.from_dlpack(pt_dlpack_capsule)
        print(f"通过 DLPack 创建的 JAX 数组。设备: {jax_array_shared.device()}")

        # 验证数据共享和内容
        print(f"JAX 数组共享内存: {torch_tensor.data_ptr() == jax_array_shared.device_buffer.unsafe_buffer_pointer()}")
        # 检查值是否近似相等（浮点比较）
        # 可能需要将 JAX 数组移至 CPU 进行 numpy 转换，
        # 或将 torch 张量转换为 numpy。我们通过 JAX 操作在设备上进行比较。
        diff = jnp.abs(jax_array_shared - jnp.array(torch_tensor.cpu().numpy())).max() # 示例：通过 numpy 桥接进行比较
        # 更好的方法可能是如果尺寸较小，则将 torch 张量转换为 numpy，然后再转换为 JAX
        # 或者尽可能在设备上只使用 torch/jax 操作进行比较。

        print(f"最大绝对差值: {diff}")
        assert diff < 1e-6, "DLPack 传输后数据不匹配"

        # 胶囊已被 from_dlpack 使用
        try:
            another_jax_array = jax.dlpack.from_dlpack(pt_dlpack_capsule)
        except Exception as e:
            print(f"\n按预期，尝试重用胶囊失败: {e}")

    except Exception as e:
        print(f"\n将 DLPack 导入 JAX 时出错: {e}")

# 清理原始张量（可选）
del torch_tensor
# 内存可能仍由 jax_array_shared 持有
```

关于 `from_dlpack` 的重要说明：

- 它以 `PyCapsule`（从另一个库的 DLPack 导出函数获得）作为输入。
- 它返回一个标准的 `jax.Array`。
- 它会使用胶囊，该胶囊不能重复使用。
- 生成的 JAX 数组直接使用由原始库管理的内存缓冲区。

### 零拷贝共享的可视化

此图显示了该原理：

> 此图显示框架 A（如 JAX）将其指向 GPU 内存缓冲区的张量导出到 DLPack 胶囊中。然后，框架 B（如 PyTorch）从该胶囊导入，创建自己的张量对象，该对象指向 *相同* 的 GPU 内存缓冲区，从而避免了数据复制。

### 重要考量

尽管 DLPack 能够实现高效的数据共享，但请注意以下几点：

1. **设备兼容性：** 零拷贝共享仅在两个框架使用 *完全相同* 的物理设备时才有效。你不能在 JAX 中直接使用 DLPack 将 `gpu:0` 上的张量与 PyTorch 中打算用于 `gpu:1` 的张量共享，除非先在其中一个框架内进行明确的设备传输。同样，通过 DLPack 直接在 CPU 和 GPU 内存之间共享数据是不可能的；数据在导出前必须驻留在目标设备上。
2. **数据同步：** JAX 操作在加速器上异步执行。在使用 `to_dlpack` 导出 JAX 数组之前，请通过在该数组或其计算图中的任一父节点上调用 `.block_until_ready()` 来确保生成该数组的任何计算都已完成。类似地，如果将刚刚在另一个框架（例如，PyTorch 使用 CUDA）中计算的数据导入 JAX，请确保在调用 `jax.dlpack.from_dlpack` 之前该框架的计算流已同步。未能同步可能导致竞态条件或使用不完整/不正确的数据。
3. **生命周期和所有权：** 内存缓冲区通常由分配它的原始框架拥有。DLPack 胶囊提供临时访问。原始张量对象（例如，你导出的 PyTorch 张量）通常必须在 DLPack 胶囊或从其导入的对象（例如，生成的 JAX 数组）使用期间保持活动。一旦原始张量被删除，内存可能会被释放，如果导入的对象仍在使用，这将导致无效指针和崩溃。使用函数（`from_dlpack`）通常会使胶囊失效，防止意外重复使用。
4. **可变性：** JAX 数组是不可变的。当你通过 `jax.dlpack.from_dlpack` 导入数据时，生成的 JAX 数组会遵循这一不可变性。即使源库中的原始张量（例如 PyTorch）是可变的，尝试对 JAX 数组进行原地修改也会失败。相反，`jax.dlpack.to_dlpack` 导出的是 JAX 数组数据的视图。尽管 DLPack 标准规定了指示读/写访问的条款，但 JAX 通常导出只读视图，以反映其自身的不可变原则。如果接收框架 *可能* 修改底层缓冲区，请务必小心，因为这会破坏 JAX 的假设，尽管通常会有保护措施。
5. **流顺序（高级）：** 在 GPU 上，操作会发送到计算流。如果 JAX 和另一个库（如 PyTorch）在不同的 CUDA 流上操作，那么仅仅调用 `block_until_ready()` 或 `torch.cuda.synchronize()` 可能不足以保证数据通过 DLPack 传输时的可见性，除非进行了适当的流事件处理。对于基本的顺序使用（在框架 A 中计算，同步，传递给框架 B，在 B 中计算），框架通常会正确处理。然而，在复杂、高度并发的场景中，为了在使用 DLPack 的框架之间实现可靠的同步，可能需要仔细管理 CUDA 事件。

使用 DLPack 是一种有效方法，用于消除在同一加速器上运行的兼容库之间不必要的数据复制。通过了解它的工作方式及其在设备放置、同步和对象生命周期方面的相关要求，你可以大大提高 Python 科学计算栈中涉及多个框架的工作流的性能和效率。

## 参考资料

- [DLPack: A Common Tensor Exchange Format](https://github.com/dmlc/dlpack) — DMLC Project Developers (2023)
  定义DLPack C ABI规范的官方仓库，用于零拷贝张量交换。
- [DLPack utilities (jax.dlpack)](https://jax.readthedocs.io/en/latest/jax.dlpack.html) — The JAX authors (2024)
  Publisher: The JAX authors
  JAX官方文档，介绍其DLPack模块对框架间张量共享的支持。
- [Tensor interoperability with DLPack](https://pytorch.org/docs/stable/dlpack.html) — PyTorch developers (2024)
  PyTorch官方文档，详述使用DLPack标准导出和导入张量的函数。
- [cupy.from_dlpack](https://docs.cupy.dev/en/stable/reference/generated/cupy.from_dlpack.html) — CuPy developers (2024)
  CuPy官方文档，说明如何从DLPack胶囊导入张量。

---

[上一节](01-JAX%20%E4%B8%8E%20NumPy%20%E7%9A%84%E9%9B%86%E6%88%90.md) · [下一节](03-%E4%BD%BF%E7%94%A8%20jax.experimental.host_callback%20%E8%B0%83%E7%94%A8%E5%A4%96%E9%83%A8%20CPU-GPU%20%E4%BB%A3%E7%A0%81.md)
