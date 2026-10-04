---
course: "advanced-python-programming-ml"
chapter: "advanced-python-constructs-ml-pipelines"
lesson: "context-managers-ml-resources"
sourceId: 2239
sourceUrl: "https://apxml.com/zh/courses/advanced-python-programming-ml/chapter-1-advanced-python-constructs-ml-pipelines/context-managers-ml-resources"
title: "ML工作流中的资源管理上下文管理器"
description: "在Python中实现自定义上下文管理器，用于管理ML管道中的文件、模型和连接等资源。"
order: 2
plots: []
sourceHash: "4306339955de6608037591e47135e0ee4883d5a00e3ac35d190ec354fb6d8d55"
sourceCorrections: []
---

机器学习 (machine learning)管道经常与外部资源进行交互：磁盘上的数据集、到数据库或特征存储的连接、用于中间结果的临时文件，甚至专用硬件会话。正确管理这些资源对于构建可靠的系统非常重要。未能释放文件句柄或网络连接等资源可能导致内存泄漏、性能下降甚至崩溃，尤其是在长时间运行的训练任务或持续运行的推理 (inference)服务中。

Python的上下文 (context)管理器提供了一种优雅而有效的资源管理方式。它们自动化了与资源相关的设置和清理阶段，确保即使操作中发生错误，也能可靠地执行清理动作，例如关闭文件或释放锁。使用上下文管理器的主要方式是通过 `with` 语句。

### `with` 语句和上下文 (context)管理协议

您可能对在文件操作中使用 `with` 语句很熟悉：

```python
# 确保文件关闭的标准方法
try:
    f = open('data.csv', 'r')
    data = f.read()
    # 处理数据...
except Exception as e:
    print(f"发生错误: {e}")
finally:
    if 'f' in locals() and not f.closed:
        f.close()

# 使用上下文管理器的等效方法
try:
    with open('data.csv', 'r') as f:
        data = f.read()
        # 处理数据...
except Exception as e:
    print(f"发生错误: {e}")

# 文件 f 会在此处自动关闭，
# 无论代码块内是否发生异常。
```

`with` 语句简化了 `try...finally` 模式。任何支持*上下文管理协议*的对象都可以与 `with` 一同使用。该协议包含两个特殊方法：

1. `__enter__(self)`: 在进入 `with` 块时执行。它执行设置操作，通常返回一个对象，该对象将被绑定到 `as` 子句中指定的变量（如果未使用 `as` 子句，则为 `None`）。
2. `__exit__(self, exc_type, exc_val, traceback)`: 在退出 `with` 块时执行，无论是正常退出还是因异常退出。它执行清理操作。
   - 如果代码块在没有异常的情况下完成，`exc_type`、`exc_val` 和 `traceback` 都将是 `None`。
   - 如果在 `with` 块内发生异常，这些参数 (parameter)将接收异常详情。`__exit__` 方法可以处理该异常（例如，记录它），并可通过返回 `True` 来选择性地抑制它。如果它返回 `False`（或隐式返回 `None`），则在 `__exit__` 完成后，异常会再次引发。

正是这种 `__exit__` 的保证执行，使得上下文管理器在资源管理中非常有价值。

### ML 工作流中的内置上下文 (context)管理器

除了文件处理 (`open`)，Python 的标准库还提供了其他有用的上下文管理器。例如，`tempfile.TemporaryDirectory` 对于在管道阶段创建所需的临时存储很方便，它能确保在使用后删除该目录及其内容：

```python
import tempfile
import os
import pandas as pd

# 模拟一些需要临时存储的数据处理
data = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})

try:
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"已创建临时目录: {temp_dir}")
        temp_file_path = os.path.join(temp_dir, 'intermediate_data.parquet')

        # 保存中间结果
        data.to_parquet(temp_file_path)
        print(f"已将中间数据保存到 {temp_file_path}")

        # 加载并执行进一步处理...
        loaded_data = pd.read_parquet(temp_file_path)
        print("已为下一步加载中间数据。")

    # 在 'with' 块之外:
    print(f"临时目录 {temp_dir} 存在吗？ {os.path.exists(temp_dir)}")

except Exception as e:
    print(f"处理过程中发生错误: {e}")

# 输出可能如下:
# 已创建临时目录: /tmp/tmpxxxxxxx
# 已将中间数据保存到 /tmp/tmpxxxxxxx/intermediate_data.parquet
# 已为下一步加载中间数据。
# 临时目录 /tmp/tmpxxxxxxx 存在吗？ False
```

临时目录在 `with` 块退出时会自动清理，防止杂乱并避免潜在的磁盘空间问题。

### 为 ML 任务创建自定义上下文 (context)管理器

虽然内置上下文管理器涵盖了常见情况，但 ML 管道通常有特定的资源管理需求。您可能需要：

- 测量特定管道阶段的时间。
- 管理与实验跟踪服务（如 MLflow 或 Weights & Biases）的连接。
- 确保自定义硬件接口的设置和清理。
- 记录复杂操作的进入和退出点。

您可以通过两种主要方式创建自定义上下文管理器：使用包含 `__enter__` 和 `__exit__` 方法的类，或者使用 `contextlib.contextmanager` 装饰器。

#### 基于类的自定义上下文管理器

直接实现 `__enter__` 和 `__exit__` 方法可以为您提供完全的控制。让我们创建一个上下文管理器，以记录管道阶段的开始和结束，并测量其持续时间。

```python
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class PipelineStageTimer:
    def __init__(self, stage_name):
        self.stage_name = stage_name
        self.start_time = None

    def __enter__(self):
        logging.info(f"进入阶段: {self.stage_name}")
        self.start_time = time.perf_counter()
        # 无需返回特定内容，可以返回 self 或 None
        return self

    def __exit__(self, exc_type, exc_val, traceback):
        end_time = time.perf_counter()
        duration = end_time - self.start_time
        if exc_type:
            logging.error(f"退出阶段: {self.stage_name} (失败，耗时 {duration:.4f}秒)")
            # 可选：在此处处理异常日志
            # 返回 False (或 None) 以传播异常
            return False
        else:
            logging.info(f"退出阶段: {self.stage_name} (完成，耗时 {duration:.4f}秒)")
        # 如果想抑制异常，则返回 True（对于计时器不常见）
        return False

# 在管道中的使用
def load_data(path):
    print(f"   正在从 {path} 加载数据...")
    time.sleep(0.5) # 模拟工作
    return "一些数据"

def preprocess_data(data):
    print(f"   正在预处理数据...")
    time.sleep(1.0) # 模拟工作
    # 取消注释以模拟错误
    # raise ValueError("预处理时出现问题")
    return "已处理的数据"

def train_model(data):
    print(f"   正在训练模型...")
    time.sleep(1.5) # 模拟工作
    return "已训练的模型"

try:
    with PipelineStageTimer("数据加载"):
        data = load_data("dataset.csv")

    with PipelineStageTimer("预处理"):
        processed_data = preprocess_data(data)

    with PipelineStageTimer("模型训练"):
        model = train_model(processed_data)

    print("\n管道成功完成。")

except Exception as e:
    print(f"\n管道失败: {e}")


# 示例输出 (无错误):
# 2023-10-27 10:30:00,123 - 进入阶段: 数据加载
#    正在从 dataset.csv 加载数据...
# 2023-10-27 10:30:00,624 - 退出阶段: 数据加载 (完成，耗时 0.5010秒)
# 2023-10-27 10:30:00,624 - 进入阶段: 预处理
#    正在预处理数据...
# 2023-10-27 10:30:01,626 - 退出阶段: 预处理 (完成，耗时 1.0020秒)
# 2023-10-27 10:30:01,626 - 进入阶段: 模型训练
#    正在训练模型...
# 2023-10-27 10:30:03,128 - 退出阶段: 模型训练 (完成，耗时 1.5020秒)
#
# 管道成功完成。

# 示例输出 (预处理中出现错误):
# 2023-10-27 10:35:00,123 - 进入阶段: 数据加载
#    正在从 dataset.csv 加载数据...
# 2023-10-27 10:35:00,624 - 退出阶段: 数据加载 (完成，耗时 0.5010秒)
# 2023-10-27 10:35:00,624 - 进入阶段: 预处理
#    正在预处理数据...
# 2023-10-27 10:35:01,626 - 退出阶段: 预处理 (失败，耗时 1.0020秒)
#
# 管道失败: 预处理时出现问题
```

这个计时器能可靠地记录进入、退出和持续时间，并在托管块内发生异常时正确报告失败。

#### 使用 `contextlib.contextmanager`

`contextlib` 模块提供了一个方便的装饰器 `contextmanager`，它允许您使用生成器函数实现上下文管理器。这通常能为简单的设置/清理逻辑带来更简洁的代码。

生成器应：

1. 在 `yield` 语句之前执行设置操作。
2. `yield` 且仅 `yield` 一次。`yield` 的值将成为 `__enter__` 方法的结果（绑定到 `as` 变量）。
3. 在 `yield` 语句之后执行清理操作。此代码在 `with` 块退出时运行。`yield` 周围的异常处理会自动进行。

让我们使用这种方法重写 `PipelineStageTimer`：

```python
import time
import logging
from contextlib import contextmanager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

@contextmanager
def pipeline_stage_timer(stage_name):
    logging.info(f"进入阶段: {stage_name}")
    start_time = time.perf_counter()
    try:
        # 将控制权返回给 'with' 块。
        # 在此情况下无需 yield 特定内容。
        yield
    except Exception:
        # 如果 'with' 块内发生异常，则执行此块
        end_time = time.perf_counter()
        duration = end_time - start_time
        logging.error(f"退出阶段: {stage_name} (失败，耗时 {duration:.4f}秒)")
        raise # 再次引发异常以传播它
    else:
        # 如果 'with' 块成功完成，则执行此块
        end_time = time.perf_counter()
        duration = end_time - start_time
        logging.info(f"退出阶段: {stage_name} (完成，耗时 {duration:.4f}秒)")
    finally:
        # 无论成功或失败（在 except/else 之后），此处的代码都会运行
        # 需要时可用于最终清理，但清理通常在 except/else 中完成
        pass

# 用法与基于类的示例相同:
try:
    with pipeline_stage_timer("数据加载"):
        data = load_data("dataset.csv")

    with pipeline_stage_timer("预处理"):
        processed_data = preprocess_data(data) # 在此处尝试有无错误

    with pipeline_stage_timer("模型训练"):
        model = train_model(processed_data)

    print("\n管道成功完成。")

except Exception as e:
    print(f"\n管道失败: {e}")
```

行为与基于类的版本相同，但使用 `@contextmanager` 的实现对于直接的设置/yield/清理模式来说，可读性可能更高。

### ML 中的高级使用场景

在处理复杂 ML 工作流中常见的有状态或外部服务时，上下文 (context)管理器变得尤其强大：

- **数据库连接：** 确保数据库连接在事务前打开，并在之后可靠地关闭或提交/回滚。
- **API 会话：** 管理与云服务（例如 AWS S3、Google Cloud Storage）或实验跟踪平台（MLflow、W&B）的会话，确保正确的身份验证设置和清理。
- **远程资源：** 在分布式计算环境中获取和释放锁或资源。
- **模型加载：** 确保模型（特别是大型模型）加载到内存或特定设备（如 GPU）上，然后正确卸载或移回 CPU 内存以释放资源。

考虑管理 MLflow 运行上下文：

```python
# 示例 - 实际的 MLflow API 可能略有不同
import mlflow

@contextmanager
def mlflow_run_manager(experiment_name, run_name):
    mlflow.set_experiment(experiment_name)
    try:
        with mlflow.start_run(run_name=run_name) as run:
            print(f"MLflow 运行已开始: {run.info.run_id}")
            yield run # 将运行对象提供给 'with' 块
        print(f"MLflow 运行 {run.info.run_id} 成功完成。")
    except Exception as e:
        print(f"MLflow 运行失败: {e}")
        # MLflow 通常在 start_run 内部出现异常时处理运行结束
        raise # 再次引发异常


# 用法:
try:
    with mlflow_run_manager("我的实验", "训练运行 - 试验 1") as current_run:
        # 记录参数
        mlflow.log_param("learning_rate", 0.01)
        mlflow.log_param("epochs", 10)

        # 模拟训练和记录指标
        print("   正在模拟训练...")
        time.sleep(1)
        mlflow.log_metric("accuracy", 0.95, step=10)
        mlflow.log_metric("loss", 0.15, step=10)

        # 记录一个工件（例如，模型）
        # mlflow.sklearn.log_model(...)

except Exception as e:
    print(f"实验失败: {e}")
```

这个自定义上下文管理器封装了设置实验和启动运行的样板代码，确保运行上下文得到妥善处理。

### 总结

上下文 (context)管理器通过 `with` 语句访问，是编写可靠 Python 代码的基础工具，尤其是在机器学习 (machine learning)管道这类资源密集型应用中。它们保证了清理代码的执行，简化了与资源管理相关的错误处理，并通过清晰界定资源活跃范围来提高代码可读性。无论是使用 `open` 和 `TemporaryDirectory` 等内置管理器，还是为计时、日志记录或与外部服务交互等特定任务创建自定义管理器，掌握上下文管理器都是构建专业级别 ML 系统的重要一步。

## 参考资料

- [The \`with\` statement and \`contextlib\` module](https://docs.python.org/3/reference/compound_stmts.html#the-with-statement) — Python Software Foundation (2023)
  Publisher: Python Software Foundation
  官方文档，详细介绍了`with`语句、上下文管理协议（`__enter__`、`__exit__`）以及用于简化上下文管理器创建的`contextlib`模块。
- [Fluent Python: Clear, Concise, and Effective Programming](https://play.google.com/store/books/details/Fluent_Python_Clear_Concise_and_Effective_Programm?id=y91dEAAAQBAJ) — Luciano Ramalho (2022)
  Publisher: O'Reilly Media
  一本关于高级Python功能的综合指南，其中包含专门章节，详细解释和实际演示了上下文管理器。
- [mlflow.start_run](https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.start_run) — MLflow Developers (2023)
  `mlflow.start_run`的官方API参考，它是用于管理MLflow实验运行和日志记录的常用上下文管理器。
- [Effective Python: 90 Specific Ways to Write Better Python](https://www.informit.com/store/effective-python-90-specific-ways-to-write-better-9780134853987) — Brett Slatkin (2019)
  Publisher: Addison-Wesley
  提供了编写Python代码的具体指导和最佳实践，包括关于使用上下文管理器进行可靠资源处理的建议。
