---
course: "building-advanced-llm-agent-tools"
chapter: "advanced-tool-functionality"
lesson: "code-execution-tools"
sourceId: 6600
sourceUrl: "https://apxml.com/zh/courses/building-advanced-llm-agent-tools/chapter-5-advanced-tool-functionality/code-execution-tools"
title: "代码解释与执行工具"
description: "学习构建允许LLM执行代码的工具，着重于沙盒和安全措施。"
order: 1
plots: []
sourceHash: "91fab2c4e4621c655c5ea385f2020a5bbc68bd39f4407049f324d36525f72019"
sourceCorrections: []
---

赋予您的LLM代理解释和执行代码的能力，开辟了新的问题解决能力范围。想象一下，一个代理不仅能理解用户对复杂计算的请求，还能编写必要的Python脚本，执行它，并返回精确的数值结果。这是要增加的功能。然而，伴随强大功能而来的是重大的责任，在代码执行的背景下，“责任”直接意味着严格的安全措施。在此，提供了构建代码解释和执行工具的指导，始终坚持对安全和沙盒的关注。

尽管LLM擅长生成代码片段，但它们本身缺乏运行这些代码的环境。代码执行工具作为这种缺失的运行时环境，使LLM能够执行以下任务：

- **数值计算：** 进行统计分析、解决数学方程或运行模拟。
- **数据处理：** 处理文本、转换数据结构或为其他工具准备数据。
- **动态内容生成：** 根据计算数据创建图表、示意图或其他输出（尽管实际的渲染可能由另一个专用工具处理，或通过返回数据进行渲染）。
- **算法任务：** 实现并运行特定算法，以解决难以声明性表达的问题。

其核心在于，执行LLM提供的一段代码字符串，就是将该字符串在解释器中运行，在LLM代理的场景下，最常见的是Python。

### 挑战：任意代码执行

最直接也是最重要的挑战是，从定义上来说，您正在启用任意代码执行。如果LLM可以被提示编写任何代码，而您的工具在没有保护措施的情况下执行它，您的系统将容易受到多种攻击。恶意代码可能：

- 访问主机系统上的敏感文件。
- 将数据传送到外部服务器。
- 消耗过多的系统资源（CPU、内存、磁盘），导致拒绝服务。
- 尝试损害您基础设施的其他部分。

因此，**切勿在不安全的环境中，直接对未经净化的LLM生成代码使用诸如Python的`exec()`或`eval()`等函数。** 这些函数在与您的主应用程序相同的进程中执行代码，不提供任何隔离。

运行外部脚本的一个稍好一些的方法是使用Python的`subprocess`模块。它允许您在新进程中运行命令，提供一定程度的隔离。然而，`subprocess`本身并不构成一个完整的沙盒。新进程仍然继承了可能过于宽泛的权限和访问权利。

```python
# 警告：这是一个简化示例，缺乏适当的沙盒机制。
# 在没有强大隔离措施的情况下，请勿直接在生产环境中使用。
import subprocess

def execute_python_code_subprocess(code_string: str, timeout_seconds: int = 5):
    try:
        process = subprocess.run(
            ['python', '-c', code_string],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False # 不自动引发 CalledProcessError
        )
        if process.returncode == 0:
            return {"stdout": process.stdout, "stderr": "", "status": "success"}
        else:
            return {"stdout": process.stdout, "stderr": process.stderr, "status": "error"}
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": "执行超时。", "status": "timeout"}
    except Exception as e:
        return {"stdout": "", "stderr": f"发生意外错误：{str(e)}", "status": "error"}

# 示例用法（说明性）
# code_to_run = "print('Hello from sandboxed code!')\nimport sys; sys.exit(0)"
# result = execute_python_code_subprocess(code_to_run)
# print(result)
```

上述示例使用`subprocess.run`来执行Python代码字符串。它包含了超时设置并捕获了`stdout`和`stderr`。尽管`subprocess`将执行隔离到单独的进程中，但该进程通常仍然以与父Python脚本相同的用户权限运行，并拥有对相同网络和文件系统的访问权限，除非受到操作系统级别控制的进一步限制。这不足以安全地运行LLM生成的代码。

### 沙盒的必要性

沙盒是一个隔离的、受限的环境，其中可以以最小风险执行不受信任的代码，以保护宿主系统或其他应用程序。对于LLM代码执行工具而言，强大的沙盒机制并非可选项；它是一项基本的要求。

沙盒的主要目标是：

- **隔离：** 阻止执行的代码访问或修改其指定环境之外的任何内容。
- **资源控制：** 限制代码可以消耗的CPU、内存、磁盘空间和网络带宽量。
- **权限限制：** 确保代码以最低限度的必要权限运行。

可以采用多种沙盒技术，通常结合使用：

#### 1. 容器化（例如Docker）

Docker等容器化技术是创建隔离环境的一种流行且有效的方式。您可以将一个最小的Python运行时（或其他语言解释器）打包到容器镜像中。LLM的每个代码执行请求随后将启动一个新的、短生命周期的容器实例来运行代码。

使用容器的好处：

- **文件系统隔离：** 容器拥有自己隔离的文件系统。在其中运行的代码默认情况下无法访问宿主文件系统。
- **资源限制：** Docker允许您为每个容器指定CPU份额、内存限制和其他资源约束。
- **网络策略：** 您可以定义严格的网络策略，例如拒绝所有出站网络访问，或只允许访问特定白名单服务。
- **可复现性：** 容器确保执行环境每次都保持一致。

一个典型的工作流程将包括：

1. 您的工具从LLM接收代码。
2. 它使用Docker API（或命令行工具）从预构建镜像运行一个新容器。
3. 代码字符串被传入容器以执行。
4. 从容器中获取`stdout`、`stderr`和任何结果文件。
5. 容器被停止并移除。

#### 2. 操作系统级别沙盒

操作系统提供了各种限制进程的机制，例如：

- **`chroot` 隔离环境（类Unix系统）：** 更改进程及其子进程的根目录，限制文件系统可见性。
- **命名空间（Linux）：** 隔离进程ID、网络栈和挂载点等资源。容器大量依赖命名空间。
- **`seccomp`（Linux）：** 过滤进程可进行的系统调用，显著降低其能力。
- **AppArmor/SELinux（Linux）：** 强制访问控制系统，可以强制执行细粒度权限。
- **Windows作业对象 / AppContainers：** 提供在Windows上管理和限制进程组的方式。

直接实现这些可能很复杂，这就是为什么容器化（通常使用这些底层操作系统功能）是应用程序级沙盒更常见的方法。

#### 3. 特定语言的受限环境

一些语言提供了旨在执行带有限制的不受信任代码的库或模式。例如，Python曾有`RestrictedPython`，它试图限制对不安全属性和内置函数的访问。然而，这些语言级别的沙盒众所周知难以正确实现，并且随着新攻击向量 (vector)的发现，通常会随着时间被绕过。对于来自LLM的真正不受信任的代码，它们本身通常不被认为是足够的，如果使用，也应作为更强大的操作系统级别或基于容器的隔离的*补充*。

### 设计您的代码执行工具

一个设计良好的代码执行工具需要清晰的输入输出接口，以及强大的内部逻辑来安全地管理执行过程。

#### 工具的输入

您的工具接口（如果是一个服务，通常是一个API端点）应接受以下参数 (parameter)：

- `code`：一个包含要执行的源代码的字符串。
- `language`（可选）：一个字符串，指定编程语言（例如，“python”，“javascript”）。默认为您主要支持的语言。
- `timeout_seconds`（可选，但强烈推荐）：一个整数，指定进程终止前的最大执行时间。默认为一个安全的短时长（例如，5-10秒）。
- `allowed_modules`（可选，高级）：一个模块或库列表，代码被允许导入。这需要仔细管理和白名单。
- `network_access`（可选，高级）：一个布尔值或更细粒度的策略，指定代码是否以及如何访问网络。默认为`false`或“none”。

#### 工具的输出

工具应返回一个结构化响应，通常是JSON格式，指明结果：

- `status`：一个字符串，表示“成功”、“错误”、“超时”或“沙盒错误”。
- `stdout`：一个字符串，包含执行代码的标准输出。
- `stderr`：一个字符串，包含标准错误输出。这对于调试很重要。
- `result`（可选）：如果代码产生了一个可以被捕获的特定返回值（例如，脚本中最后一个表达式的值），则将其包含在此处。
- `execution_time_ms`：代码实际运行的时间。
- `error_message`（如果状态为“错误”）：一个可读的错误消息。

成功执行的JSON输出示例：

```json
{
  "status": "success",
  "stdout": "结果是：42\n",
  "stderr": "",
  "result": null,
  "execution_time_ms": 120
}
```

错误时的JSON输出示例：

```json
{
  "status": "error",
  "stdout": "",
  "stderr": "Traceback (most recent call last):\n  File \"<string>\", line 1, in <module>\nNameError: name 'pritn' is not defined\n",
  "result": null,
  "execution_time_ms": 50,
  "error_message": "执行期间发生 NameError。"
}
```

#### 设计中的安全考量

- **默认拒绝：** 默认拒绝所有潜在危险操作（例如，网络访问、指定临时区域外的文件系统写入）。仅在绝对必要且极其谨慎的情况下启用特定功能。
- **最小权限：** 沙盒环境和运行代码的进程应拥有绝对最低限度的所需权限。
- **资源配额：** 在沙盒内强制实施CPU时间、内存使用和磁盘空间的严格限制。
- **输入验证（针对参数）：** 尽管`code`本身不受信任，但要验证`timeout_seconds`等其他参数，以确保它们在合理范围内。
- **环境中无敏感信息：** 确保沙盒内没有API密钥、数据库凭证或其他敏感信息作为环境变量或文件存在，除非为特定受信任用例显式且安全地挂载（这在通用代码执行中很少见）。

### 与LLM代理的交互流程

1. **LLM决定使用工具：** 根据用户的查询或其内部计划，LLM判断需要执行代码。
2. **LLM生成代码和参数 (parameter)：** LLM生成代码字符串，并可能决定超时等参数（或您的代理框架设置默认值）。
3. **代理调用工具：** 代理用生成的代码和参数调用您的代码执行工具。
4. **工具在沙盒中执行代码：** 工具准备一个沙盒，运行代码，并收集`stdout`、`stderr`和任何结果。
5. **工具返回结构化输出：** 工具将结构化JSON响应发送回代理。
6. **LLM解释结果：** LLM解析响应。
   - 如果`status`是“成功”，它会在后续的推理 (inference)或响应生成中使用`stdout`或`result`。
   - 如果`status`是“错误”，它可以使用`stderr`来理解问题，可能尝试修复代码，或告知用户失败。
   - 如果`status`是“超时”，LLM知道代码运行时间过长，可以调整其策略。

`stderr`中清晰且信息丰富的错误消息非常重要。它们允许LLM可能调试自己的代码。例如，如果LLM生成带有`SyntaxError`的Python代码，`stderr`中的追溯信息将为LLM在后续尝试中纠正它提供所需的信息。

以下图表显示了代码执行工具的高级架构：

> LLM代理的代码执行请求，通过工具的API，进入受管的沙盒环境，并带着结果返回LLM的流程。

构建一个安全的代码执行工具是一项重要的工程任务。它需要仔细设计、沙盒机制的实现以及对安全的持续关注。尽管LLM编写和运行代码的吸引力很大，但潜在风险必须勤奋地管理。对于许多应用程序而言，使用为多租户和安全性设计的现有、经过强化的“代码解释器”或“Notebook”服务可能比从头构建一切更实用，除非您有特定、复杂的要求和资源来维护此类重要的基础设施。

在开发执行LLM生成代码的工具时，请始终将安全放在首位。从最严格的沙盒开始，并且只在谨慎且经过充分审查后才开放功能。请记住，在代码生成方面，LLM是一个不受信任的方，而您的工具是负责保护您系统的守门员。

## 参考资料

- [Best practices for security](https://docs.docker.com/engine/security/) — Docker Documentation (2024)
  提供了保护 Docker 守护程序、容器和镜像的指南，这对于为不可信代码执行构建强大的基于容器的沙箱至关重要。
- [Security Engineering: A Guide to Building Dependable Distributed Systems](https://www.cl.cam.ac.uk/~rja14/book.html) — Ross J. Anderson (2008)
  Publisher: Wiley
  一本经典教材（第三版），涵盖了系统安全的基本原则，包括隔离、沙箱和安全设计，对于开发强大的代码执行工具至关重要。
