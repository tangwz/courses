---
course: "getting-started-model-context-protocol"
chapter: "architecture-and-fundamentals"
lesson: "hands-on-practice-environment-setup"
sourceId: 7842
sourceUrl: "https://apxml.com/zh/courses/getting-started-model-context-protocol/chapter-1-architecture-and-fundamentals/hands-on-practice-environment-setup"
title: "动手实践：环境配置"
description: "安装MCP SDK并配置开发环境的实用步骤。"
order: 6
plots: []
sourceHash: "a20736cb53775c2d0542abd9a0189f80d1e8e14d3641932c6621e15a012067b2"
sourceCorrections: []
---

构建MCP服务器需要专用工具，以有效处理JSON-RPC通信和传输层。尽管该协议与语言无关，但官方软件开发工具包（SDK）封装了大部分底层序列化和连接管理逻辑。本节将指导你配置一个能够运行、检查和调试MCP服务器的开发环境。

我们将着重介绍如何设置`mcp` Python包、TypeScript SDK以及MCP检查器。检查器是工作流程中不可或缺的组成部分；它充当一个代理客户端，使你可以在Web界面中测试服务器的资源和工具，然后与像Claude Desktop这样的生产客户端集成。

### 开发生命周期

在安装软件包之前，了解本地环境中的各个组件如何相互配合运作，这将很有帮助。与你可能使用`curl`或Postman针对正在运行的HTTP服务器进行标准Web开发不同，MCP服务器通常通过标准输入/输出（stdio）进行通信。这需要一个主机进程来启动服务器并管理输入和输出流。

MCP检查器在开发期间承担了主机的作用。它位于你的浏览器和服务器脚本之间，将UI操作转换为通过stdio发送的JSON-RPC消息。

> 开发数据流显示了检查器如何通过标准流在浏览器界面和本地服务器进程之间进行通信。

### 前提条件

确保你的系统满足以下基本要求：

- **Python:** Python SDK建议使用3.10或更高版本。
- **Node.js:** 18或更高版本。即使你主要使用Python进行开发，MCP检查器也是一个Node.js应用程序，需要`npm`或`npx`来运行。

### 安装MCP检查器

检查器是一个全局工具，用于验证你的服务器是否正确实现了协议。它会可视化你的服务器所提供的资源、提示和工具。

使用npm全局安装检查器：

```bash
npm install -g @modelcontextprotocol/inspector
```

你可以通过运行`mcp-inspector --version`来验证安装，或者简单地检查`npx @modelcontextprotocol/inspector`是否无错误执行。我们将在后续章节中大量使用此工具来调试连接生命周期。

### Python环境配置

对于基于Python的服务器，我们建议在虚拟环境中隔离依赖项。这可以避免与系统软件包发生冲突，并模拟生产部署容器。

1. **创建项目目录**
   组织你的工作区。创建一个目录来存放服务器代码。

   ```bash
   mkdir mcp-server-quickstart
   cd mcp-server-quickstart
   ```
2. **初始化虚拟环境**
   使用内置的`venv`模块创建隔离环境。

   ```bash
   # 在macOS/Linux上
   python3 -m venv .venv
   source .venv/bin/activate

   # 在Windows上
   python -m venv .venv
   .venv\Scripts\activate
   ```
3. **安装MCP SDK**
   安装核心库。此软件包包含服务器原语和stdio传输层。

   ```bash
   pip install mcp
   ```

   如果你打算使用`uv`进行更快的包管理（在现代Python数据栈中很常见），你也可以运行`uv pip install mcp`。

### TypeScript环境配置

如果你偏爱TypeScript，设置过程需要初始化一个Node.js项目。TypeScript SDK是模块化的，将核心协议与特定的传输实现分离。

1. **初始化项目**

   ```bash
   mkdir mcp-ts-server
   cd mcp-ts-server
   npm init -y
   ```
2. **安装依赖项**
   你需要SDK和类型定义。

   ```bash
   npm install @modelcontextprotocol/sdk zod
   npm install --save-dev typescript @types/node tsx
   ```

   我们在这里包含了`zod`，因为它是MCP TypeScript生态系统中用于定义工具输入的标准模式验证库。包含`tsx`是为了在开发过程中直接执行TypeScript文件，而无需单独的构建步骤。
3. **配置TypeScript**
   生成`tsconfig.json`以处理模块解析。

   ```bash
   npx tsc --init
   ```

   确保你的`tsconfig.json`配置为NodeNext模块解析，以支持现代ESM导入，MCP SDK使用这些导入方式。

   ```json
   {
     "compilerOptions": {
       "target": "ES2022",
       "module": "NodeNext",
       "moduleResolution": "NodeNext",
       "outDir": "./dist",
       "rootDir": "./src",
       "strict": true
     }
   }
   ```

### 验证安装

为确认你的环境已正确配置，创建一个最小的“Hello World”脚本，用于初始化一个服务器实例。这尚未提供数据，但它确认了库可导入且解释器工作正常。

**Python验证 (`check_env.py`):**

```python
from mcp.server import Server
from mcp.server.stdio import StdioServerTransport

def main():
    # 使用名称和版本初始化服务器
    server = Server("test-server")
    
    print(f"MCP SDK 版本已验证。服务器对象已创建: {server.name}")

if __name__ == "__main__":
    main()
```

使用`python check_env.py`运行此脚本。如果它打印成功消息而没有引发`ImportError`，则你的Python环境已准备就绪。

### 与Claude Desktop集成（可选）

检查器是开发的主要工具，你最终可能希望针对Claude Desktop应用程序进行测试。Claude Desktop会读取配置文件以定位本地MCP服务器。

根据你的操作系统找到配置文件：

- **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

你暂时不需要编辑此文件。在第4章中，我们将修改此JSON结构以注册你在后续章节中构建的服务器。

### 常见设置问题排查

**Stdio缓冲**
当通过stdio运行Python脚本时，输出缓冲有时会导致JSON-RPC消息挂起。MCP SDK在内部处理了这一点，但如果你正在编写自定义包装器，请确保使用`-u`（无缓冲）标志运行Python或设置环境变量`PYTHONUNBUFFERED=1`。

**端口冲突**
MCP检查器默认使用端口5173。如果此端口被占用，你可以在启动检查器命令时使用`--port`参数 (parameter)指定不同的端口。

随着SDK的安装和检查器的就绪，你的环境已准备好定义资源和提示。在下一章中，我们将编写代码，将本地文件系统提供给LLM，将静态文件转换为可用的上下文 (context)。

## 参考资料

- [\`venv\` - Creation of virtual environments](https://docs.python.org/3/library/venv.html) — Python Software Foundation (2024)
  Python官方文档，解释如何创建和管理独立的Python环境，这是设置过程中的关键步骤。
