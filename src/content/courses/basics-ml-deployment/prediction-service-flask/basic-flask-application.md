---
course: "basics-ml-deployment"
chapter: "prediction-service-flask"
lesson: "basic-flask-application"
sourceId: 3983
sourceUrl: "https://apxml.com/zh/courses/basics-ml-deployment/chapter-3-prediction-service-flask/basic-flask-application"
title: "构建一个基础的 Flask 应用"
description: "使用 Flask 编写您的第一个简单“Hello, World!”应用。"
order: 4
plots: []
sourceHash: "dac92cc4a08618ae9335ccf938ead8be49392bcb93bb613f79cd14bb2650a75b"
sourceCorrections: []
---

开发一个基本的 Flask Web 应用程序是创建 Flask Web 服务的第一步。这个初步步骤将让您在集成机器学习 (machine learning)模型之前，对 Flask 的运行方式有个大致的了解。它可以作为 Web 服务的“Hello, World!”示例。

一个 Flask 应用的核心部分只需要几行 Python 代码。我们将从创建一个名为 `app.py` 的 Python 文件开始。在这个文件中，我们将执行三个主要操作：

1. **导入 Flask：** 我们需要将 Flask 框架引入到脚本中。
2. **创建应用实例：** 我们创建一个代表 Web 应用的对象。
3. **定义路由和视图函数：** 我们告诉 Flask 当用户访问我们网站上特定 URL（一个“路由”）时要运行的代码。处理请求并生成响应的函数称为“视图函数”。

这是我们基础应用的代码。请将其放入您的 `app.py` 文件中：

```python
# 1. 导入 Flask
from flask import Flask

# 2. 创建应用实例
#    Flask 使用 __name__ 来确定应用的根路径，
#    以便它能找到相对于脚本位置的资源文件。
app = Flask(__name__)

# 3. 定义路由及其对应的视图函数
#    @app.route('/') 装饰器告诉 Flask，hello_world 函数
#    应该处理根 URL (例如，http://yourdomain.com/) 的请求。
@app.route('/')
def hello_world():
    # 当有人访问 '/' 路由时，此函数会被执行。
    # 它只返回字符串 "Hello, World!"。
    return 'Hello, World!'

# 可选：添加此代码块，以便直接使用 'python app.py' 运行服务器
# 这对于开发很有用。
if __name__ == '__main__':
    # debug=True 启用调试模式，提供有用的错误信息
    # 并在代码更改时自动重新加载服务器。
    # 在生产环境中请勿使用 debug=True！
    app.run(debug=True)
```

我们来分析一下 `if __name__ == '__main__':` 代码块。这是标准的 Python 做法。当您直接运行脚本（例如 `python app.py`）时，Python 会将特殊变量 `__name__` 设置为字符串 `'__main__'`。此时，这个 `if` 条件为真，其内部的代码 `app.run(debug=True)` 便会执行。`app.run()` 方法会启动 Flask 内置的开发 Web 服务器。在开发过程中，将 `debug` 设置为 `True` 很方便，因为它会在您保存文件更改时自动重启服务器，并在出现问题时提供更详细的错误页面。请记住，**绝不能**在生产环境中启用 `debug=True` 来运行应用，因为它会带来安全风险。

### 运行您的第一个 Flask 应用

`app.py` 保存后，打开您的终端或命令提示符，导航到保存文件的目录，然后运行它：

```bash
python app.py
```

或者，如果您没有包含 `if __name__ == '__main__':` 代码块，您可以使用 `flask` 命令（您可能需要先告诉 Flask 您的应用在哪里，这取决于您的设置）：

```bash
# 告诉 Flask 您的应用在哪里（如果您的文件不是 app.py 或 wsgi.py）
# 在 Linux/macOS 上：
export FLASK_APP=app.py
# 在 Windows (cmd) 上：
set FLASK_APP=app.py
# 在 Windows (PowerShell) 上：
$env:FLASK_APP = "app.py"

# 可选地启用开发模式（包含调试模式）
# Linux/macOS：
export FLASK_ENV=development
# Windows (cmd)：
set FLASK_ENV=development
# Windows (PowerShell)：
$env:FLASK_ENV = "development"

# 运行开发服务器
flask run
```

您应该在终端中看到类似以下的输出：

```
 * Serving Flask app 'app'
 * Environment: development
 * Debug mode: on
 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
 * Restarting with stat
 * Debugger is active!
 * Debugger PIN: XXX-XXX-XXX
```

此输出表明 Flask 的开发服务器正在运行。它正在监听您本地机器 (`127.0.0.1`，也称为 `localhost`) 端口 `5000` 上的连接。

现在，打开您的 Web 浏览器并访问显示的地址：`http://127.0.0.1:5000/`。

您应该会在页面上看到“Hello, World!”的文本。恭喜！您刚刚使用 Flask 创建并运行了您的第一个 Web 应用。

### 理解流程

我们来可视化一下当您访问 URL 时发生的情况：

> 我们的基础 Flask 应用中的一个简单请求-响应循环。

您的浏览器向运行 Flask 应用的服务器发送了一个 HTTP GET 请求，请求根路径 (`/`) 上的资源。Flask 收到此请求后，查看其路由规则 (`@app.route('/')`)，找到匹配项，并调用了相关的视图函数 (`hello_world()`)。此函数返回了字符串 `'Hello, World!'`。Flask 随后将此字符串打包成 HTTP 响应，并将其发送回您的浏览器，浏览器在页面上渲染了该文本。

这个简单的例子构成了我们预测服务的核心。我们将在此结构上进行扩展，方法是定义一个不同的路由（例如 `/predict`），在应用启动时加载我们保存的机器学习 (machine learning)模型，并修改视图函数以接受输入数据、使用模型进行预测，并将该预测结果在响应中返回。

## 参考资料

- [Flask Quickstart](https://flask.palletsprojects.com/en/latest/quickstart/) — Pallets (2023)
  Publisher: Pallets
  提供设置和运行简单 Flask 应用程序的基础步骤，包括“Hello, World!”示例以及路由和视图函数的说明。
- [Flask Web Development: Developing Web Applications with Python](https://www.oreilly.com/library/view/flask-web-development/9781491991732/) — Miguel Grinberg (2018)
  Publisher: O'Reilly Media; Pages: 312
  这是一本详尽介绍 Flask 应用程序开发的书籍，涵盖从初始设置和路由到更高级功能的方方面面。
- [2. Modules - Python 3.12.0 documentation: Executing modules as scripts](https://docs.python.org/3/tutorial/modules.html#executing-modules-as-scripts) — Python Software Foundation (2023)
  Publisher: Python Software Foundation
  阐明了 'if __name__ == '__main__': ' 结构的作用和用法，这对直接运行 Python 脚本至关重要。
