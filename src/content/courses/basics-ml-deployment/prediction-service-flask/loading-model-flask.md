---
course: "basics-ml-deployment"
chapter: "prediction-service-flask"
lesson: "loading-model-flask"
sourceId: 3986
sourceUrl: "https://apxml.com/zh/courses/basics-ml-deployment/chapter-3-prediction-service-flask/loading-model-flask"
title: "在 Flask 中加载已保存的模型"
description: "了解如何在 Flask 应用上下文中加载序列化的模型文件。"
order: 5
plots: []
sourceHash: "ed43e4e4c5d0374620e3ac129a346f6b3feea02c1de325049f7807fcb51d19df"
sourceCorrections: []
---

部署机器学习 (machine learning)模型时，训练好的模型通常会使用 `pickle` 或 `joblib` 等技术保存到文件中。为了通过 Flask Web 应用提供预测服务，需要将这些已保存的模型重新加载到内存中。将模型载入内存的过程，通常称为反序列化或“加载”模型，使模型能够被应用使用。

### 将模型引入 Flask 应用

核心思想很简单：你的 Flask 应用脚本需要执行相应的函数（`pickle.load` 或 `joblib.load`）来读取模型文件，并在内存中重建模型对象。

模型加载应该在何处进行？对于许多简单的预测服务来说，在 Flask 应用启动时加载模型一次是最有效的方法。这样可以避免每次收到预测请求时都重新读取文件和重建模型对象的开销。我们可以通过将模型加载到 Flask 脚本中的一个全局变量来实现这一点。

假设你使用 `joblib` 将模型保存到名为 `model.joblib` 的文件中（如第二章所述）。以下是如何在你的 Flask 应用脚本（`app.py`）中加载它：

```python
# app.py
from flask import Flask
import joblib
import os # 导入 os 模块

# 创建 Flask 应用实例
app = Flask(__name__)

# --- 模型加载 ---
# 定义模型文件路径
MODEL_FILE_PATH = 'model.joblib' # 确保此路径正确

# 在尝试加载前检查模型文件是否存在
if os.path.exists(MODEL_FILE_PATH):
    # 从文件中加载训练好的模型
    try:
        model = joblib.load(MODEL_FILE_PATH)
        print(f"* 模型已成功从 {MODEL_FILE_PATH} 加载")
    except Exception as e:
        print(f"* 加载模型出错: {e}")
        # 适当处理错误 - 也许退出或使用默认行为
        model = None # 如果加载失败，将模型设为 None
else:
    print(f"* 未在 {MODEL_FILE_PATH} 找到模型文件")
    model = None # 如果文件不存在，将模型设为 None

# --- 定义路由（端点）---
# 我们将在下一节定义我们的预测端点
@app.route('/')
def home():
    # 简单的路由，用于检查应用是否正在运行
    return "Flask 应用正在运行！"

# 在运行应用前添加对模型的检查
if model is None:
    print("! 警告：模型无法加载。预测端点可能无法正常工作。")

# --- 运行应用 ---
if __name__ == '__main__':
    # 运行 Flask 开发服务器
    # host='0.0.0.0' 使其可在网络上的其他设备访问
    # debug=True 启用自动重载和详细错误页面（仅用于开发）
    app.run(host='0.0.0.0', port=5000, debug=True)
```

### 代码说明

1. **导入库：** 我们导入 `Flask` 和 `joblib`。我们还导入 `os` 模块，以帮助在尝试加载模型文件之前检查它是否存在。
2. **定义模型路径：** 我们指定已保存模型所在的 `MODEL_FILE_PATH` 文件名。请确保此路径相对于运行 Flask 脚本的位置是正确的。
3. **检查文件存在性：** 使用 `os.path.exists(MODEL_FILE_PATH)`，我们验证文件是否存在。这可以防止文件丢失时出现错误。
4. **加载模型：** 在 `try...except` 代码块中，`model = joblib.load(MODEL_FILE_PATH)` 读取文件并重建模型对象，将其赋值给 `model` 变量。将此代码放置在定义任何路由 *之前*，使得 `model` 成为应用上下文 (context)中的一个全局变量。
5. **错误处理：** `try...except` 代码块提供了基本的错误处理。如果 `joblib.load` 失败（例如，文件损坏、格式不正确），将打印错误消息，并将 `model` 设置为 `None`。我们还处理了文件未找到的情况。在生产系统中，适当的错误处理非常重要。
6. **全局可访问性：** 由于 `model` 是在任何特定函数（例如路由处理程序）之外加载的，因此它将对脚本中稍后定义的所有路由函数可用。
7. **启动：** 此加载过程只在你启动 Flask 应用 (`python app.py`) 时发生一次。

### 注意事项

- **启动时间：** 加载大型模型可能会增加应用的启动时间。
- **内存占用：** 加载的模型驻留在应用的内存中。请确保你的服务器有足够的 RAM。
- **依赖项：** 请记住，运行 Flask 应用的环境必须安装所有必需的库，包括 `flask`、`joblib`（或 `pickle`），以及模型本身所需的库（例如 `scikit-learn`、`numpy`）。训练环境和部署环境之间库版本不匹配是常见的错误来源。这强调了管理依赖项的重要性，正如第二章中所述。

随着模型加载到我们的 Flask 应用内存中，我们现在可以创建特定的端点，该端点将接收输入数据，使用模型进行预测，并返回结果。

## 参考资料

- [Flask Documentation](https://flask.palletsprojects.com/en/latest/) — Pallets (2024)
  Flask Web框架的官方文档，提供了应用程序设置、请求处理和构建Web服务的最佳实践指南。
- [joblib: running Python functions as pipeline jobs](https://joblib.readthedocs.io/en/latest/) — joblib developers (2024)
  joblib的官方文档，详细介绍了其高效序列化和反序列化大型Python对象（包括机器学习模型）的功能。
- [Designing Machine Learning Systems: New Responsibilities for Data Scientists, ML Engineers, and Software Engineers](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) — Chip Huyen (2022)
  Publisher: O'Reilly Media
  本书提供了设计、构建和操作机器学习系统的全面指导，包括模型服务、部署和依赖管理的实用策略。
- [pickle - Python object serialization](https://docs.python.org/3/library/pickle.html) — Python Software Foundation (2024)
  Python pickle模块的官方文档，该模块实现了用于序列化和反序列化Python对象结构的二进制协议。
