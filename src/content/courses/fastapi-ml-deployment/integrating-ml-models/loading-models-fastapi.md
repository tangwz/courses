---
course: "fastapi-ml-deployment"
chapter: "integrating-ml-models"
lesson: "loading-models-fastapi"
sourceId: 5274
sourceUrl: "https://apxml.com/zh/courses/fastapi-ml-deployment/chapter-3-integrating-ml-models/loading-models-fastapi"
title: "将模型载入FastAPI应用"
description: "关于在应用启动时或按需高效载入机器学习模型的策略。"
order: 2
plots: []
sourceHash: "075a3f4e06fb5ca8a27d7f0a83b9181fa0ed2f02d0c7cfee6477c56d48d5797a"
sourceCorrections: []
---

为了进行预测，序列化的机器学习 (machine learning)模型必须被载入运行中的FastAPI应用。载入模型的方式会对应用的启动时间、内存使用以及首次预测请求的延迟产生显著影响。FastAPI中载入模型的常见策略在此进行考察。

### 在应用启动时载入模型

最直接的方法通常是在应用首次启动时载入模型。这意味着模型在第一个请求到达之前就已经准备好并存在于内存中，从而保证预测延迟保持一致。主要的代价是应用启动时间可能会变慢，因为模型载入会成为初始化过程的一部分。

#### 使用全局变量（简单方法）

一种简单的方法是将模型载入到主应用文件中的一个全局变量里。

```python
# main.py
import joblib
from fastapi import FastAPI

app = FastAPI()

# 在模块被导入时（启动时）载入模型
try:
    model = joblib.load("models/sentiment_model.pkl")
    # 您可能还需要载入相关对象，例如向量化器
    # vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    print("模型在启动时成功载入。")
except FileNotFoundError:
    print("错误：模型文件未找到。请确保'models/sentiment_model.pkl'文件存在。")
    model = None # 优雅地处理模型缺失的情况
except Exception as e:
    print(f"载入模型时出错：{e}")
    model = None

@app.post("/predict")
async def predict_sentiment(text: str):
    if model is None:
        # 如果模型载入失败，则返回错误
        raise HTTPException(status_code=503, detail="Model is not available")

    # 假设这里是预处理和预测逻辑
    # features = vectorizer.transform([text])
    # prediction = model.predict(features)
    # return {"text": text, "sentiment_prediction": prediction[0]}
    # 演示用的占位符
    return {"text": text, "sentiment_prediction": "positive"} # 替换为实际逻辑

# 注意：为简化起见，此处省略了使用Pydantic进行输入/输出验证，
# 但您应该按照第2章的说明使用它。
```

虽然简单，但直接使用全局变量有时会使测试和管理应用状态变得更复杂，尤其是在大型应用中。

#### 使用FastAPI启动事件

FastAPI提供了`lifespan`事件（或旧版`startup`/`shutdown`事件），允许您在应用开始接收请求之前和完成运行之后执行代码。这是载入ML模型等资源的一个更整洁的位置。

```python
# main_lifespan.py
import joblib
from fastapi import FastAPI
from contextlib import asynccontextmanager

# 用于存储应用状态的字典，包括载入的模型
app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 在应用启动前运行的代码
    print("应用启动：正在载入模型...")
    try:
        app_state["model"] = joblib.load("models/sentiment_model.pkl")
        # 如果需要，载入其他组件
        # app_state["vectorizer"] = joblib.load("models/tfidf_vectorizer.pkl")
        print("模型载入成功。")
    except FileNotFoundError:
        print("错误：模型文件未找到。")
        app_state["model"] = None
    except Exception as e:
        print(f"启动时载入模型出错：{e}")
        app_state["model"] = None
    yield
    # 应用关闭时运行的代码
    print("应用关闭：正在清理资源...")
    app_state.clear()

app = FastAPI(lifespan=lifespan)

@app.post("/predict")
async def predict_sentiment(text: str):
    model = app_state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not available")

    # 预测逻辑的占位符
    # features = app_state["vectorizer"].transform([text])
    # prediction = model.predict(features)
    # return {"text": text, "sentiment_prediction": prediction[0]}
    return {"text": text, "sentiment_prediction": "positive"} # 替换为实际逻辑
```

使用`lifespan`上下文 (context)管理器是管理需要在启动时初始化并在关闭时清理的资源的推荐方法。它将模型载入逻辑与主应用定义和请求处理分离。

### 按需载入模型（惰性载入）

或者，您可以在第一个预测请求到达时才载入模型。这种方法通常称为“惰性载入”，它能让应用启动更快，因为模型不会立即载入。但是，触发模型载入的第一个请求将遇到更高的延迟。

为了避免在每个后续请求中重新载入模型，您通常会将惰性载入与某种形式的缓存结合起来。Python的`functools.lru_cache`是为载入资源的函数实现这一目标的一种便捷方式。

```python
# main_lazy.py
import joblib
from fastapi import FastAPI, HTTPException
from functools import lru_cache

app = FastAPI()

@lru_cache(maxsize=1) # 缓存此函数的结果
def get_model():
    print("正在尝试载入模型（惰性载入）...")
    try:
        model = joblib.load("models/sentiment_model.pkl")
        print("模型载入成功。")
        return model
    except FileNotFoundError:
        print("错误：惰性载入期间未找到模型文件。")
        return None # 表示失败
    except Exception as e:
        print(f"惰性载入模型出错：{e}")
        return None

@app.post("/predict")
async def predict_sentiment(text: str):
    model = get_model() # 函数调用只触发一次载入
    if model is None:
        raise HTTPException(status_code=503, detail="Model could not be loaded")

    # 预测逻辑的占位符
    # features = ... # 假设向量化器也已载入，可能通过另一个缓存函数
    # prediction = model.predict(features)
    # return {"text": text, "sentiment_prediction": prediction[0]}
    return {"text": text, "sentiment_prediction": "positive"} # 替换为实际逻辑
```

`lru_cache(maxsize=1)`确保`get_model`只执行一次。后续调用将返回缓存的结果（已载入的模型对象，如果载入失败则为`None`），而不会重新执行载入逻辑。

惰性载入在以下情况有益：

- 模型非常大且载入时间很长，因此快速的应用启动很重要。
- 预测端点可能不经常被调用，因此载入成本仅在必要时才产生。
- 内存受限，并且您希望避免载入模型，除非它确实被使用。

### 载入策略比较

> 模型载入策略比较：在启动时载入会带来初始延迟，但能提供一致的请求处理时间。惰性载入能更快地启动应用，但会增加第一个需要模型请求的延迟。

### 处理载入失败

无论采用哪种策略，在模型载入过程中妥善处理潜在错误都非常重要。常见的问题包括模型文件丢失、损坏或与当前库版本不兼容。请在您的载入代码周围使用`try...except`块。

如果模型在启动时载入失败，您可能需要完全阻止应用启动，或者记录错误并让预测端点返回适当的错误状态码（例如`503 Service Unavailable`）。如果使用惰性载入，触发载入的端点应处理失败情况，例如通过记录错误并返回`503`状态。

### 内存考量

机器学习 (machine learning)模型，特别是深度学习 (deep learning)模型，会占用大量内存。在选择载入策略和部署应用时请记住这一点。在启动时载入大型模型会增加应用的基本内存占用。确保您的部署环境有足够的RAM来容纳您打算服务的模型。

在接下来的部分中，我们将在这些载入技术的基础上构建实际的预测端点，并研究FastAPI的依赖注入系统如何更好地组织模型以供您的请求处理器使用。

## 参考资料

- [FastAPI Documentation](https://fastapi.tiangolo.com/) — Sebastián Ramírez (2024)
  FastAPI官方文档，包含应用生命周期事件（lifespan）和异常处理等功能，与模型部署中的资源管理和错误处理相关。
- [functools - Higher-order functions and operations on callable objects](https://docs.python.org/3/library/functools.html#functools.lru_cache) — Python Software Foundation (2024)
  标准库文档，解释functools.lru_cache，这是一个用于记忆化和缓存函数调用结果的装饰器，对于模型的高效按需加载至关重要。
- [Designing Machine Learning Systems: An Iterative Approach to Development, Deployment, and Maintenance](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) — Chip Huyen (2022)
  Publisher: O'Reilly Media
  一本关于设计和部署机器学习系统的综合书籍，提供架构选择、服务策略、资源管理和可靠性方面的见解，与模型加载相关。
- [Joblib: run Python functions as pipeline jobs](https://joblib.readthedocs.io/en/latest/) — The Joblib Developers (2024)
  joblib官方文档，这是一个常用于持久化和加载大型Python对象的库，包括机器学习模型，以实现高效数据管理。
