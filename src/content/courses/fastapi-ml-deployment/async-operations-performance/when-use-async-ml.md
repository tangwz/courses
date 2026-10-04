---
course: "fastapi-ml-deployment"
chapter: "async-operations-performance"
lesson: "when-use-async-ml"
sourceId: 5321
sourceUrl: "https://apxml.com/zh/courses/fastapi-ml-deployment/chapter-5-async-operations-performance/when-use-async-ml"
title: "机器学习推理何时适用异步"
description: "理解异步为机器学习模型服务（I/O密集型预处理/后处理）带来优势的适用情况。"
order: 2
plots: []
sourceHash: "ea4151bc5985f89a32db672553f52e0fd07f31233f3e738ca00071b3679f7330"
sourceCorrections: []
---

FastAPI的异步能力对于构建响应式Web服务来说是一个重要优点。通过为路由处理器使用`async def`，FastAPI可以高效地并发管理多个传入请求，尤其当这些请求涉及等待外部操作时，例如数据库查询或API调用（I/O密集型任务）。随之而来的问题是：这如何应用于机器学习 (machine learning)推理 (inference)呢？机器学习推理通常是计算密集型（CPU密集型）任务。

简短的回答是，如果推理本身是纯CPU密集型的Python代码，那么为路由处理器使用`async def`并*不会*自动使机器学习模型的预测功能运行更快或与其他请求并行。Python的全局解释器锁（GIL）通常会阻止多个线程在不同的CPU核心上同时执行Python字节码。标准的`async`/`await`是为协作式多任务处理而设计的，主要是在I/O等待期间让出控制权，而不是在大量计算期间。

那么，在机器学习推理接口的背景下，`async def`何时实际有益呢？当你的请求处理不仅仅涉及原始模型预测时，其优势就会显现。思考一下预测请求的典型生命周期：

1. **接收请求：** 数据到达接口。
2. **预处理：** 输入数据可能需要清洗、转换或增加信息。此步骤可能包括：
   - 从数据库获取额外特征 (`await db.fetch_features(...)`)。
   - 调用另一个内部或外部API (`await external_service.get_user_data(...)`)。
   - 从存储读取辅助文件 (`await storage.read_config(...)`)。
3. **模型推理：** 预处理后的数据送入已加载模型 (`model.predict(processed_data)`)。这通常是CPU密集型部分。
4. **后处理：** 模型的输出可能需要格式化、解释，或根据预测结果采取进一步行动。这可能包括：
   - 将预测结果和输入特征保存到日志数据库 (`await db.log_prediction(...)`)。
   - 根据结果发送通知 (`await notifications.send_alert(...)`)。
   - 调用另一个API以触发后续流程 (`await workflow_service.trigger_action(...)`)。
5. **返回响应：** 最终结果发送回客户端。

如果你的接口在预处理（步骤2）或后处理（步骤4）阶段执行任何I/O密集型操作，那么为路由处理器使用`async def`将非常有益。当I/O操作等待时（例如，等待数据库响应），FastAPI事件循环可以切换去处理其他传入请求，从而提升应用程序的整体吞吐量 (throughput)和响应速度。

```python
# 示例展示推理周围I/O的异步用法

from fastapi import FastAPI
from pydantic import BaseModel
import asyncio # 导入asyncio库（用于模拟I/O操作）

# 假设'model'已在其他地方加载
# 假设'db'和'external_service'是异步客户端

app = FastAPI()

class InputData(BaseModel):
    raw_feature: str
    user_id: int

class OutputData(BaseModel):
    prediction: float
    info: str

async def fetch_extra_data_from_db(user_id: int):
    # 模拟异步数据库调用
    await asyncio.sleep(0.05) # 模拟I/O等待
    return {"db_feature": user_id * 10}

async def call_external_service(raw_feature: str):
    # 模拟异步外部API调用
    await asyncio.sleep(0.1) # 模拟I/O等待
    return {"service_info": f"Info for {raw_feature}"}

def run_model_inference(processed_data: dict):
    # 模拟CPU密集型推理
    # 注意：在真实的异步路由中，这种阻塞调用
    # 应该谨慎处理（参见下一节）
    import time
    time.sleep(0.2) # 模拟计算
    return processed_data.get("db_feature", 0) / 100.0

@app.post("/predict", response_model=OutputData)
async def predict_endpoint(data: InputData):
    # --- 异步I/O密集型预处理 ---
    # 并发执行I/O操作
    db_data_task = asyncio.create_task(fetch_extra_data_from_db(data.user_id))
    service_data_task = asyncio.create_task(call_external_service(data.raw_feature))

    db_data = await db_data_task
    service_data = await service_data_task
    # ------------------------------------

    processed_input = {**db_data} # 合并特征

    # --- CPU密集型推理 ---
    # !!! 警告：如果处理不当，可能造成阻塞
    prediction_value = run_model_inference(processed_input)
    # （我们将在下一节讨论如何处理此阻塞调用）
    # ---------------------------

    # --- 潜在的异步后处理 ---
    # 示例：可以在此处等待 db.log_prediction(...)
    # ------------------------------------

    return OutputData(
        prediction=prediction_value,
        info=service_data.get("service_info", "N/A")
    )
```

在上面的示例中，`fetch_extra_data_from_db`和`call_external_service`代表I/O密集型操作。使用`async def`允许接口高效地`await`这些操作。在等待期间，FastAPI可以服务其他请求。

然而，请注意`run_model_inference`函数。如果此函数执行大量CPU工作（如通过`time.sleep`模拟），直接在`async def`路由处理器中调用它仍然会造成问题。因为它同步且是CPU密集型的，在执行时它会阻塞单个事件循环线程，阻止FastAPI在此期间处理*任何*其他请求。这会抵消异步在推理阶段本身的并发性带来的好处。

> 此图说明了异步FastAPI接口处理ML预测请求的流程。`async`/`await`直接有利于I/O密集型步骤，而CPU密集型推理需要特定方法（接下来讨论）来避免阻塞事件循环。

**总结来说：** 当请求处理涉及核心模型预测步骤之前或之后的异步I/O操作时，主要应为你的ML推理接口使用`async def`。如果你的接口*只*对请求中已有的数据执行同步的CPU密集型推理，那么仅凭`async def`不会提升推理本身的性能，并且可能需要额外方法来避免阻塞服务器，我们将在接下来介绍。

## 参考资料

- [FastAPI Documentation](https://fastapi.tiangolo.com/) — Sebastián Ramírez (2024)
  介绍了FastAPI的异步特性及其在构建Web服务中的应用。
- [asyncio - Asynchronous I/O](https://docs.python.org/3/library/asyncio.html) — Python Software Foundation (2024)
  Publisher: Python Software Foundation
  详细介绍了Python内置的异步I/O框架，包括事件循环和协程。
- [Global Interpreter Lock](https://wiki.python.org/moin/GlobalInterpreterLock) — Python Wiki Contributors (2023)
  解释了Python全局解释器锁的机制及其对并发的影响。
- [Fluent Python: Clear, Concise, and Effective Programming](https://www.oreilly.com/library/view/fluent-python-2nd/9781492056314/) — Luciano Ramalho (2022)
  Publisher: O'Reilly Media
  提供了对Python并发模型，包括asyncio和GIL的深入审视。
