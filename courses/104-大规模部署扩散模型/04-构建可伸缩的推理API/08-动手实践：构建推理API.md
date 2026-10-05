# 动手实践：构建推理API

来源：[原文](https://apxml.com/zh/courses/deploying-diffusion-models-scale/chapter-4-building-scalable-inference-apis/practice-building-inference-api)

[返回章节目录](README.md) · [返回课程目录](../README.md)

构建一个功能性扩散模型推理 (inference)API，使用FastAPI（一个以速度快、易于使用而知名的流行Python Web框架）。实现一个基本的异步生成端点，演示如何在不阻塞主服务器进程的情况下处理可能耗时的推理任务。

本次实践假定您拥有一个可用的Python环境（建议3.8+），并且能够熟练使用`pip`安装软件包。您还应对扩散模型和Hugging Face的`diffusers`库有基本认识，以及对REST API有基本了解。

### 前提条件

首先，安装所需的库：

```bash
pip install "fastapi[all]" diffusers transformers accelerate torch torchvision torchaudio Pillow
```

- `fastapi[all]`：安装FastAPI及其常用依赖项，包括Uvicorn（一个ASGI服务器）和Pydantic（用于数据验证）。
- `diffusers`、`transformers`、`accelerate`：Hugging Face的库，用于处理扩散模型。
- `torch`、`torchvision`、`torchaudio`：PyTorch核心库。如果您有兼容的NVIDIA GPU并安装了CUDA，请务必按照PyTorch官方网站上的说明安装支持CUDA的PyTorch版本。否则，这将安装CPU版本。
- `Pillow`：用于图像处理。

您还需要一个预训练 (pre-training)的扩散模型。在此示例中，我们将使用Stable Diffusion模型，但您可以根据`diffusers`库中可用的其他扩散模型调整代码。请确保您有足够的磁盘空间和内存（最好还有GPU）来下载和运行模型。

### 1. 设置FastAPI应用

创建一个Python文件，例如`api_server.py`。首先导入必要的模块，并初始化FastAPI应用和扩散模型管道。

```python
import io
import base64
import uuid
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from diffusers import DiffusionPipeline
import torch
from PIL import Image
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 应用设置 ---
app = FastAPI(title="扩散模型推理API")

# --- 模型加载 ---
# 如果使用私有模型，请确保已配置凭据
# （例如，huggingface-cli login）
model_id = "stabilityai/stable-diffusion-2-1-base" # 或其他扩散模型
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = None

try:
    logger.info(f"正在将模型：{model_id}加载到设备：{device}")
    # 加载管道
    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
    pipe = pipe.to(device)
    logger.info("模型加载成功。")
    # 可选：如果模型需要/可用，添加安全检查器
    # from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
    # safety_checker = StableDiffusionSafetyChecker.from_pretrained(...)
    # pipe.safety_checker = safety_checker

except Exception as e:
    logger.error(f"加载扩散模型失败：{e}")
    # 决定如何处理失败：退出、无模型运行等。
    # 在此示例中，我们允许应用启动，但生成会失败。
    pipe = None

# --- 任务结果的内存存储（在生产环境中请替换为持久化存储） ---
results_store = {}
```

在此设置中：

- 我们导入了必要的库。
- 我们初始化了`FastAPI`应用。
- 我们定义了`model_id`并确定了可用的`device`（GPU或CPU）。
- 我们尝试加载`DiffusionPipeline`。在应用启动时加载模型对性能来说很重要。为每个请求重新加载模型会引入显著延迟。我们在CUDA上使用`torch.float16`以加快推理 (inference)速度并减少内存占用。包含错误处理，以防模型加载失败。
- 一个简单的字典`results_store`用于临时保存生成结果。在生产系统中，您会将其替换为更直接的方案，例如Redis、数据库或云存储。

### 2. 定义请求和响应模型

FastAPI使用Pydantic模型进行请求和响应数据验证和序列化。让我们为图像生成端点定义模型。

```python
class GenerationRequest(BaseModel):
    prompt: str
    negative_prompt: str | None = None
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    seed: int | None = None # 用于重现性

class TaskResponse(BaseModel):
    task_id: str
    status: str
    message: str | None = None

class ResultResponse(BaseModel):
    task_id: str
    status: str
    prompt: str
    image_base64: str | None = None # Base64编码的图片
    error_message: str | None = None
```

- `GenerationRequest`：定义生成请求预期的输入JSON体。它包含`prompt`以及`negative_prompt`、`num_inference_steps`、`guidance_scale`和`seed`等可选参数 (parameter)。为方便起见，提供了默认值。
- `TaskResponse`：用于在接受任务后立即回复客户端，提供`task_id`以便后续状态检查。
- `ResultResponse`：定义了用于获取生成任务结果的结构，包括状态、原始提示、生成的图像（Base64编码）或错误消息。

### 3. 实现异步生成逻辑

扩散模型的推理 (inference)可能需要几秒甚至几分钟，尤其是在CPU上或对于高分辨率图像。同步API端点会阻塞，可能导致超时或阻止服务器处理其他请求。我们将使用FastAPI的`BackgroundTasks`在后台运行推理过程。

首先，定义执行实际推理的函数：

```python
def run_diffusion_inference(task_id: str, req: GenerationRequest):
    """在后台运行扩散模型推理。"""
    global results_store
    logger.info(f"Starting inference for task_id: {task_id}")
    results_store[task_id] = {"status": "processing", "prompt": req.prompt}

    if pipe is None:
        logger.error(f"Model not loaded. Cannot process task_id: {task_id}")
        results_store[task_id].update({"status": "failed", "error_message": "Model not available."})
        return

    try:
        # 如果提供了种子，则设置以确保可重现性
        generator = None
        if req.seed is not None:
            generator = torch.Generator(device=device).manual_seed(req.seed)

        # 执行推理
        with torch.inference_mode(): # 比torch.no_grad()更节省内存
             image: Image.Image = pipe(
                prompt=req.prompt,
                negative_prompt=req.negative_prompt,
                num_inference_steps=req.num_inference_steps,
                guidance_scale=req.guidance_scale,
                generator=generator
            ).images[0] # 从结果列表中获取第一张图片

        # 将图片转换为Base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # 存储结果
        results_store[task_id].update({"status": "completed", "image_base64": img_str})
        logger.info(f"Inference completed successfully for task_id: {task_id}")

    except Exception as e:
        logger.exception(f"Inference failed for task_id: {task_id}") # 记录完整的堆栈跟踪
        results_store[task_id].update({"status": "failed", "error_message": str(e)})
```

此函数：

- 接收`task_id`和`GenerationRequest`数据。
- 更新`results_store`，将任务标记 (token)为“处理中”。
- 检查模型`pipe`是否成功加载。
- 如果提供了种子，则设置一个PyTorch生成器。
- 在`torch.inference_mode()`中运行扩散管道推理以提高效率。
- 获取生成的第一张图片（扩散管道通常返回一个列表）。
- 将PIL Image对象转换为Base64编码的PNG字符串。这是在JSON响应中嵌入 (embedding)图像的常用方法。或者，您可以将图片保存到云存储（如S3或GCS）并返回一个URL。
- 用最终状态（“完成”或“失败”）和结果（图像或错误消息）更新`results_store`。
- 包含错误处理，以捕获推理期间的异常。

### 4. 创建API端点

现在，定义FastAPI端点：一个用于提交生成任务，另一个用于检查其状态并获取结果。

```python
@app.post("/generate", response_model=TaskResponse, status_code=202)
async def submit_generation_task(req: GenerationRequest, background_tasks: BackgroundTasks):
    """
    接受生成请求，将其添加到后台队列，
    并返回任务ID。
    """
    if pipe is None:
        raise HTTPException(status_code=503, detail="模型不可用或加载失败。")

    task_id = str(uuid.uuid4())
    results_store[task_id] = {"status": "pending", "prompt": req.prompt} # 初始状态

    logger.info(f"Received generation request. Assigning task_id: {task_id}")
    background_tasks.add_task(run_diffusion_inference, task_id, req)

    return TaskResponse(task_id=task_id, status="pending", message="任务已接收并排队等待处理。")

@app.get("/results/{task_id}", response_model=ResultResponse)
async def get_generation_result(task_id: str):
    """
    获取给定任务ID的状态和结果（如果可用）。
    """
    result = results_store.get(task_id)

    if not result:
        raise HTTPException(status_code=404, detail="未找到任务ID。")

    return ResultResponse(
        task_id=task_id,
        status=result.get("status", "unknown"),
        prompt=result.get("prompt", ""),
        image_base64=result.get("image_base64"),
        error_message=result.get("error_message")
    )

@app.get("/health")
async def health_check():
    """基本健康检查端点。"""
    # 这里可以添加更多检查（例如，模型响应性）
    model_status = "available" if pipe is not None else "unavailable"
    return {"status": "ok", "model_status": model_status}
```

- **`/generate` (POST)：**
  - 在请求体中接受一个`GenerationRequest`。
  - 使用FastAPI提供的`BackgroundTasks`。
  - 使用`uuid`生成一个唯一的`task_id`。
  - 在`results_store`中设置初始的“pending”状态。
  - 使用`background_tasks.add_task()`将`run_diffusion_inference`函数添加到后台任务队列，并传递`task_id`和请求数据。
  - 立即返回`202 Accepted`状态码以及`TaskResponse`，表示任务已被接受但尚未完成。
  - 包含一个检查，如果模型未加载，则返回`503 Service Unavailable`。
- **`/results/{task_id}` (GET)：**
  - 将`task_id`作为路径参数 (parameter)。
  - 在`results_store`中查找任务ID。
  - 如果ID不存在，则返回`404 Not Found`。
  - 否则，返回当前状态和结果（或错误），封装在`ResultResponse`中。客户端需要轮询此端点，直到状态为“completed”或“failed”。
- **`/health` (GET)：**
  - 一个简单的端点，用于检查API服务器是否正在运行以及模型是否已加载。

> 使用FastAPI BackgroundTasks的异步推理 (inference)请求流程。

### 5. 运行和测试API

将代码保存为`api_server.py`。从终端使用Uvicorn运行API：

```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
```

- `api_server:app`：告诉Uvicorn在`api_server.py`文件中查找FastAPI应用实例（`app`）。
- `--host 0.0.0.0`：使服务器可以从网络上的其他机器访问。
- `--port 8000`：指定运行端口。
- `--reload`：在检测到代码更改时自动重启服务器（在开发期间很有用）。

**使用`curl`测试：**

1. **提交生成任务：**

   ```bash
   curl -X POST "http://localhost:8000/generate" \
        -H "Content-Type: application/json" \
        -d '{
              "prompt": "A photo of an astronaut riding a horse on the moon",
              "num_inference_steps": 25,
              "guidance_scale": 9.0
            }'
   ```

   这应该返回一个JSON响应，例如：

   ```json
   {
     "task_id": "some-unique-uuid-string",
     "status": "pending",
     "message": "任务已接收并排队等待处理。"
   }
   ```
2. **检查结果（将`some-unique-uuid-string`替换为实际收到的ID）：**

   ```bash
   curl -X GET "http://localhost:8000/results/some-unique-uuid-string"
   ```

   最初，这可能会返回：

   ```json
   {
     "task_id": "some-unique-uuid-string",
     "status": "处理中",
     "prompt": "一名宇航员在月球上骑马的照片",
     "image_base64": null,
     "error_message": null
   }
   ```

   稍等片刻（取决于您的硬件和推理 (inference)步数）后，再次轮询应该返回已完成的结果：

   ```json
   {
     "task_id": "some-unique-uuid-string",
     "status": "完成",
     "prompt": "一名宇航员在月球上骑马的照片",
     "image_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
     "error_message": null
   }
   ```

   （`image_base64`字符串会很长）。您可以复制此字符串，并使用在线Base64转图片转换器或简单脚本来查看生成的图片。

### 扩展实践示例

本次实践提供了一个基本结构。在此基础上，您可以整合本章前面讨论过的一些思想：

- **请求队列：** 将FastAPI的`BackgroundTasks`替换为专用的消息队列系统（如使用RabbitMQ/Redis的Celery或AWS SQS），以实现更好的可扩展性、持久性以及API服务器和推理 (inference)工作器之间的职责分离。
- **请求批处理：** 实现逻辑（在API服务器排队前或在工作器中）将传入请求分组为批次，然后再将其馈送给模型，从而大幅提升GPU利用率。这里可以使用库或自定义逻辑。
- **速率限制：** 为`/generate`端点添加速率限制中间件（例如，使用`slowapi`），以保护您的服务免受滥用或过载。
- **身份验证：** 使用FastAPI的安全工具（例如API密钥、OAuth2）保护您的端点。
- **持久化存储：** 将生成的图片存储到云存储（S3、GCS、Azure Blob Storage），而不是返回Base64字符串，并在`ResultResponse`中返回一个URL。更新`results_store`以使用持久化数据库或缓存，如Redis。
- **容器化：** 将此API服务器打包成Docker容器进行部署，如第3章所述。

此动手练习呈现了为扩散模型构建异步API的核心组件。通过运用这些原则并整合进一步的扩展技术，您可以创建高效的推理服务，以应对生产工作负载。

## 参考资料

- [FastAPI Documentation](https://fastapi.tiangolo.com/) — Sebastián Ramírez (2024)
  用于构建推理API的Python Web框架官方文档，涵盖请求/响应模型、异步端点和后台任务管理。
- [Hugging Face Diffusers Library Documentation](https://huggingface.co/docs/diffusers/index) — Hugging Face (2024)
  Publisher: Hugging Face
  使用`diffusers`库加载、运行和定制扩散模型的指南，是实践推理逻辑的核心。
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html) — PyTorch Contributors (2024)
  Publisher: PyTorch Foundation
  PyTorch的官方参考资料，与理解实践中模型加载、设备处理以及`torch.inference_mode()`等性能优化有关。
- [Designing Data-Intensive Applications: The Big Ideas Behind Reliable, Scalable, and Maintainable Systems](https://www.oreilly.com/library/view/designing-data-intensive-applications/9781449373320/) — Martin Kleppmann (2017)
  Publisher: O'Reilly Media
  介绍分布式系统、数据存储和构建可靠、可扩展应用程序的见解，与高容量推理服务的架构决策相关。

---

[上一节](07-API%20%E7%89%88%E6%9C%AC%E7%AE%A1%E7%90%86%E7%AD%96%E7%95%A5.md) · [下一节](../05-%E7%9B%91%E6%8E%A7%E5%92%8C%E7%BB%B4%E6%8A%A4%E5%B7%B2%E9%83%A8%E7%BD%B2%E6%A8%A1%E5%9E%8B/01-%E6%89%A9%E6%95%A3%E6%A8%A1%E5%9E%8B%E9%83%A8%E7%BD%B2%E7%9A%84%E6%A0%B8%E5%BF%83%E6%8C%87%E6%A0%87.md)
