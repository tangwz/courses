---
course: "monitoring-managing-ml-models-production"
chapter: "scalable-monitoring-infrastructure"
lesson: "practice-mlflow-grafana-monitoring"
sourceId: 3316
sourceUrl: "https://apxml.com/zh/courses/monitoring-managing-ml-models-production/chapter-5-scalable-monitoring-infrastructure/practice-mlflow-grafana-monitoring"
title: "实践：使用MLflow和Grafana设置监控"
description: "实践使用MLflow进行日志记录和Grafana进行可视化来设置基础监控流程。"
order: 7
plots: ["plots/3316-0.json"]
sourceHash: "dca9b02236d20db8f9a0312471cbaf0a917a70a8cc7b5fdd165806711331a201"
sourceCorrections: []
---

搭建一个基础的监控流程，用MLflow追踪指标，并用Grafana近乎实时地呈现它们。这个练习展示了这些工具如何协同工作，MLflow提供详细的日志记录用于分析和模型溯源，而Grafana则提供操作仪表板功能，通常由Prometheus这样的时序数据库支持。

为此练习，我们假设您有一个已安装`mlflow`和`prometheus_client`（`pip install mlflow prometheus_client Flask`）的Python运行环境，并且已安装Docker用于运行Prometheus和Grafana。

### 场景

我们将模拟一个简单的预测服务。对于每个预测请求，我们希望：

1. 使用MLflow记录预测值，以便进行历史追踪和分析。
2. 将当前平均预测值作为Prometheus可抓取的指标公开。
3. 在Grafana中呈现此平均预测值随时间的变化。

### 步骤 1：模拟服务的指标化

首先，我们来创建一个简单的Python脚本（`prediction_service.py`），它模拟进行预测并使用MLflow和Prometheus客户端库记录指标。我们将使用Flask创建一个最小的Web端点，供Prometheus抓取。

```python
import mlflow
import time
import random
from flask import Flask, Response
from prometheus_client import Gauge, CollectorRegistry, generate_latest

# --- 配置 ---
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000" # 默认的本地MLflow追踪服务器URI
EXPERIMENT_NAME = "Production Monitoring Simulation"
SERVICE_PORT = 8080 # Flask应用暴露Prometheus指标的端口

# --- MLflow 设置 ---
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# 确保实验存在
try:
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        client.create_experiment(EXPERIMENT_NAME)
    experiment_id = client.get_experiment_by_name(EXPERIMENT_NAME).experiment_id
except Exception as e:
    print(f"Could not connect to MLflow tracking server at {MLFLOW_TRACKING_URI} or create experiment. Please ensure it's running. Error: {e}")
    experiment_id = None # 处理MLflow服务器未运行的情况

# --- Prometheus 设置 ---
# 使用注册表管理指标
prometheus_registry = CollectorRegistry()
# 定义一个Gauge指标来追踪平均预测值
avg_prediction_gauge = Gauge(
    'average_prediction_value',
    'Average value of predictions made by the service',
    registry=prometheus_registry
)

# --- 用于Prometheus指标的Flask应用 ---
app = Flask(__name__)

@app.route('/metrics')
def metrics():
    """ 暴露Prometheus指标。 """
    return Response(generate_latest(prometheus_registry), mimetype='text/plain')

# --- 模拟逻辑 ---
def simulate_predictions():
    """ 模拟进行预测并记录指标。 """
    print(f"Starting prediction simulation. Logging to MLflow experiment: '{EXPERIMENT_NAME}'")
    prediction_values = []
    max_history = 50 # 为计算平均值保留一个滚动窗口

    with mlflow.start_run(experiment_id=experiment_id, run_name="Simulated Production Run") as run:
        print(f"MLflow Run ID: {run.info.run_id}")
        mlflow.log_param("simulation_type", "average_prediction_monitoring")

        step = 0
        while True:
            # 模拟一个新的预测（例如，分数、概率、回归输出）
            # 让我们模拟一个随时间漂移的值
            base_value = 50
            drift = step * 0.1 # 逐渐向上漂移
            noise = random.gauss(0, 5) # 添加一些噪声
            prediction = base_value + drift + noise
            prediction = max(0, min(100, prediction)) # 将值限制在0到100之间

            # 将单个预测记录到MLflow
            mlflow.log_metric("prediction_value", prediction, step=step)

            # 更新滚动列表以计算平均值
            prediction_values.append(prediction)
            if len(prediction_values) > max_history:
                prediction_values.pop(0)

            # 计算并更新Prometheus Gauge
            if prediction_values:
                current_avg = sum(prediction_values) / len(prediction_values)
                avg_prediction_gauge.set(current_avg)
                # （可选）也将平均值记录到MLflow
                mlflow.log_metric("average_prediction_value_gauge", current_avg, step=step)

            print(f"Step {step}: Prediction={prediction:.2f}, Current Avg={current_avg:.2f}")

            step += 1
            time.sleep(5) # 模拟预测之间等待5秒

if __name__ == '__main__':
    # 在单独的线程（或进程）中启动Flask应用以处理/metrics端点
    # 为简化此示例，我们可以单独运行它或集成。
    # 这里我们专注于模拟循环。单独运行Flask。
    # 示例：在另一个终端中运行 'flask --app prediction_service run --port 8080'
    # 或集成线程：
    # from threading import Thread
    # metrics_thread = Thread(target=lambda: app.run(host='0.0.0.0', port=SERVICE_PORT, debug=False), daemon=True)
    # metrics_thread.start()

    # 运行模拟
    simulate_predictions()
```

**运行前：**

1. 在一个单独的终端中启动MLflow追踪服务器：

   ```bash
   mlflow ui
   ```

   这通常会在`http://127.0.0.1:5000`启动服务器。
2. 您需要运行Flask应用部分以暴露`/metrics`端点。对于本示例，最简单的方法是运行Flask开发服务器，指向我们脚本中的`app`对象。将脚本保存为`prediction_service.py`。在一个终端中，运行模拟：

   ```bash
   python prediction_service.py
   ```

   在*另一个*终端中，运行Flask服务器以暴露指标：

   ```bash
   flask --app prediction_service run --port 8080
   ```

   （确保已安装Flask：`pip install Flask`）。您现在应该能够访问`http://localhost:8080/metrics`并看到`average_prediction_value`指标。

### 步骤 2：设置 Prometheus

现在，我们需要Prometheus抓取`/metrics`端点。创建一个简单的`prometheus.yml`配置文件：

```yaml
# prometheus.yml
global:
  scrape_interval: 10s # 每10秒抓取目标

scrape_configs:
  - job_name: 'prediction_service'
    static_configs:
      - targets: ['host.docker.internal:8080'] # 适用于Mac/Windows上的Docker
        # 如果在Linux上运行Docker，请使用您的宿主机IP而不是host.docker.internal
        # 示例：- targets: ['172.17.0.1:8080']（如果需要，请查找您的Docker桥接IP）
        # 或者如果直接在宿主机上运行Prometheus：- targets: ['localhost:8080']
```

*关于`host.docker.internal`的说明*：在Docker Desktop（Mac/Windows）上，这个特殊的DNS名称从Docker容器内部解析到宿主机的IP地址。如果您在Linux上使用Docker，您可能需要使用Docker桥接网络上的宿主机IP地址（通常是`172.17.0.1`）或配置不同的网络。如果直接在宿主机上运行Prometheus（而不是在Docker中），请使用`localhost:8080`。

使用Docker运行Prometheus，挂载您的配置文件：

```bash
docker run -d --name prometheus \
  -p 9090:9090 \
  -v $(pwd)/prometheus.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus
```

您现在应该能够访问`http://localhost:9090`上的Prometheus UI。检查“Targets”页面以确认Prometheus已成功抓取您的`prediction_service`端点。您也可以在Prometheus查询浏览器中查询指标`average_prediction_value`。

### 步骤 3：设置 Grafana

使用Docker运行Grafana：

```bash
docker run -d --name grafana \
  -p 3000:3000 \
  grafana/grafana-oss
```

访问`http://localhost:3000`上的Grafana（默认登录名是admin/admin，您将被提示更改密码）。

1. **添加数据源：**
   - 前往配置（齿轮图标）-> 数据源。
   - 点击“添加数据源”。
   - 选择“Prometheus”。
   - 将HTTP URL设置为您的Prometheus实例。如果Grafana与Prometheus（如上所述启动）一起在Docker中运行，您通常可以使用`http://prometheus:9090`。如果Grafana在宿主机上而Prometheus在Docker中，请使用`http://localhost:9090`。
   - 点击“保存并测试”。您应该会看到一条成功消息。

### 步骤 4：创建 Grafana 仪表板

1. **创建仪表板：**
   - 前往仪表板部分（四个方块图标）-> 新建 -> 新建仪表板。
   - 点击“添加可视化”。
2. **配置面板：**
   - 选择您的“Prometheus”数据源。
   - 在“指标浏览器”或查询字段中，输入指标名称：`average_prediction_value`。
   - Grafana应自动获取数据并显示时序图。
   - 自定义面板标题（例如，“平均预测值随时间变化”）。
   - 根据需要调整可视化设置（例如，线宽、颜色、轴标签）。
   - 点击“应用”或“保存”将面板添加到您的仪表板。用一个有意义的名称保存仪表板。

您现在应该能看到一个图表，呈现平均预测值，大约每10秒更新一次（基于Prometheus抓取间隔），反映了我们在模拟中引入的漂移。



![模拟的平均预测值随时间变化](plots/3316-0.json)



> `average_prediction_value`指标在Grafana面板中的呈现，显示由于模拟漂移导致的随时间上升的趋势。

### 讨论

本练习提供了一个基础的设置。我们使用了：

- **MLflow：** 用于记录单个预测指标（`prediction_value`）以及可能的配置参数 (parameter)或聚合指标（`average_prediction_value_gauge`）。这会创建一个与特定运行关联的详细历史记录，对于调试、审计和再训练分析非常有价值。您可以在MLflow UI中查看这些运行。
- **Prometheus：** 用于抓取服务暴露的操作指标（`average_prediction_value`）。Prometheus为高效的时序数据存储和查询而设计，使其适合高频监控。
- **Grafana：** 用于查询Prometheus并在近乎实时的仪表板中呈现操作指标。Grafana还根据这些指标提供警报功能。

在更复杂的生产场景中：

- 预测服务可能会被容器化并部署（例如，在Kubernetes上）。
- 指标收集可能涉及专门的代理或库，与服务框架更深入地集成。
- Prometheus抓取将通过服务发现机制进行配置。
- Grafana仪表板将更加精细，可能会结合来自多个来源（基础设施、应用性能、模型指标）的指标，并包含警报规则（例如，如果平均预测值变化过快或超出特定阈值则发出警报）。
- MLflow的详细日志与Grafana的操作视图之间的关联可能涉及定期将MLflow（或其后端数据库）中的聚合数据导出到时序数据库，或者使用在两个系统中记录的唯一标识符来关联问题。

这个实际示例展示了如何组合不同的工具来构建分层监控策略，既处理即时操作运行状况，也处理长期模型性能分析。

## 参考资料

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html) — MLflow Project (Current)
  提供使用 MLflow 进行机器学习工作流中的实验追踪、模型日志记录和部署的全面指南。
- [Prometheus Documentation](https://prometheus.io/docs/introduction/overview/) — Prometheus Authors (Current)
  Publisher: Cloud Native Computing Foundation
  官方来源，用于理解 Prometheus 监控系统的架构、数据模型、配置和查询。
- [Grafana Documentation](https://grafana.com/docs/grafana/latest/) — Grafana Labs (Current)
  Publisher: Grafana Labs
  提供设置 Grafana、配置数据源（如 Prometheus）以及构建用于数据可视化和告警的仪表盘的详细说明。
- [Machine Learning Engineering](https://leanpub.com/machine-learning-engineering) — Andriy Burkov (2020)
  Publisher: True Positive Inc.; Pages: 310 pages
  一本关于整个机器学习生命周期的实践指南，涵盖了生产环境中模型监控和操作化等主题。
