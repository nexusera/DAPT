# KV-BERT Serving：远端 H200 部署与测试说明

本文说明如何在配备 NVIDIA H200（或同类 GPU）的 Linux 服务器上部署 `serving` 服务，并用本地 JSON 文件调用 `POST /api/v1/extract` 做验证。

---

## 1. 前置条件

| 项目 | 说明 |
|------|------|
| 操作系统 | Linux x86_64，已安装 NVIDIA 驱动 |
| GPU | H200 / A100 等，需能 `nvidia-smi` 正常显示 |
| CUDA | 与 PyTorch 轮子匹配（推荐使用与训练环境一致的 PyTorch 版本） |
| 代码 | 本仓库 `DAPT/` 目录完整（含 `noise_fusion.py`、`noise_feature_processor.py`、`dapt_eval_package/`） |
| 模型 | 微调后的 KV-NER checkpoint 目录（含 `pytorch_model.bin` 或 `model.safetensors`、`config.json`、`label2id.json`、`tokenizer` 等） |
| 噪声分桶 | `noise_bins.json`（与训练/评测时 `compare_models.py --noise_bins` 一致） |

---

## 2. 目录约定与代码同步

本地 `DAPT/` 子目录通过 **GitHub** 与远端 H200 保持同步，远端固定路径为 `/data/ocean/DAPT`。

**每次部署前，先在远端拉取最新代码：**

```bash
cd /data/ocean/DAPT
git pull
```

目录结构：

```text
/data/ocean/DAPT/          # 本仓库根目录（下文记为 $DAPT_ROOT）
├── noise_fusion.py
├── noise_feature_processor.py
├── dapt_eval_package/
├── serving/
└── ...
```

**重要**：启动 `uvicorn` 时，当前工作目录应能作为 Python 包根使用；推荐在 **`DAPT` 的上一级** 或 **`DAPT` 目录内** 设置 `PYTHONPATH`（见下文）。

---

## 3. 方式 A：Conda 环境部署（推荐）

> **当前服务器已有可用的训练环境 `medical_bert`**，模型加载已验证通过。  
> **直接复用该环境**，只需额外安装几个纯服务层的包，无需新建环境。

### 3.1 激活已有环境并补装服务依赖

```bash
conda activate medical_bert

# 补装服务层依赖（FastAPI、uvicorn、pydantic-settings、safetensors）
# safetensors 必须安装：模型使用 model.safetensors 格式
pip install fastapi uvicorn pydantic-settings safetensors

# 安装测试依赖（可选）
pip install pytest requests
```

> **如果需要全新隔离环境**（可选）：
> ```bash
> conda create -n kv-bert-serving python=3.10 -y
> conda activate kv-bert-serving
> pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
> pip install -r /data/ocean/DAPT/serving/requirements.txt
> pip install safetensors
> ```

### 3.2 依赖自洽性验证

```bash
python -c "
import torch, transformers, torchcrf, fastapi, pydantic, safetensors
print('torch:', torch.__version__)
print('transformers:', transformers.__version__)
print('torchcrf: OK')
print('fastapi:', fastapi.__version__)
print('pydantic:', pydantic.__version__)
print('safetensors: OK')
"
```

### 3.3 环境变量

将 **模型目录** 与 **noise_bins.json** 换成你在远端的真实路径：

```bash
#必须指定空闲卡
export CUDA_VISIBLE_DEVICES=1

export MODEL_DIR=/data/ocean/DAPT/runs/kv_ner_finetuned_noise_bucket/best
export NOISE_BINS_PATH=/data/ocean/DAPT/workspace/noise_bins.json
export DEVICE=cuda

# 可选：生产高并发时开启动态批
# export ENABLE_DYNAMIC_BATCHING=true
# export BATCH_MAX_SIZE=16
# export BATCH_MAX_WAIT_MS=10.0

# 可选：PyTorch 2.x 编译加速（首次请求会慢，之后提速）
# export USE_TORCH_COMPILE=true
```

> 建议将这些 `export` 写入 `~/.bashrc` 或项目下的 `.env` 文件，重新登录后自动生效。

### 3.4 启动服务

在 **`DAPT` 目录**下执行（保证能 import `serving` 与仓库根目录模块）：

```bash
# 确认 conda 环境已激活
conda activate medical_bert

cd /data/ocean/DAPT
export PYTHONPATH="/data/ocean/DAPT:/data/ocean/DAPT/dapt_eval_package:${PYTHONPATH}"

uvicorn serving.app:app --host 0.0.0.0 --port 8000 --workers 1
```

说明：

- **`--workers 1`**：每个进程加载一份模型；多 worker 会重复占显存。多卡请用不同 `CUDA_VISIBLE_DEVICES` 起多个进程、不同端口。
- 后台持久运行可用 `nohup` 或 `screen`（见下文 §3.6）。
- 仅内网调试可加 `--reload`，生产勿用。

### 3.5 后台持久运行（screen / nohup）

**方式一：screen**（推荐，可随时 attach 查看日志）

```bash
screen -S kv-bert-serving
conda activate medical_bert
cd /data/ocean/DAPT
export PYTHONPATH="/data/ocean/DAPT:/data/ocean/DAPT/dapt_eval_package:${PYTHONPATH}"
export MODEL_DIR=...  # 同上
export NOISE_BINS_PATH=...
uvicorn serving.app:app --host 0.0.0.0 --port 8000 --workers 1
# Ctrl+A D 脱离 screen，进程保持运行
# 重新连接：screen -r kv-bert-serving
```

**方式二：nohup**

```bash
nohup uvicorn serving.app:app \
  --host 0.0.0.0 --port 8000 --workers 1 \
  > /tmp/kv-bert-serving.log 2>&1 &
echo "PID=$!"
tail -f /tmp/kv-bert-serving.log   # 查看启动日志
```

### 3.6 快速探活

```bash
curl -s http://127.0.0.1:8000/health
# 期望：{"status":"ok"}

curl -s http://127.0.0.1:8000/ready
# 模型加载完成后：{"status":"ready"}；加载中：HTTP 503
```

`GET /ready` 返回 HTTP 200 / `{"status":"ready"}` 表示模型已就绪，可开始接受推理请求。

---

## 4. 方式 B：Docker 部署

在 **`DAPT` 仓库根目录**（含 `serving/` 与 `dapt_eval_package/`）构建：

```bash
cd /data/ocean/DAPT

docker build -f serving/Dockerfile -t kv-bert-serving:latest .
```

运行（将宿主上的模型与分桶文件挂载进容器）：

```bash
docker run --gpus all --rm -p 8000:8000 \
  -v /path/on/host/kv_ner_checkpoint:/models/kv_ner_checkpoint:ro \
  -v /path/on/host/noise_bins.json:/models/noise_bins.json:ro \
  -e MODEL_DIR=/models/kv_ner_checkpoint \
  -e NOISE_BINS_PATH=/models/noise_bins.json \
  -e DEVICE=cuda \
  kv-bert-serving:latest
```

按需增加 `-e ENABLE_DYNAMIC_BATCHING=true` 等环境变量。

---

## 5. 测试

### 5.1 冒烟脚本（内置示例 JSON，不依赖本地文件）

服务启动后（**确保 conda 环境已激活**）：

```bash
conda activate medical_bert
cd /data/ocean/DAPT
export PYTHONPATH="/data/ocean/DAPT:/data/ocean/DAPT/dapt_eval_package"
python serving/test_api.py --base-url http://127.0.0.1:8000
```

若服务在另一台机器上，将 `--base-url` 改为 `http://<远端IP>:8000`（注意防火墙与安全组）。

### 5.2 使用远端磁盘上的 JSON 文件（与接口契约一致）

1. 准备一个请求体 JSON 文件，字段与方案一致，例如：

```json
{
  "ocr_text": "姓名：张三\n性别：男",
  "report_title": "病理报告",
  "words_result": []
}
```

2. 用 `curl` 直接投递该文件：

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/extract \
  -H "Content-Type: application/json" \
  -d @/path/on/h200/my_ocr_request.json | jq .
```

### 5.3 自动化单元/集成测试（pytest）

```bash
conda activate medical_bert
```

**使用仓库内自带样例**（`serving/tests/fixtures/sample_ocr_request.json`）：

在 **`serving` 目录**下执行，以便读取 `pytest.ini` 中的标记配置：

```bash
cd /data/ocean/DAPT/serving
export PYTHONPATH="/data/ocean/DAPT:/data/ocean/DAPT/dapt_eval_package"
export SERVING_BASE_URL=http://127.0.0.1:8000

pytest tests/test_extract_from_file.py -v -m integration
```

**使用你在 H200 上的任意 JSON 文件**（例如评测导出的 OCR 样本）：

```bash
cd /data/ocean/DAPT/serving
export PYTHONPATH="/data/ocean/DAPT:/data/ocean/DAPT/dapt_eval_package"
export EXTRACT_TEST_JSON=/data/ocean/DAPT/some_dir/my_payload.json
export SERVING_BASE_URL=http://127.0.0.1:8000

pytest tests/test_extract_from_file.py -v -m integration
```

若服务未启动或 `/ready` 非 200，`test_extract_payload_from_json_file` 会跳过并提示原因；`test_health_endpoints` 仍会尝试访问 `/health`。

---

## 6. 完整操作流水线（针对当前 H200 服务器快速参考）

> 当前服务器状态：conda 25.5.1 ✓，CUDA 12.8 ✓，GPU 5 和 7 空闲（~141 GB）。  
> 代码路径已固定为 `/data/ocean/DAPT`，通过 `git pull` 与本地同步。

```bash
# ① 拉取最新代码（本地通过 GitHub 同步到此路径）
cd /data/ocean/DAPT
git pull

# ② 确认可用 GPU（选 5 或 7）
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv

# ③ 激活已有训练环境（已验证模型可正常加载）
conda activate medical_bert

# ④ 补装服务层依赖（safetensors 必须，模型为 .safetensors 格式）
pip install fastapi uvicorn pydantic-settings safetensors pytest requests

# ⑤ 验证 GPU（必须指定空闲卡，否则 VLLM 占用的 GPU 0 会报错）
CUDA_VISIBLE_DEVICES=5 python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0))
print('Free mem:', round(torch.cuda.mem_get_info()[0]/1e9, 1), 'GB')
"

# ⑥ 验证所有依赖自洽
python -c "import torch, transformers, torchcrf, fastapi, pydantic, safetensors; print('ALL OK')"

# ⑦ 设置环境变量
export CUDA_VISIBLE_DEVICES=5          # 使用空闲的 GPU 5
export MODEL_DIR=/data/ocean/DAPT/runs/kv_ner_finetuned_noise_bucket/best
export NOISE_BINS_PATH=/data/ocean/DAPT/workspace/noise_bins.json
export DEVICE=cuda
export PYTHONPATH="/data/ocean/DAPT:/data/ocean/DAPT/dapt_eval_package"

# ⑨ 后台启动服务（screen 方式）
screen -S kv-bert
cd /data/ocean/DAPT
uvicorn serving.app:app --host 0.0.0.0 --port 8000 --workers 1
# Ctrl+A D 脱离 screen

# ⑩ 等待模型加载（约 10~30 秒）
watch -n2 'curl -s http://127.0.0.1:8000/ready'
# 出现 {"status":"ready"} 后 Ctrl+C

# ⑪ 运行集成测试
cd /data/ocean/DAPT/serving
export SERVING_BASE_URL=http://127.0.0.1:8000
pytest tests/test_extract_from_file.py -v -m integration
```

---

## 7. H200 上的实践建议

1. **显存**：MacBERT-base + BiLSTM + CRF 单卡通常远小于 H200 显存；仍建议 `nvidia-smi` 观察峰值。
2. **QPS**：提高并发时可开启 `ENABLE_DYNAMIC_BATCHING=true`，并视延迟调 `BATCH_MAX_WAIT_MS`。
3. **多卡**：每张卡起一个进程 + 独立端口 + 前置负载均衡，避免单进程多 worker 重复加载模型占满显存。
4. **安全**：生产环境应对外网加鉴权、TLS，且不要将模型路径暴露在日志中。
5. **conda 管理**：serving 环境建议与训练环境分开，用 `conda env export > env_serving.yml` 备份，方便迁移。

---

## 8. 常见问题

| 现象 | 处理 |
|------|------|
| `torch.cuda.is_available()` 返回 `False` | 确认设置了 `CUDA_VISIBLE_DEVICES=5`（或 7）；GPU 0-4,6 已被 VLLM 占满 |
| `CUDA error: out of memory` 启动时就 OOM | 未设 `CUDA_VISIBLE_DEVICES`，模型落到了满载的 GPU 上 |
| `torchcrf` 安装失败 / `No module named torchcrf` | 改用 `pip install pytorch-crf`（import 写法不变） |
| `MODEL_NOT_READY` / `/ready` 503 | 检查 `MODEL_DIR`、`NOISE_BINS_PATH` 是否存在；查看服务端日志 |
| `ImportError: noise_fusion` | 确认 `PYTHONPATH` 包含 `DAPT` 根目录与 `dapt_eval_package` |
| `No module named 'pydantic_settings'` | `pip install pydantic-settings` |
| Python 3.13 兼容报错 | conda 环境必须用 `python=3.10`，不要用系统 3.13 |
