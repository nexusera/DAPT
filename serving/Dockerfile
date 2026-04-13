FROM nvcr.io/nvidia/pytorch:24.03-py3

WORKDIR /app

# 安装 serving 依赖
COPY serving/requirements.txt /app/requirements_serving.txt
RUN pip install --no-cache-dir -r requirements_serving.txt

# 复制全部 DAPT 代码（包含 noise_fusion.py, noise_feature_processor.py, dapt_eval_package/ 等）
COPY . /app/

ENV PYTHONPATH="/app:/app/dapt_eval_package"
ENV MODEL_DIR="/models/kv_ner_checkpoint"
ENV NOISE_BINS_PATH="/models/noise_bins.json"
ENV DEVICE="cuda"
ENV NOISE_MODE="bucket"

EXPOSE 8000

CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
