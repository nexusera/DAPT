未完成：！！！重要：：：：混合16%通用数据，1:5，1:6。
```

## 输出与保存建议：
	- `--output_dir` 下应包含 `predictions.json`（模型预测）与 `eval_summary.json`（聚合指标）。
	- 把每个实验的 `eval_summary.json` 汇总到单独目录（例如 `results/compare/`）便于横向对比。

```
可能用到的辅助命令：

ls -lt 
按照时间顺序列出来文件

nvidia-smi | less
看GPU使用情况
```


第零步：源头数据清洗与提取 (Source Data Cleaning)
这一步是从海量 JSON/TXT 中提取纯文本，并做物理去重。

执行脚本: extract_and_dedup_json_v2.py

作用: 扫描 /data/ocean/semi_pic，区分 Type A/B 提取文本，MD5 去重。
产出: train.txt (干净的纯文本语料)。

命令:

Bash

cd /data/ocean/bpe_workspace
python extract_and_dedup_json_v2.py
检查: 看控制台打印的采样数据是否正常。

第一步：重新挖掘 OCR 词表 (WordPiece)
目标：从新的 train.txt 中挖掘高频词，并进行基础清洗（去人名、去噪音）。

执行脚本：train_ocr_clean.py

命令：

Bash

python train_ocr_clean.py
产出：更新 medical_vocab_ocr_only/vocab.txt。



第二步：重新构建 Tokenizer (Merge)
目标：将“挖掘词”+“Key集合”+“(可选)医学词典”+“基座词表”合并，并应用正则智能拆分与 VIP 保护。

分支选择（择一运行）：
- 精简版（去词典，推荐缩短词表）：`python final_merge_v9_regex_split_slim.py`
- 全量版（含词典）：`python final_merge_v9_regex_split.py` 

（可选）LLM 辅助筛选候选词表：
- 场景：对 `medical_vocab_ocr_only/vocab.txt` 做质量过滤，生成 kept/dropped 列表，再用于 Merge。
- 依赖：本地/远端可用的 Qwen3 32B API，环境变量 `LLF_API_BASE`（逗号分隔多个 endpoint）、`OPENAI_API_KEY`（可占位）。
- 示例：
```
export LLF_API_BASE="http://127.0.0.1:8008/v1"
export OPENAI_API_KEY="EMPTY"
python filter_vocab_with_llm.py \
  --vocab medical_vocab_ocr_only/vocab.txt \
  --kept kept_vocab.txt \
  --dropped dropped_vocab.txt \
  --batch_size 64 \
  --topn 50000
```
- 输出：`kept_vocab.txt`（保留词），`dropped_vocab.txt`（含原因），可在 Merge 时替换/追加。

产出：更新 my-medical-tokenizer/ 文件夹。
可选验证：`python inspect_tokenizer_final.py`


第三步：更新 Jieba 外挂词典
目标：把挖掘出的好词（VIP词、短词）做成词典喂给 Jieba，防止 Jieba 把核心词切碎。

执行脚本：generate_jieba_vocab.py

命令：

Bash

python generate_jieba_vocab.py
产出：更新 vocab_for_jieba.txt。


第四步：构建对齐数据集 (Dataset Alignment) —— 耗时步骤，可后台运行
利用 Jieba 和新词典，生成带有 word_ids 的训练数据。

分支选择（与第二步保持一致）：
- 精简版：`python build_dataset_final_slim.py`
- 全量版：`python build_dataset_final.py`

后台运行示例（防 webserver 超时）：
```
nohup python build_dataset_final_slim.py > build_dataset_slim.log 2>&1 &
# nohup: 忽略挂起信号，终端断开进程仍继续
# python build_dataset_final_slim.py: 运行数据对齐脚本
# > build_dataset_slim.log: 将标准输出写入日志
# 2>&1: 把标准错误重定向到同一个日志
# &: 放入后台运行，立即返回 shell

tail -f build_dataset_slim.log
# tail -f: 持续追踪日志末尾，观察进度
```

检查: 确保 TRAIN_FILE 指向 train.txt 且加载 vocab_for_jieba.txt。运行完查看采样，确认 brca1 等词的 word_ids 一致。

产出: processed_dataset/ (HuggingFace Dataset 格式)。


第五步：泄露自检 (Leakage Check) - 必做
虽然我们在第一步做了去重，但为了保险，训练前最后查一次。

执行脚本: check_leakage.py

作用: 检查 Train 和 Test 是否有重复。

预期: 泄露率应该 < 1% (或者为 0)。

命令:

Bash

python check_leakage.py

第六步：正式训练 (Start Training) —— 耗时步骤，可后台运行
一切就绪，开始炼丹。

执行脚本: train_dapt_distributed.py（RoBERTa）或 train_dapt_distributed_mcbert.py（MacBERT）。
⚠️ 输出目录避免覆盖：可用 --output_dir 指定新目录，如：
```
python train_dapt_distributed.py --output_dir /data/ocean/bpe_workspace/output_medical_bert_exp1
```

清理旧文件: 建议先删掉旧的 output 目录，防止混淆：`rm -rf output_medical_bert_v1`（或对应目录）。

后台运行示例（双卡/多卡按需改 nproc 与 CUDA_VISIBLE_DEVICES）：
```
CUDA_VISIBLE_DEVICES=0,1 \  # 只用 0,1 两张卡，如需更多自行调整
nohup torchrun --nproc_per_node=2 train_dapt_distributed.py > training.log 2>&1 &
# torchrun: 启动分布式；--nproc_per_node=2 表示本机启动 2 个进程（两张卡）
# train_dapt_distributed.py: 训练脚本路径，如用 MacBERT 换成对应脚本
# > training.log: 标准输出写入日志
# 2>&1: 标准错误同样写入日志
# nohup/ &: 断开终端仍运行，并放后台

tail -f training.log
# 追踪训练日志，查看 loss/PPL 进度
查看更多日志行数：tail -n 200 training.log（显示末尾 200 行，可改数字）。
````

若远程会话易被断开，推荐完全脱离终端运行，任选其一：



方法 B：tmux 会话保持
```
tmux new -s dapt
cd /data/ocean/bpe_workspace
conda activate medical_bert
CUDA_VISIBLE_DEVICES=2,4 torchrun --nproc_per_node=2 train_dapt_distributed.py | tee training.log
# 退出会话: Ctrl+B 然后 D；重新进入: tmux attach -t dapt
```
如用 MacBERT，脚本改为 train_dapt_distributed_mcbert.py；如用更多 GPU，调整 CUDA_VISIBLE_DEVICES 与 --nproc_per_node。
```
注意修改脚本内 TOKENIZER_PATH/OUTPUT_DIR 指向当前实验的 tokenizer 与输出目录。


