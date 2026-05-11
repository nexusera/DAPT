# KV-NSP 注意力可视化结果复核命令（固定版）

本文档用于：
- 快速定位一次运行结果目录
- 抽取“是否支撑论文观点”的关键统计
- 导出可回传给助手复核的文本输出
- 同步图片到本地（Git 或打包下载）

> 默认根路径：`/data/ocean/DAPT`  
> 默认结果目录格式：`/data/ocean/DAPT/runs/attention_kv_nsp_YYYYMMDD_HHMMSS`

---

## 0) 进入环境并定位最新结果目录

```bash
cd /data/ocean/DAPT
LATEST=$(ls -dt runs/attention_kv_nsp_* | head -n 1)
OUT="/data/ocean/DAPT/${LATEST}"
export OUT
echo "OUT=${OUT}"
ls -lah "${OUT}"
```

---

## 1) 快速查看关键文件是否齐全

```bash
ls -lah "${OUT}/summary.json" "${OUT}/report.md" "${OUT}/per_sample_metrics.jsonl"
ls -lah "${OUT}/cases" | head -n 30
ls -lah "${OUT}/figures" | head -n 30
```

---

## 2) 一条命令打印“论文判断核心指标”

这条命令会输出：
- 样本量
- 各组 CSAM / Top-k 均值和标准差
- 显著性检验（p-value + Cohen's d）
- H1/H2 是否初步成立（基于正负对照）

```bash
python - <<'PY'
import json, os
out = os.environ.get("OUT")
if not out:
    raise SystemExit("OUT is empty. Please run: export OUT=/data/ocean/DAPT/runs/attention_kv_nsp_xxx")
s = json.load(open(os.path.join(out, "summary.json"), "r", encoding="utf-8"))

print("=== BASIC ===")
print("num_samples:", s.get("num_samples"))
print("groups:", list((s.get("groups") or {}).keys()))

print("\n=== GROUP STATS ===")
for g, info in (s.get("groups") or {}).items():
    c = info.get("csam", {})
    t = info.get("topk_align_rate", {})
    print(f"{g:>14} | CSAM mean={c.get('mean',0):.4f} std={c.get('std',0):.4f} n={c.get('n',0)}"
          f" | Topk mean={t.get('mean',0):.4f} std={t.get('std',0):.4f} n={t.get('n',0)}")

print("\n=== TESTS ===")
tests = s.get("tests") or {}
for k, v in tests.items():
    print(f"{k:>28} | p={v.get('p_value',1):.6g} | d={v.get('cohens_d',0):.4f} | method={v.get('method','n/a')}")

def gmean(name):
    return (((s.get("groups") or {}).get(name) or {}).get("csam") or {}).get("mean")
pos, rev, rnd = gmean("positive"), gmean("reverse"), gmean("random")
print("\n=== HYPOTHESIS QUICK CHECK ===")
if pos is not None and rev is not None:
    print("H1/H2(part-1): pos CSAM > reverse CSAM ?", pos > rev, f"({pos:.4f} vs {rev:.4f})")
if pos is not None and rnd is not None:
    print("H1/H2(part-2): pos CSAM > random  CSAM ?", pos > rnd, f"({pos:.4f} vs {rnd:.4f})")
PY
```

---

## 3) 导出“可贴给助手复核”的文本文件

```bash
python - <<'PY'
import json, os, textwrap
out = os.environ.get("OUT")
if not out:
    raise SystemExit("OUT is empty. Please run: export OUT=/data/ocean/DAPT/runs/attention_kv_nsp_xxx")
s = json.load(open(os.path.join(out, "summary.json"), "r", encoding="utf-8"))
report = open(os.path.join(out, "report.md"), "r", encoding="utf-8").read()

lines = []
lines.append("## SUMMARY.JSON (core)")
lines.append(json.dumps({
    "num_samples": s.get("num_samples"),
    "groups": s.get("groups"),
    "noise_groups": s.get("noise_groups"),
    "tests": s.get("tests"),
    "config": s.get("config"),
}, ensure_ascii=False, indent=2))
lines.append("\n## REPORT.MD")
lines.append(report)

fp = os.path.join(out, "review_bundle.txt")
open(fp, "w", encoding="utf-8").write("\n".join(lines))
print(fp)
PY

wc -l "${OUT}/review_bundle.txt"
```

把以下文件内容发给助手复核即可：
- `${OUT}/review_bundle.txt`

---

## 4) 关键样本抽查（看是否符合直觉）

### 4.1 各组 Top-5 CSAM 样本
```bash
python - <<'PY'
import json, os, collections
out = os.environ["OUT"]
fp = os.path.join(out, "per_sample_metrics.jsonl")
rows = [json.loads(x) for x in open(fp, "r", encoding="utf-8") if x.strip()]
b = collections.defaultdict(list)
for r in rows:
    b[r.get("group","unknown")].append(r)
for g, arr in b.items():
    arr = sorted(arr, key=lambda x: x.get("csam",0), reverse=True)[:5]
    print(f"\n[{g}]")
    for x in arr:
        print(x.get("sample_id"), "csam=", round(x.get("csam",0),4), "topk=", round(x.get("topk_align_rate",0),4))
PY
```

### 4.2 查看可视化图片文件名
```bash
ls -lah "${OUT}/cases" | head -n 50
ls -lah "${OUT}/figures" | head -n 50
```

---

## 5) 把图片同步到本地查看（两种方式）

### 方式 A（推荐）：打包下载，不污染仓库

远端：
```bash
cd /data/ocean/DAPT/runs
BASE=$(basename "${OUT}")
tar -czf "${BASE}.tar.gz" "${BASE}"
echo "/data/ocean/DAPT/runs/${BASE}.tar.gz"
```

本地：
```bash
scp -P 22222 -i ~/hao.li-jumpserver.pem \
  hao.li@192.168.50.79:/data/ocean/DAPT/runs/attention_kv_nsp_*.tar.gz \
  ~/Downloads/
```

### 方式 B：GitHub 同步（如果远端目录在 git 仓库内）

远端：
```bash
cd /data/ocean/DAPT
git checkout -b exp/attn-review-$(date +%Y%m%d_%H%M)
BASE=$(basename "${OUT}")
git add "runs/${BASE}"
git commit -m "Add attention visualization outputs (${BASE})"
git push -u origin HEAD
```

本地：
```bash
cd /Users/shanqi/Documents/BERT_DAPT
git fetch origin
# 把下面分支名替换成远端实际打印的分支名
git checkout exp/attn-review-YYYYMMDD_HHMM
git pull
```

---

## 6) 论文观点判读建议（简版）

可先按以下经验阈值做“初判”：
- `positive` 组 CSAM 均值 > `reverse` / `random`
- `csam_positive_vs_reverse` 与 `csam_positive_vs_random` 的 p-value < 0.05
- Cohen's d：  
  - `|d| >= 0.2` 小效应  
  - `|d| >= 0.5` 中效应  
  - `|d| >= 0.8` 大效应

若以上成立，通常可支撑：
- H1：正样本更集中的 Key→Value 跨段注意力
- H2：hard negative / random negative 破坏对齐

若噪声分组里 `low` 的方差更大，也可支持“低质量文本更分散”的鲁棒性叙述。
