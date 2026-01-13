import argparse
import re
from pathlib import Path


def extract_keys(src: Path) -> list[str]:
    keys = []
    pat = re.compile(r"^键名:\s*(.+)$")
    for line in src.read_text(encoding="utf-8").splitlines():
        m = pat.match(line.strip())
        if m:
            keys.append(m.group(1).strip())
    return keys


def write_keys(keys: list[str], dst: Path):
    dst.write_text("\n".join(keys) + "\n", encoding="utf-8")
    print(f"写入 {len(keys)} 条键名 -> {dst}")


def main():
    parser = argparse.ArgumentParser(description="从 biaozhu_keys_freq*.txt 提取纯键名列表")
    parser.add_argument("--freq", type=str, required=True, help="源频次文件，如 biaozhu_keys_freq.txt")
    parser.add_argument("--output", type=str, required=True, help="输出的纯键名文件，如 biaozhu_keys_only.txt")
    args = parser.parse_args()

    src = Path(args.freq).resolve()
    dst = Path(args.output).resolve()
    if not src.exists():
        raise FileNotFoundError(f"未找到源文件: {src}")
    keys = extract_keys(src)
    write_keys(keys, dst)


if __name__ == "__main__":
    main()

