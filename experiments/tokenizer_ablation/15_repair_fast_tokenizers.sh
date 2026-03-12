#!/usr/bin/env bash
set -euo pipefail

# Repair/re-generate tokenizer.json for each tokenizer variant.
# This is critical when vocab.txt was edited/merged but tokenizer.json is stale.
# Downstream KV-NER/EBQA requires fast tokenizers for return_offsets_mapping.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/config.env"

: "${OUT_ROOT:?OUT_ROOT must be set in config.env}"

TOK_ROOT="${OUT_ROOT}/tokenizers"

if [[ ! -d "${TOK_ROOT}" ]]; then
  echo "Tokenizer root not found: ${TOK_ROOT}" >&2
  echo "Run: bash 10_make_tokenizers.sh first" >&2
  exit 1
fi

echo "[repair_fast_tokenizers] OUT_ROOT=${OUT_ROOT}"
echo "[repair_fast_tokenizers] TOK_ROOT=${TOK_ROOT}"

echo "[1/2] Repair tokenizer.json for each variant"
for d in "${TOK_ROOT}"/*; do
  [[ -d "$d" ]] || continue
  echo "---"
  echo "Repairing: $d"
  python /data/ocean/DAPT/repair_fast_tokenizer.py --tokenizer_dir "$d"
done

echo "[2/2] Quick sanity check (fast vs slow)"
for d in "${TOK_ROOT}"/*; do
  [[ -d "$d" ]] || continue
  echo "---"
  echo "Check FAST: $d"
  python "${SCRIPT_DIR}/debug_tokenizer_settings.py" --tokenizer_path "$d" --use_fast true --strict
  echo "Check SLOW: $d"
  python "${SCRIPT_DIR}/debug_tokenizer_settings.py" --tokenizer_path "$d" --use_fast false || true
done

echo "Done. All tokenizer variants should now have a consistent fast backend (tokenizer.json)."
