from kv_llm.kv_nsp import LlmKvnspCollator, extract_direct_pairs
from kv_llm.span_corruption import build_span_corruption_text, select_entity_spans


def test_build_span_corruption_text_uses_sentinels():
    source, target = build_span_corruption_text(
        "姓名张三诊断肺炎",
        [(2, 4), (6, 8)],
        sentinels=["<extra_id_0>", "<extra_id_1>", "<extra_id_2>"],
    )
    assert source == "姓名<extra_id_0>诊断<extra_id_1>"
    assert target == "<extra_id_0>张三<extra_id_1>肺炎<extra_id_2>"


def test_select_entity_spans_prefers_dictionary_terms():
    import random

    spans = select_entity_spans(
        "姓名张三诊断肺炎",
        ["张三", "肺炎"],
        mask_prob=1.0,
        max_spans=4,
        rng=random.Random(42),
    )
    assert spans == [(2, 4), (6, 8)]


def test_extract_direct_pairs_supports_pair_list():
    pairs = extract_direct_pairs(
        {
            "pairs": [
                {"key": "姓名", "value": "张三"},
                {"key": "诊断", "value": "肺炎"},
            ]
        }
    )
    assert pairs == [("姓名", "张三"), ("诊断", "肺炎")]


def test_llm_kvnsp_collator_uses_sep_format():
    class _Tok:
        sep_token = "<kv_sep>"

    collator = LlmKvnspCollator(tokenizer=_Tok(), sep_token="<kv_sep>")
    assert collator._pair_text("姓名", "张三") == "姓名<kv_sep>张三"
