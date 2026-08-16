#!/usr/bin/env python3
"""Token 用量统计工具。读取 token_usage.jsonl 日志，按调用环节汇总。

定价说明（¥/百万 token）:
  deepseek-v4-pro:   输入 ¥2  | 输出 ¥8
  deepseek-v4-flash: 输入 ¥0.5 | 输出 ¥2
"""

import json
import os
from collections import defaultdict
from pathlib import Path

LOG_PATH = os.path.join(Path(__file__).resolve().parent, "logs", "token_usage.jsonl")

PRICING = {
    "deepseek-v4-pro":  {"input": 2.0, "output": 8.0},
    "deepseek-v4-flash": {"input": 0.5, "output": 2.0},
}


def _get_price(model: str, token_type: str) -> float:
    for key, prices in PRICING.items():
        if key in model:
            return prices.get(token_type, 0) / 1_000_000
    return PRICING["deepseek-v4-pro"][token_type] / 1_000_000


def load_entries():
    if not os.path.exists(LOG_PATH):
        print("暂无 token 日志。")
        return []
    entries = []
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries


def main():
    entries = load_entries()
    if not entries:
        return

    by_caller = defaultdict(lambda: {"count": 0, "prompt": 0, "completion": 0})
    by_model = defaultdict(lambda: {"count": 0, "prompt": 0, "completion": 0})

    total_prompt = 0
    total_completion = 0

    for e in entries:
        c = e.get("caller", "unknown")
        m = e.get("model", "unknown")
        p = e.get("prompt_tokens", 0)
        comp = e.get("completion_tokens", 0)

        by_caller[c]["count"] += 1
        by_caller[c]["prompt"] += p
        by_caller[c]["completion"] += comp

        by_model[m]["count"] += 1
        by_model[m]["prompt"] += p
        by_model[m]["completion"] += comp

        total_prompt += p
        total_completion += comp

    total_input_cost = sum(
        e["prompt_tokens"] * _get_price(e.get("model", ""), "input")
        for e in entries
    )
    total_output_cost = sum(
        e["completion_tokens"] * _get_price(e.get("model", ""), "output")
        for e in entries
    )

    print("=" * 75)
    print("  Token 用量统计")
    print("=" * 75)
    print(f"  总调用次数: {len(entries)}")
    print(f"  输入 tokens: {total_prompt:,}  (预估 ¥{total_input_cost:.4f})")
    print(f"  输出 tokens: {total_completion:,}  (预估 ¥{total_output_cost:.4f})")
    print(f"  合计 tokens: {total_prompt + total_completion:,}  (预估 ¥{total_input_cost + total_output_cost:.4f})")
    print()

    print("  按调用环节:")
    print(f"  {'环节':<22s} {'次数':>5s} {'输入':>10s} {'输出':>10s} {'合计':>10s}")
    print("  " + "-" * 59)
    for caller in sorted(by_caller.keys()):
        stats = by_caller[caller]
        t = stats["prompt"] + stats["completion"]
        print(f"  {caller:<22s} {stats['count']:>5d} {stats['prompt']:>10,d} {stats['completion']:>10,d} {t:>10,d}")

    print()
    print("  按模型:")
    print(f"  {'模型':<22s} {'次数':>5s} {'输入':>10s} {'输出':>10s} {'合计':>10s}")
    print("  " + "-" * 59)
    for model in sorted(by_model.keys()):
        stats = by_model[model]
        t = stats["prompt"] + stats["completion"]
        print(f"  {model:<22s} {stats['count']:>5d} {stats['prompt']:>10,d} {stats['completion']:>10,d} {t:>10,d}")

    print()

    per_chapter_prompt = total_prompt / max(by_caller.get("writting_novel", {}).get("count", 1), 1)
    per_chapter_completion = total_completion / max(by_caller.get("writting_novel", {}).get("count", 1), 1)
    print(f"  生成一章平均: 输入 {per_chapter_prompt:,.0f} tokens, 输出 {per_chapter_completion:,.0f} tokens")


if __name__ == "__main__":
    main()
