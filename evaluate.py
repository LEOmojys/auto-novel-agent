import json
import os
import re
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional

from config import NOVEL_FOLDER, CHARACTERS_FOLDER
from tools_call_deepseek import access_deepseek


def load_chapters(folder: str) -> List[Dict[str, Any]]:
    chapters = []
    if not os.path.exists(folder):
        return chapters
    for filename in sorted(os.listdir(folder)):
        if not filename.endswith(".json"):
            continue
        filepath = os.path.join(folder, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                data["_filename"] = filename.replace(".json", "")
                data["_chapter_num"] = _extract_chapter_number(filename)
                chapters.append(data)
            except (json.JSONDecodeError, IOError) as e:
                print(f"跳过 {filename}: {e}")
    return chapters


def load_characters(folder: str) -> Dict[str, Dict[str, Any]]:
    characters = {}
    if not os.path.exists(folder):
        return characters
    for filename in os.listdir(folder):
        if not filename.endswith(".json"):
            continue
        filepath = os.path.join(folder, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                name = data.get("name", "")
                characters[name] = data
            except (json.JSONDecodeError, IOError) as e:
                print(f"跳过角色文件 {filename}: {e}")
    return characters


def _extract_chapter_number(filename: str) -> int:
    match = re.search(r"(\d+)", filename)
    return int(match.group(1)) if match else 0


def _resolve_char_names(char_data: Dict[str, Any]) -> List[str]:
    raw = char_data.get("name", "")
    return [n.strip() for n in raw.split("/") if n.strip()]


def _count_chinese_chars(text: str) -> int:
    return len(re.findall(r"[\u4e00-\u9fff]", text))


def _count_sensory_words(text: str) -> Dict[str, int]:
    patterns = {
        "视觉": r"[看见瞧瞅瞄审视眺望观瞻]",
        "听觉": r"[听闻聆](?!到|说|讯|名|人)",
        "嗅觉": r"[闻嗅](?!到|说|讯|名|人|声)",
        "触觉": r"[触碰抚摸按捏握抓揉擦拍打]|冰凉|温暖|粗糙|柔软|坚硬",
        "味觉": r"[尝品吃]|酸|甜|苦|辣|咸|涩|淡",
    }
    result = {}
    for sense, pattern in patterns.items():
        result[sense] = len(re.findall(pattern, text))
    return result


def _count_dialogue_ratio(text: str) -> float:
    dialogue_chars = 0
    in_quote = False
    for ch in text:
        if ch in "\u201c\u201d\u300c\u300d\"":
            in_quote = not in_quote
            continue
        if in_quote:
            dialogue_chars += 1
    total = len(text.replace("\n", "").replace(" ", "").replace("\u3000", ""))
    if total == 0:
        return 0.0
    return round(dialogue_chars / total * 100, 1)


def rule_based_metrics(
    passage: str,
    char_data: Dict[str, Any],
    target_words: int,
) -> Dict[str, Any]:
    chinese_chars = _count_chinese_chars(passage)
    deviation_pct = round((chinese_chars - target_words) / target_words * 100, 1)
    sensory = _count_sensory_words(passage)
    sensory_total = sum(sensory.values())
    sensory_density = round(sensory_total / (chinese_chars / 1000), 1) if chinese_chars else 0
    dialogue_ratio = _count_dialogue_ratio(passage)

    all_names = []
    for name_key in char_data:
        all_names.extend(_resolve_char_names(char_data[name_key]))

    name_counts = {}
    for name in all_names:
        count = passage.count(name)
        if count > 0:
            name_counts[name] = count
    appeared = list(name_counts.keys())

    paragraphs = [p.strip() for p in passage.split("\n\n") if p.strip()]
    para_count = len(paragraphs)

    return {
        "chinese_chars": chinese_chars,
        "target_words": target_words,
        "deviation_pct": deviation_pct,
        "sensory": sensory,
        "sensory_total": sensory_total,
        "sensory_density": sensory_density,
        "dialogue_ratio": dialogue_ratio,
        "characters_appeared": appeared,
        "character_mentions": name_counts,
        "paragraph_count": para_count,
    }


EVALUATION_PROMPT = """你是一位资深文学编辑，需要评估一段小说的写作质量。请只根据提供的文本和角色信息，给出评分和简短理由。

## 评分标准（1-5 分）
- 5：优秀，达到出版级水平
- 4：良好，有小瑕疵但不影响阅读
- 3：及格，存在明显问题
- 2：较差，多处硬伤
- 1：很差，几乎不可读

## 评估维度

**文风一致性**：是否保持日常系/空气系风格？是否做到 Show-Don't-Tell？细节描写是否克制而精准？情感表达是否含蓄内敛？
**角色一致性**：每个登场角色的言行是否符合其人物设定？性格是否前后统一？
**文学流畅度**：句子节奏是否自然？有无明显的 AI 感、翻译腔、重复句式？段落衔接是否流畅？
**场景完整度**：本章是否构成了一个有起承转合的完整场景？读者读完后是否有"得到了什么"的感觉？
**对话自然度**：对话是否简短真实？是否有潜台词？沉默和停顿的运用是否恰当？
**内容一致性**：本章是否完成了“本章要求”中的核心事件、情节推进和重点表达？是否有明显遗漏、偏离或错位？

## 角色信息
{character_info}

## 本章要求
{chapter_requirement}

## 需要评估的文本
{passage}

请严格按以下 JSON 格式返回（不要输出其他内容）：
```json
{{
  "文风一致性": {{"score": 4, "reason": "一句话理由"}},
  "角色一致性": {{"score": 4, "reason": "一句话理由"}},
  "文学流畅度": {{"score": 4, "reason": "一句话理由"}},
  "场景完整度": {{"score": 4, "reason": "一句话理由"}},
  "对话自然度": {{"score": 4, "reason": "一句话理由"}},
  "内容一致性": {{"score": 4, "reason": "一句话理由"}}
}}
```
"""


EVALUATION_DIMS = [
    "文风一致性",
    "角色一致性",
    "文学流畅度",
    "场景完整度",
    "对话自然度",
    "内容一致性",
]


def _format_character_info(char_data: Dict[str, Any]) -> str:
    lines = []
    for name_key, info in char_data.items():
        lines.append(
            f"- {info.get('name', name_key)}: {info.get('role_introduction', '')} "
            f"性格: {info.get('role_personality', '')}"
        )
    return "\n".join(lines)


def llm_evaluate_chapter(
    passage: str,
    char_data: Dict[str, Any],
    chapter_label: str,
    chapter_requirement: str = "",
    max_retry: int = 2,
) -> Dict[str, Any]:
    char_info = _format_character_info(char_data)
    requirement_text = chapter_requirement.strip() or "（未提供本章要求，请仅依据文本本身判断，不要因缺失要求而扣分）"
    prompt = EVALUATION_PROMPT.format(
        character_info=char_info,
        chapter_requirement=requirement_text,
        passage=passage,
    )

    for attempt in range(max_retry):
        try:
            raw = access_deepseek(prompt, max_retry=1, model_name="deepseek-v4-flash",
                                  caller="evaluate", return_tokens=True)[0]
            match = re.search(r"```json\s*([\s\S]*?)\s*```", raw)
            if match:
                raw = match.group(1)
            result = json.loads(raw)
            scores = {}
            reasons = {}
            for dim in EVALUATION_DIMS:
                item = result.get(dim, {})
                scores[dim] = item.get("score", 0)
                reasons[dim] = item.get("reason", "")
            avg_score = round(sum(scores.values()) / len(scores), 1) if scores else 0
            return {"scores": scores, "reasons": reasons, "avg_score": avg_score}
        except Exception as e:
            print(f"  [{chapter_label}] 评委调用失败 (尝试 {attempt+1}/{max_retry}): {e}")
            if attempt < max_retry - 1:
                time.sleep(2)
    return {
        "scores": {d: 0 for d in EVALUATION_DIMS},
        "reasons": {},
        "avg_score": 0,
    }


def evaluate_single(
    passage: str,
    target_words: int = 6000,
    label: str = "",
    chapter_requirement: str = "",
) -> Dict[str, Any]:
    characters = load_characters(CHARACTERS_FOLDER)
    rules = rule_based_metrics(passage, characters, target_words)
    llm = llm_evaluate_chapter(
        passage,
        characters,
        label or "未命名章节",
        chapter_requirement=chapter_requirement,
    )

    print(f"  字数: {rules['chinese_chars']} ({rules['deviation_pct']:+.1f}%)  "
          f"感官密度: {rules['sensory_density']}/千字  对话占比: {rules['dialogue_ratio']}%")
    print(f"  出场角色: {', '.join(rules['characters_appeared']) if rules['characters_appeared'] else '(无)'}")
    print(f"  LLM评分: 文风{llm['scores'].get('文风一致性','?')} "
          f"角色{llm['scores'].get('角色一致性','?')} "
          f"流畅{llm['scores'].get('文学流畅度','?')} "
          f"场景{llm['scores'].get('场景完整度','?')} "
          f"对话{llm['scores'].get('对话自然度','?')} "
          f"内容{llm['scores'].get('内容一致性','?')} "
          f"→ 均分 {llm['avg_score']}")
    print()
    for dim, reason in llm.get("reasons", {}).items():
        if reason:
            print(f"  [{dim}] {reason}")
    return {"rule_metrics": rules, "llm_scores": llm["scores"], "llm_reasons": llm["reasons"], "llm_avg": llm["avg_score"]}


def evaluate_all(
    target_words: int = 6000,
    model_label: str = "deepseek-v4-pro",
) -> Dict[str, Any]:
    chapters = load_chapters(NOVEL_FOLDER)
    characters = load_characters(CHARACTERS_FOLDER)

    if not chapters:
        print("没有找到已生成的章节，请先运行生成。")
        return {}

    print(f"加载了 {len(chapters)} 个章节, {len(characters)} 个角色")
    print(f"目标字数: {target_words} 字\n")

    results = []
    total_llm_scores = {d: [] for d in EVALUATION_DIMS}

    for i, ch in enumerate(chapters):
        label = ch["_filename"]
        passage = ch.get("passage", "")
        chapter_requirement = ch.get("chapter_requirement") or ch.get("this_passage_request") or ch.get("requirement", "")
        num = ch["_chapter_num"]

        if not passage.strip():
            continue

        header = f"[{i+1}/{len(chapters)}] Ch.{num} {label}"
        print(f"{header}")
        print("-" * len(header))

        rules = rule_based_metrics(passage, characters, target_words)
        llm = llm_evaluate_chapter(
            passage,
            characters,
            label,
            chapter_requirement=chapter_requirement,
        )

        entry = {
            "filename": label,
            "chapter_num": num,
            "chapter_requirement": chapter_requirement,
            "rule_metrics": rules,
            "llm_scores": llm["scores"],
            "llm_reasons": llm["reasons"],
            "llm_avg": llm["avg_score"],
        }
        results.append(entry)

        for dim, score in llm["scores"].items():
            if score > 0:
                total_llm_scores[dim].append(score)

        print(f"  字数: {rules['chinese_chars']} ({rules['deviation_pct']:+.1f}%)  "
              f"感官密度: {rules['sensory_density']}/千字  对话占比: {rules['dialogue_ratio']}%")
        print(f"  出场角色: {', '.join(rules['characters_appeared']) if rules['characters_appeared'] else '(无)'}")
        print(f"  LLM评分: 文风{llm['scores'].get('文风一致性','?')} "
              f"角色{llm['scores'].get('角色一致性','?')} "
              f"流畅{llm['scores'].get('文学流畅度','?')} "
              f"场景{llm['scores'].get('场景完整度','?')} "
              f"对话{llm['scores'].get('对话自然度','?')} "
              f"内容{llm['scores'].get('内容一致性','?')} "
              f"→ 均分 {llm['avg_score']}")
        print()

    avg_scores = {}
    for dim, score_list in total_llm_scores.items():
        avg_scores[dim] = round(sum(score_list) / len(score_list), 1) if score_list else 0

    all_llm_avgs = [r["llm_avg"] for r in results if r["llm_avg"] > 0]
    overall_avg = round(sum(all_llm_avgs) / len(all_llm_avgs), 1) if all_llm_avgs else 0

    all_devs = [r["rule_metrics"]["deviation_pct"] for r in results]
    median_dev = sorted(all_devs)[len(all_devs) // 2] if all_devs else 0

    summary = {
        "model": model_label,
        "evaluator_model": "deepseek-v4-flash",
        "chapters_evaluated": len(results),
        "target_words": target_words,
        "timestamp": datetime.now().isoformat(),
        "dimension_averages": avg_scores,
        "overall_avg": overall_avg,
        "median_word_deviation_pct": median_dev,
        "chapters": results,
    }

    print("=" * 50)
    print("  评估汇总")
    print("=" * 50)
    print(f"  评估章数: {len(results)}")
    print(f"  综合均分: {overall_avg}/5.0")
    print(f"  篇幅偏差中位数: {median_dev:+.1f}%")
    print()
    for dim, avg in avg_scores.items():
        bar = "█" * int(avg) + "░" * (5 - int(avg))
        print(f"  {dim:　<6s}  {bar}  {avg}/5.0")
    print()

    return summary


def save_report(summary: Dict[str, Any], output_dir: str = "."):
    if not summary:
        return
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"eval_report_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"报告已保存: {filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="小说生成质量评估工具")
    parser.add_argument("-w", "--words", type=int, default=6000, help="目标字数 (默认 6000)")
    parser.add_argument("-c", "--chapter", type=int, default=None, help="只评估指定章节号")
    parser.add_argument("-t", "--text", type=str, default=None, help="直接评估一段文本")
    parser.add_argument("--no-save", action="store_true", help="不保存报告文件")
    args = parser.parse_args()

    if args.text:
        evaluate_single(args.text, target_words=args.words, label="手动输入文本")
    elif args.chapter:
        chapters = load_chapters(NOVEL_FOLDER)
        found = None
        for ch in chapters:
            if ch["_chapter_num"] == args.chapter:
                found = ch
                break
        if found:
            print(f"评估: Ch.{args.chapter} {found['_filename']}")
            print("-" * 30)
            report = evaluate_single(
                found["passage"],
                target_words=args.words,
                label=found["_filename"],
                chapter_requirement=found.get("chapter_requirement") or found.get("this_passage_request") or found.get("requirement", ""),
            )
            if not args.no_save:
                save_report({"chapters": [{
                    "filename": found["_filename"],
                    "chapter_num": found["_chapter_num"],
                    "chapter_requirement": found.get("chapter_requirement") or found.get("this_passage_request") or found.get("requirement", ""),
                    **report,
                }]}, output_dir=os.path.join(os.path.dirname(NOVEL_FOLDER), "evaluate"))
        else:
            print(f"未找到第 {args.chapter} 章")
    else:
        report = evaluate_all(target_words=args.words)
        if not args.no_save:
            save_report(report, output_dir=os.path.join(os.path.dirname(NOVEL_FOLDER), "evaluate"))
