#!/usr/bin/env python3
"""
中途新增角色工具 —— 基于已有剧情和现有角色，生成一个能融入故事的新角色。

用法:
    python add_character.py --project 少女乐队
    python add_character.py --project 少女乐队 --auto   # 自动模式，让 AI 决定加什么角色
"""

import os
import sys
import json
import argparse
import re
from config import DEFAULT_PROJECT, ensure_project_dirs, get_project_paths, get_project_list
from tools_call_deepseek import access_deepseek


def load_chapter_summaries(project: str) -> list:
    folder = get_project_paths(project)["novel"]
    chapters = []
    if not os.path.exists(folder):
        return chapters
    for f in sorted(os.listdir(folder)):
        if not f.endswith(".json"):
            continue
        match = re.search(r"(\d+)", f)
        ch_num = int(match.group(1)) if match else 0
        with open(os.path.join(folder, f), "r", encoding="utf-8") as fp:
            try:
                data = json.load(fp)
                chapters.append({
                    "num": ch_num,
                    "summary": data.get("summary", ""),
                    "filename": f.replace(".json", ""),
                })
            except (json.JSONDecodeError, IOError):
                continue
    chapters.sort(key=lambda c: c["num"])
    return chapters


def load_characters(project: str) -> list:
    folder = get_project_paths(project)["characters"]
    chars = []
    if not os.path.exists(folder):
        return chars
    for f in sorted(os.listdir(folder)):
        if not f.endswith(".json"):
            continue
        with open(os.path.join(folder, f), "r", encoding="utf-8") as fp:
            try:
                data = json.load(fp)
                chars.append({
                    "name": data.get("name", ""),
                    "intro": data.get("role_introduction", ""),
                    "personality": data.get("role_personality", ""),
                    "experience": data.get("role_experience", ""),
                })
            except (json.JSONDecodeError, IOError):
                continue
    return chars


CHARACTER_PROMPT = """你是一位角色设计师。请基于下面的已有剧情、现有角色和用户需求，设计一个新角色。

## 已有剧情（章数: {chapter_count}）
{story_context}

## 现有角色
{existing_characters}

## 新角色要求
{requirements}

请生成角色卡，必须直接输出 JSON（不要用代码块包裹）：
{{
    "name": "角色名（2-4字中文）",
    "role_introduction": "一句话简介（20-50字，说明角色身份和在故事中的定位）",
    "role_experience": "角色背景故事（50-200字，要自然地连接到已有剧情）",
    "role_personality": "角色性格描述（10-50字）",
    "plot_history": []
}}

要求：
- 角色名不能和现有角色重复
- 角色背景要能自然融入已有剧情，不要突兀
- 性格要和现有角色形成互补或冲突（不要重复已有角色的性格类型）
- 要有明确的入局理由——为什么这个角色现在才出现？"""


def generate_character(project: str, requirements: str) -> dict:
    chapters = load_chapter_summaries(project)
    characters = load_characters(project)

    if not chapters:
        print("⚠️ 当前没有任何章节，将以无上下文方式生成角色。")
        story_context = "（暂无章节）"
        chapter_count = 0
    else:
        recent = chapters[-5:]
        story_context = "\n".join(
            f"第{c['num']}章: {c['summary'][:100]}" for c in recent
        )
        chapter_count = len(chapters)

    existing_names = [c["name"] for c in characters]
    existing_text = "\n".join(
        f"- {c['name']}: {c['intro']} (性格: {c['personality']})" for c in characters
    )

    prompt = CHARACTER_PROMPT.format(
        story_context=story_context,
        chapter_count=chapter_count,
        existing_characters=existing_text,
        requirements=requirements,
    )
    prompt += f"\n\n注意: 角色名不能是以下任何一个: {', '.join(existing_names)}"

    for attempt in range(3):
        try:
            raw = access_deepseek(prompt, model_name="deepseek-v4-pro", max_retry=1,
                                  caller="add_character")
            data = json.loads(raw)
            for field in ["name", "role_introduction", "role_experience", "role_personality"]:
                if field not in data:
                    raise ValueError(f"缺少字段: {field}")
            data.setdefault("plot_history", [])
            return data
        except Exception as e:
            print(f"  ⚠️ 第 {attempt+1} 次生成失败: {e}")
    raise Exception("生成失败，请重试")


def save_character(project: str, data: dict) -> str:
    folder = get_project_paths(project)["characters"]
    ensure_project_dirs(project)
    name_for_file = data["name"].replace("/", "_")
    filepath = os.path.join(folder, f"{name_for_file}.json")
    if os.path.exists(filepath):
        base = filepath.replace(".json", "")
        counter = 2
        while os.path.exists(f"{base}_{counter}.json"):
            counter += 1
        filepath = f"{base}_{counter}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    return filepath


def interactive(project: str = DEFAULT_PROJECT):
    print("=" * 60)
    print("  中途新增角色工具")
    print("=" * 60)

    characters = load_characters(project)
    chapters = load_chapter_summaries(project)

    print(f"\n📖 当前项目: {project}")
    print(f"   已有 {len(chapters)} 章, {len(characters)} 个角色")

    if characters:
        print("   现有角色:", ", ".join(c["name"] for c in characters))

    if chapters:
        latest = chapters[-1]
        print(f"   最新剧情: 第{latest['num']}章 {latest['summary'][:60]}...")

    print("\n需要什么样的新角色？")
    print("  例如: 一个转学生，性格外向但内心有创伤，和主角在音乐理念上有冲突")
    print("  或者: 主角失散多年的姐姐，突然出现在 livehouse 演出中")
    print("  直接回车让我根据剧情缺口自动设计")
    requirements = input("\n> ").strip()

    if not requirements:
        requirements = (
            f"当前已写了 {len(chapters)} 章，现有角色 {len(characters)} 个。"
            "请分析剧情走向和现有角色关系网的缺口，设计一个有存在价值的新角色。"
            "可以是盟友、对手、导师、或在情感线上制造张力的角色。"
        )

    print(f"\n🎭 正在生成新角色...")
    data = generate_character(project, requirements)

    print(f"\n  姓名: {data['name']}")
    print(f"  简介: {data['role_introduction']}")
    print(f"  性格: {data['role_personality']}")
    print(f"  背景: {data['role_experience']}")

    confirm = input("\n保存到角色库？(Y/n) : ").strip().lower()
    if confirm in ("n", "no"):
        print("已取消。")
        return

    filepath = save_character(project, data)
    print(f"\n✅ 已保存: {filepath}")
    print(f"   刷新 Gradio 页面后，角色下拉框会出现「{data['name']}」。")
    print(f"   在「本章要求」里写上一句「{data['name']}在本章首次登场」即可引入。")


def auto_generate(project: str = DEFAULT_PROJECT):
    print("🤖 自动模式: 分析剧情缺口，自动设计新角色\n")
    data = generate_character(
        project,
        "分析现有剧情和角色关系，自动判断需要什么类型的新角色来推动剧情。",
    )
    print(f"  姓名: {data['name']}")
    print(f"  简介: {data['role_introduction']}")
    filepath = save_character(project, data)
    print(f"\n✅ 已保存: {filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="中途新增角色工具")
    parser.add_argument("--project", type=str, default=DEFAULT_PROJECT, help="小说项目名称")
    parser.add_argument("--auto", action="store_true", help="自动模式")
    args = parser.parse_args()

    ensure_project_dirs(args.project)
    if args.auto:
        auto_generate(args.project)
    else:
        interactive(args.project)
