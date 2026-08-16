#!/usr/bin/env python3
"""
故事大纲生成工具 —— 基于世界观和角色卡生成章节大纲。

用法:
    python generate_outline.py                    # 交互式流程
    python generate_outline.py --project 少女乐队  # 指定项目
    python generate_outline.py --auto             # 自动模式
"""

import os
import sys
import json
import argparse
from config import DEFAULT_PROJECT, ensure_project_dirs, get_project_paths, get_project_list
from tools_call_deepseek import access_deepseek


def list_worlds(project: str = DEFAULT_PROJECT):
    folder = get_project_paths(project)["worlds"]
    if not os.path.exists(folder):
        return []
    return sorted([f.replace(".txt", "") for f in os.listdir(folder) if f.endswith(".txt")])


def list_characters(project: str = DEFAULT_PROJECT):
    folder = get_project_paths(project)["characters"]
    chars = []
    if not os.path.exists(folder):
        return chars
    for f in sorted(os.listdir(folder)):
        if f.endswith(".json"):
            with open(os.path.join(folder, f), "r", encoding="utf-8") as fp:
                try:
                    data = json.load(fp)
                    chars.append({
                        "name": data.get("name", ""),
                        "intro": data.get("role_introduction", ""),
                        "personality": data.get("role_personality", ""),
                        "experience": data.get("role_experience", ""),
                    })
                except (json.JSONDecodeError, IOError) as e:
                    print(f"  ⚠️ 跳过 {f}: {e}")
    return chars


OUTLINE_PROMPT = """你是一位资深小说编辑和剧情架构师。请根据下面的世界观、角色和用户偏好，生成一份完整的章节大纲。

## 世界观
{world_text}

## 角色列表
{characters_text}

## 用户偏好
{preferences}

## 要求

1. 生成 {chapter_count} 章的大纲，每章包含：
   - 章节标题（4-8字）
   - 出场角色（从角色列表中选出本章出场的人物，标注谁是本章主角/焦点）
   - 核心事件（50-150字，简要描述本章发生的主要情节）
   - 角色发展（本章对哪些角色产生了什么影响）

2. 剧情设计原则：
   - 前 3 章建立角色关系和日常，让读者认识角色、喜欢角色
   - 第 4-6 章引入第一个主要冲突或转折
   - 第 7-8 章冲突升级，角色面临抉择
   - 最后 2 章收束、揭示真相、角色成长
   - 节奏要有张有弛，冲突章节之间穿插日常/情感章节作为缓冲
   - 给每个角色至少一次高光时刻
   - 不能连续两章缺少同一个核心角色（主角每章必出场）

3. 输出格式（严格按以下格式）：
```
第1章：章节标题
出场角色：角色A（主角）、角色B、角色C
核心事件：xxx
角色发展：xxx

第2章：章节标题
出场角色：...
核心事件：xxx
角色发展：xxx
...
```

注意：
- 标题要凝练，能反映本章核心
- 不要把结局写成"大团圆"，留有成长空间和续篇可能
- 如果世界观里有独特的规则或设定，确保剧情围绕这些设定展开
- 出场角色必须使用角色列表中给出的准确名称
"""


def build_characters_text(characters: list) -> str:
    lines = []
    for ch in characters:
        lines.append(
            f"- {ch['name']}：{ch['intro']}。性格：{ch['personality']}。"
            f"背景：{ch['experience']}"
        )
    return "\n".join(lines)


def generate_outline(project: str, world_name: str, preferences: str, chapter_count: int = 10):
    world_folder = get_project_paths(project)["worlds"]
    world_path = os.path.join(world_folder, f"{world_name}.txt")
    with open(world_path, "r", encoding="utf-8") as f:
        world_text = f.read()

    characters = list_characters(project)
    chars_text = build_characters_text(characters)

    prompt = OUTLINE_PROMPT.format(
        world_text=world_text,
        characters_text=chars_text,
        preferences=preferences,
        chapter_count=chapter_count,
    )

    result = access_deepseek(prompt, model_name="deepseek-v4-pro")
    return result


def save_outline(project: str, outline: str):
    folder = get_project_paths(project)["novel"]
    ensure_project_dirs(project)
    filepath = os.path.join(folder, "00_故事大纲.txt")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(outline)
    return filepath


def interactive(project: str = DEFAULT_PROJECT):
    print("=" * 60)
    print("  故事大纲生成工具")
    print("=" * 60)

    worlds = list_worlds(project)
    characters = list_characters(project)

    if not worlds:
        print(f"\n❌ 项目「{project}」中没有世界观文件")
        print(f"   请先运行: python generate_world.py --project {project}")
        return

    if not characters:
        print(f"\n❌ 项目「{project}」中没有角色卡")
        print(f"   请先运行: python generate_characters.py --project {project}")
        return

    print(f"\n📖 世界观: {worlds[0]}")
    print(f"🎭 角色数量: {len(characters)}")
    for ch in characters:
        print(f"   - {ch['name']} ({ch['personality']})")

    print("\n对故事有什么想法？比如：")
    print("  - 风格：热血成长 / 治愈日常 / 青春伤痛 / 悬疑推理")
    print("  - 核心主题：友情 / 梦想 / 背叛 / 救赎")
    print("  - 特殊要求：要有 livehouse 演出场景 / 主角团有人离开乐队")
    print("\n如果没想好，直接回车让我自由发挥。")
    preferences = input("\n> ").strip()
    if not preferences:
        preferences = "自由发挥，风格偏向青春成长与音乐梦想，日常与情感并重"

    chapter_str = input("\n需要几章大纲？(默认 10) : ").strip()
    try:
        chapter_count = int(chapter_str) if chapter_str else 10
    except ValueError:
        chapter_count = 10

    print(f"\n📝 正在生成 {chapter_count} 章大纲...\n")
    outline = generate_outline(project, worlds[0], preferences, chapter_count)

    print(outline)
    print("\n" + "=" * 60)

    save = input("\n保存大纲？(Y/n) : ").strip().lower()
    if save in ("n", "no"):
        print("已取消保存。")
        return

    filepath = save_outline(project, outline)
    print(f"\n✅ 大纲已保存: {filepath}")
    print(f"   写作时可以直接复制各章的「核心事件」到「本章要求」字段。")


def auto_generate(project: str = DEFAULT_PROJECT):
    print("🤖 自动模式\n")
    outline = generate_outline(
        project,
        list_worlds(project)[0],
        "自由发挥，青春成长题材，音乐主题",
        8,
    )
    print(outline)
    filepath = save_outline(project, outline)
    print(f"\n✅ 大纲已保存: {filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="故事大纲生成工具")
    parser.add_argument("--project", type=str, default=DEFAULT_PROJECT, help="小说项目名称")
    parser.add_argument("--auto", action="store_true", help="自动模式")
    args = parser.parse_args()

    if args.auto:
        auto_generate(args.project)
    else:
        interactive(args.project)
