#!/usr/bin/env python3
"""
角色卡生成工具 —— 基于已有世界观文件生成角色卡。

用法:
    python generate_characters.py           # 交互式流程
    python generate_characters.py --list    # 列出可用的世界观文件
    python generate_characters.py --auto    # 自动模式，用第一个世界观生成 3 个角色
"""

import os
import sys
import json
import argparse
from config import DEFAULT_PROJECT, ensure_project_dirs, get_project_paths
from tools_call_deepseek import access_deepseek
import json_repair

CHARACTER_PROMPT = """你是一位资深角色设计师。请根据下面的世界观和用户要求，设计角色卡。

## 世界观
{world_text}

## 角色要求
{user_requirements}

请生成 {count} 个角色的角色卡，每个角色包含以下字段：
- name: 角色名称（中文，2-4个字）
- role_introduction: 一句话简介（20-50字，说明角色身份和定位）
- role_experience: 角色背景故事（50-200字，说明角色的过往经历和动机）
- role_personality: 角色性格描述（10-50字）

要求：
- 角色之间要有关系网（师徒、对手、暗恋、宿敌等），不要各自孤立
- 性格要有冲突空间——两个性格相反的角色放在一起自然能产生剧情
- 至少有一个角色带着秘密或隐藏身份
- 角色动机要合理，不要为了推动剧情而强行安排行为

输出严格按以下 JSON 数组格式，不要输出任何其他内容：
```json
[
  {{
    "name": "...",
    "role_introduction": "...",
    "role_experience": "...",
    "role_personality": "..."
  }}
]
```
"""


def list_worlds(project: str = DEFAULT_PROJECT):
    folder = get_project_paths(project)["worlds"]
    worlds = []
    if not os.path.exists(folder):
        return worlds
    for filename in sorted(os.listdir(folder)):
        if filename.endswith(".txt"):
            worlds.append(filename.replace(".txt", ""))
    return worlds


def read_world(world_name: str, project: str = DEFAULT_PROJECT) -> str:
    folder = get_project_paths(project)["worlds"]
    filepath = os.path.join(folder, f"{world_name}.txt")
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def save_character(char_data: dict, project: str = DEFAULT_PROJECT) -> str:
    folder = get_project_paths(project)["characters"]
    ensure_project_dirs(project)
    name_for_file = char_data["name"].replace("/", "_")
    filepath = os.path.join(folder, f"{name_for_file}.json")

    if os.path.exists(filepath):
        base = os.path.splitext(filepath)[0]
        counter = 2
        while os.path.exists(f"{base}_{counter}.json"):
            counter += 1
        filepath = f"{base}_{counter}.json"

    char_data.setdefault("plot_history", [])
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(char_data, f, ensure_ascii=False, indent=4)
    return filepath


def generate_characters(project, world_name: str, requirements: str, count: int = 3) -> list:
    world_text = read_world(world_name, project)
    prompt = CHARACTER_PROMPT.format(
        world_text=world_text,
        user_requirements=requirements,
        count=count,
    )

    for attempt in range(3):
        try:
            raw = access_deepseek(prompt, model_name="deepseek-v4-pro", max_retry=1)
            parsed = json_repair.loads(raw)
            if isinstance(parsed, dict):
                parsed = [parsed]
            if not isinstance(parsed, list):
                raise ValueError("返回的不是数组格式")
            for ch in parsed:
                for field in ["name", "role_introduction", "role_experience", "role_personality"]:
                    if field not in ch:
                        raise ValueError(f"角色缺少必填字段: {field}")
            return parsed
        except Exception as e:
            print(f"  ⚠️ 第 {attempt+1} 次生成失败: {e}")
            if attempt == 2:
                raise
    return []


def interactive(project=DEFAULT_PROJECT):
    print("=" * 60)
    print("  角色卡生成工具")
    print("=" * 60)

    worlds = list_worlds(project)
    if not worlds:
        print(f"\n❌ 项目「{project}」中没有找到世界观文件。")
        print(f"   请先运行: python generate_world.py --project {project}")
        return

    print("\n可用的世界观文件:")
    for i, w in enumerate(worlds, 1):
        preview = read_world(w, project)[:80].replace("\n", " ")
        print(f"  {i}. {w}  ({preview}...)")

    choice = input(f"\n选择世界观编号 (1-{len(worlds)}) : ").strip()
    try:
        idx = int(choice) - 1
        if idx < 0 or idx >= len(worlds):
            raise ValueError
    except ValueError:
        print("无效选择，退出。")
        return

    world_name = worlds[idx]
    world_text = read_world(world_name, project)
    print(f"\n📖 已选择世界观: {world_name}")
    print(f"   预览: {world_text[:200]}...")

    count_str = input("\n需要几个角色？(默认 3) : ").strip()
    try:
        count = int(count_str) if count_str else 3
    except ValueError:
        count = 3

    print("\n对这些角色有什么要求？")
    print("例如: 需要一个主角（底层出身但天赋异禀）、一个导师型角色（曾经是大人物但隐退）、一个反派（有正当动机的反派）")
    print("也可以直接回车，我来帮你设计。")
    requirements = input("> ").strip()
    if not requirements:
        requirements = f"设计 {count} 个角色，覆盖主角、伙伴、对手三种类型，彼此之间有紧密的关系网"

    print(f"\n🎭 正在生成 {count} 个角色...")
    try:
        characters = generate_characters(project, world_name, requirements, count)
    except Exception:
        print("\n❌ 生成失败，请重试。")
        return

    print(f"\n✅ 生成了 {len(characters)} 个角色:\n")
    for i, ch in enumerate(characters, 1):
        print(f"  {i}. {ch['name']}")
        print(f"     简介: {ch['role_introduction']}")
        print(f"     性格: {ch['role_personality']}")
        print(f"     背景: {ch['role_experience'][:80]}...")
        print()

    confirm = input("是否保存到角色库？(Y/n) : ").strip().lower()
    if confirm in ("n", "no"):
        print("已取消保存。")
        return

    saved = []
    for ch in characters:
        try:
            filepath = save_character(ch, project)
            saved.append((ch["name"], filepath))
        except Exception as e:
            print(f"  ❌ 保存 {ch['name']} 失败: {e}")

    print(f"\n✅ 已保存 {len(saved)} 个角色:")
    for name, path in saved:
        print(f"   {name} → {path}")
    print(f"\n   可在 Gradio 界面「出场角色」下拉框中选中这些角色使用。")


def auto_generate():
    worlds = list_worlds(project)
    if not worlds:
        print("❌ 没有找到世界观文件，请先运行 python generate_world.py")
        return
    world_name = worlds[0]
    print(f"🤖 自动模式: 世界观 [{world_name}]，生成 3 个角色\n")
    characters = generate_characters(project, world_name, "设计主角、伙伴、对手各一个", 3)
    for ch in characters:
        filepath = save_character(ch, project)
        print(f"  ✅ {ch['name']} → {filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="角色卡生成工具")
    parser.add_argument("--list", action="store_true", help="列出可用世界观")
    parser.add_argument("--auto", action="store_true", help="自动模式")
    parser.add_argument("--project", type=str, default=DEFAULT_PROJECT, help="小说项目名称")
    args = parser.parse_args()

    if args.list:
        list_worlds(args.project)
    elif args.auto:
        auto_generate(args.project)
    else:
        interactive(args.project)
